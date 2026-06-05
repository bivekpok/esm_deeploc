"""
DDP-aware DeepLoc 2.0 *multilabel* training driver (paper-style 5-fold CV).

Cross-validation (``config.cv_scheme="paper"``, default)::

    For each fold *i* with validation partition *p* ∈ {0,…,4}:
        Train = 4 Swiss-Prot homology partitions (all except *p*)
        Val   = 1 whole partition (*p*) — early stopping + optional threshold tuning
        Test  = ``hpa_testset.csv`` ONLY (1717 proteins, never seen in training)

    Report mean ± std of HPA metrics across 5 folds. Pick the best hyperparameter
    config from CV val scores; the final numbers to cite are on HPA.

Per launch, artifacts land in a unique subdirectory::

    cv_results_deeploc_<run_tag>/<run_id>/Outer_Fold_k/
        best_model.pth / checkpoint.pt       # best val epoch
        last_model.pth / last_checkpoint.pt  # last epoch
        epoch_history.csv
        tuned_thresholds.json              # fit on val partition only
        hpa_test_predictions.csv             # external test (never used in training)
        run_summary.json

Launch::

    ./submit_multilabel_cv.sh          # reads experiment settings from config.py
    python train.py                    # same (config.py is the source of truth)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import wandb

from config import (
    ESMC_BACKBONE_PRESETS,
    apply_esmc_backbone,
    apply_run_snapshot_from_env,
    build_run_tag,
    config,
    resolve_output_dir,
    run_settings_dict,
    sync_run_paths,
)
from dataset import (
    CLASS_TO_IDX,
    DEEPLOC_CLASSES,
    ProteinMultilabelDataset,
    create_external_test_loader,
    create_fold_loaders,
    generate_partition_manifests,
    splits_meta_matches,
    write_external_test_manifest,
)
from losses import build_multilabel_criterion, resolve_focal_alpha
from model import (
    build_model,
    synchronize_esmc_classifier_batched_path,
)
from multilabel_train_diagnostics import print_multilabel_train_diag
from wandb_training_monitor import (
    MonitorTableLogger,
    collect_training_monitor_metrics,
    emit_monitor_metrics,
    finalize_monitor_table,
    setup_wandb_chart_metrics,
    should_log_monitor,
)
from utils import (
    all_gather_numpy,
    broadcast_int,
    broadcast_scalar,
    cleanup_ddp,
    compute_restricted_metrics,
    format_restricted_report,
    multilabel_predict,
    reduce_sum,
    set_seed,
    setup_ddp,
    tune_per_class_thresholds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_id() -> str:
    """Timestamp + short random hash; safe filesystem name."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    return f"{ts}-{uuid.uuid4().hex[:6]}"


def _resolve_run_id(
    cli_run_id: str | None,
    rank: int,
    world_size: int,
    device: torch.device,
) -> str:
    """Pick a run id on rank 0 (CLI > env > auto), then broadcast to all ranks."""
    if rank == 0:
        run_id = cli_run_id or os.environ.get("DEEPLOC_RUN_ID") or _make_run_id()
    else:
        run_id = ""
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        payload = [run_id]
        try:
            if device.type == "cuda":
                dist.broadcast_object_list(payload, src=0, device=device)
            else:
                dist.broadcast_object_list(payload, src=0)
        except TypeError:
            dist.broadcast_object_list(payload, src=0)
        run_id = payload[0]
    return run_id


def _freeze_esmc_sequence_head(esmc: nn.Module) -> None:
    """Freeze ESMC modules unused when pooling from per-layer hidden states.

    ``sequence_head`` only serves AA logits. ``transformer.norm`` is applied to
    the final representation for that head; batched embeddings average pre-norm
    layer outputs, so the final LayerNorm never receives gradients under DDP.
    """
    cur: nn.Module = esmc
    if hasattr(cur, "base_model"):
        cur = cur.base_model  # type: ignore[assignment]
    if hasattr(cur, "model"):
        cur = cur.model  # type: ignore[assignment]
    head = getattr(cur, "sequence_head", None)
    if head is not None:
        for p in head.parameters():
            p.requires_grad = False
    transformer = getattr(cur, "transformer", None)
    if transformer is not None:
        norm = getattr(transformer, "norm", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad = False


def _forward_collect(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    desc: str,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Forward pass on rank 0; returns (logits, labels, ids)."""
    model.eval()
    ids, lbls, lgs = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            labels = batch["labels"].to(device).float()
            logits, _ = model(batch)
            lgs.append(logits.float().cpu())
            lbls.append(labels.cpu())
            ids.extend(batch["ids"])

    if lgs:
        t_logits = torch.cat(lgs).numpy()
        t_labels = torch.cat(lbls).numpy()
    else:
        t_logits = np.zeros((0, num_classes))
        t_labels = np.zeros((0, num_classes))
    return t_logits, t_labels, ids


def _metrics_from(
    t_labels: np.ndarray,
    t_logits: np.ndarray,
    threshold,
    classes,
    desc: str,
    print_report: bool = True,
) -> Tuple[Dict[str, float], np.ndarray]:
    _, preds = multilabel_predict(t_logits, threshold=threshold, fallback_top1=True)
    metrics = {
        "macro_f1": float(f1_score(t_labels, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(t_labels, preds, average="micro", zero_division=0)),
        "subset_acc": float(accuracy_score(t_labels, preds)),
    }
    if print_report:
        print(f"\n--- {desc} ---")
        print(
            f"macro_f1={metrics['macro_f1']:.4f} micro_f1={metrics['micro_f1']:.4f} "
            f"subset_acc={metrics['subset_acc']:.4f}"
        )
        print(classification_report(t_labels, preds, target_names=classes, zero_division=0))
    return metrics, preds


def _run_eval(
    model: nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    classes,
    desc: str,
    threshold=None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, list]:
    """Backwards-compatible helper: forward + metrics at a single threshold."""
    t_logits, t_labels, ids = _forward_collect(model, loader, device, num_classes, desc)
    thr = config.prediction_threshold if threshold is None else threshold
    metrics, preds = _metrics_from(t_labels, t_logits, thr, classes, desc)
    return metrics, t_labels, preds, ids


def _join_row(row: np.ndarray, classes, sep: str) -> str:
    idxs = np.flatnonzero(row)
    return sep.join(classes[i] for i in idxs) if len(idxs) else ""


def build_checkpoint_meta(classes: list) -> Dict:
    """Hyperparameters needed to rebuild the model for eval / resume."""
    return {
        "model_name_or_path": config.model_name_or_path,
        "pooling_type": config.pooling_type,
        "use_lora_model": config.use_lora_model,
        "lora_last_n_layers": config.lora_last_n_layers,
        "lora_total_blocks": config.lora_total_blocks,
        "embed_dim": config.embed_dim,
        "classify_dropout": config.classify_dropout,
        "deeploc_classes": list(classes),
        "k_mer_size": config.k_mer_size,
        "bom_stride": config.bom_stride,
        "bom_summary": config.bom_summary,
        "bom_inner_dim": config.bom_inner_dim,
        "bom_output_dim": config.bom_output_dim,
        "bom_attn_k_mer_size": config.bom_attn_k_mer_size,
        "bom_attn_stride": config.bom_attn_stride,
        "bom_attn_inner_dim": config.bom_attn_inner_dim,
        "bom_attn_value_dim": config.bom_attn_value_dim,
        "bom_attn_output_dim": config.bom_attn_output_dim,
        "bom_attn_dropout": config.bom_attn_dropout,
        "prediction_threshold": config.prediction_threshold,
        "label_sep": config.label_sep,
        "max_len": config.max_len,
        "membrane_only": config.membrane_only,
        "layer_aggregation": config.layer_aggregation,
        "layer_agg_n": config.layer_agg_n,
        "layer_agg_band": list(config.layer_agg_band),
        "classifier_hidden_dim": getattr(config, "classifier_hidden_dim", 512),
        "loss_type": config.loss_type,
        "focal_gamma": config.focal_gamma,
        "focal_use_alpha": config.focal_use_alpha,
        "focal_alpha_mode": config.focal_alpha_mode,
        "focal_use_pos_weight": config.focal_use_pos_weight,
    }


def apply_checkpoint_meta(meta: Dict) -> None:
    """Apply keys from ``build_checkpoint_meta`` onto the global ``config``."""
    for key, value in meta.items():
        if key == "deeploc_classes":
            continue
        setattr(config, key, value)


def save_training_checkpoint(
    path: Path,
    raw_model: torch.nn.Module,
    *,
    fold_name: str,
    best_epoch_1based: int,
    best_score: float,
    classes: list,
) -> None:
    """Save ``state_dict`` plus metadata for ``test_checkpoint.py``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": raw_model.state_dict(),
            "fold_name": fold_name,
            "best_epoch": best_epoch_1based,
            "best_score": float(best_score),
            "early_stop_metric": config.early_stop_metric,
            "meta": build_checkpoint_meta(classes),
        },
        path,
    )


# ---------------------------------------------------------------------------
# Per-fold train + validate + test
# ---------------------------------------------------------------------------

def train_and_evaluate_fold(
    fold_name: str,
    fold_dir: Path,
    external_test_manifest: Path | None,
    device: torch.device,
    rank: int,
    world_size: int,
) -> Dict[str, Dict[str, float]]:
    is_main = rank == 0
    fold_out = Path(config.output_dir) / fold_name
    if is_main:
        fold_out.mkdir(parents=True, exist_ok=True)
    best_model_path = fold_out / "best_model.pth"
    last_model_path = fold_out / "last_model.pth"
    best_ckpt_path = fold_out / "checkpoint.pt"
    last_ckpt_path = fold_out / "last_checkpoint.pt"
    history_path = fold_out / "epoch_history.csv"

    train_loader, val_loader, train_ds = create_fold_loaders(
        fold_dir, rank=rank, world_size=world_size
    )
    classes = train_ds.classes
    num_classes = len(classes)
    if is_main:
        print_multilabel_train_diag(train_ds, fold_name, config.entropy_weight)
        val_csv = fold_dir / "Inner_Fold_1" / "valid_manifest.csv"
        val_parts = sorted(
            pd.read_csv(val_csv)["partition"].unique().tolist()
            if val_csv.is_file()
            else []
        )
        print(
            f"[{fold_name}] CV scheme={config.cv_scheme} "
            f"val_partition={val_parts} "
            f"(train={len(train_ds)} proteins from remaining partitions)",
            flush=True,
        )
        print(
            f"[{fold_name}] Early stop: metric={config.early_stop_metric} "
            f"patience={config.patience} min_delta={config.early_stop_min_delta} "
            f"| external test ONLY: {Path(config.test_csv).name}",
            flush=True,
        )

    model = build_model(num_classes, config.model_name_or_path, device=device)
    synchronize_esmc_classifier_batched_path(model, device, world_size)

    if is_main:
        backbone_mode = "LoRA (PEFT)" if config.use_lora_model else "Full fine-tune"
        forward_mode = (
            "batched (tokenizer + single forward)"
            if model.use_batched
            else "per-sequence (ESMProtein encode/logits)"
        )
        print(
            f"[{fold_name}] Backbone: {backbone_mode} | "
            f"Pooling: {config.pooling_type} | "
            f"ESMC forward: {forward_mode}",
            flush=True,
        )
        if config.use_lora_model and hasattr(model.esmc, "print_trainable_parameters"):
            model.esmc.print_trainable_parameters()

    _freeze_esmc_sequence_head(model.esmc)

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    raw_model = model.module if world_size > 1 else model
    criterion = build_multilabel_criterion(
        train_ds,
        device,
        loss_type=config.loss_type,
        focal_gamma=config.focal_gamma,
        focal_use_alpha=config.focal_use_alpha,
        focal_alpha_mode=config.focal_alpha_mode,
        focal_use_pos_weight=config.focal_use_pos_weight,
    )
    if is_main:
        loss_label = (config.loss_type or "bce").lower()
        if loss_label == "focal":
            if config.focal_use_alpha:
                alpha_mode = (config.focal_alpha_mode or "manual").lower()
                alpha_note = f"alpha={alpha_mode}"
            else:
                alpha_note = "no alpha"
            print(
                f"[{fold_name}] Loss: focal (gamma={config.focal_gamma}, {alpha_note})",
                flush=True,
            )
            if config.focal_use_alpha:
                alpha = resolve_focal_alpha(
                    train_ds, device, mode=config.focal_alpha_mode
                )
                for cls, a in zip(classes, alpha.tolist()):
                    print(f"[{fold_name}]   focal_alpha {cls}: {a:.4f}", flush=True)
        else:
            print(f"[{fold_name}] Loss: BCEWithLogits (sqrt pos_weight)", flush=True)

    esmc_trainable = [p for p in raw_model.esmc.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        [
            {"params": esmc_trainable, "lr": config.lr_esmc},
            {"params": raw_model.classifier.parameters(), "lr": config.lr_classifier},
        ],
        weight_decay=config.weight_decay,
    )
    maximize = config.early_stop_metric == "macro_f1"
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if maximize else "min",
        factor=0.5,
        patience=8,
    )

    best_score = float("-inf") if maximize else float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    monitor_table: MonitorTableLogger | None = MonitorTableLogger() if is_main else None
    log_monitor = bool(getattr(config, "wandb_log_monitor", True))
    monitor_as_charts = bool(getattr(config, "wandb_log_monitor_as_charts", False))

    for epoch in range(config.num_epochs):
        if (
            world_size > 1
            and train_loader.sampler is not None
            and hasattr(train_loader.sampler, "set_epoch")
        ):
            train_loader.sampler.set_epoch(epoch)

        # ---------------- Train ----------------
        model.train()
        train_loss_sum, train_n = 0.0, 0
        last_grad_norm: float | None = None
        iterator = tqdm(
            train_loader, desc=f"{fold_name} Ep{epoch+1} train", disable=not is_main
        )
        n_batches = len(train_loader)
        for batch_idx, batch in enumerate(iterator):
            labels = batch["labels"].to(device).float()
            optimizer.zero_grad()
            logits, entropy_loss = model(batch)
            logits = logits.float()
            loss = criterion(logits, labels) + config.entropy_weight * entropy_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if batch_idx == n_batches - 1:
                last_grad_norm = float(grad_norm)
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)
            train_n += labels.size(0)

        # ---------------- Validation ----------------
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        val_logits_list, val_labels_list = [], []
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"{fold_name} Ep{epoch+1} val", disable=not is_main
            ):
                labels = batch["labels"].to(device).float()
                logits, entropy_loss = model(batch)
                logits = logits.float()
                loss = criterion(logits, labels) + config.entropy_weight * entropy_loss
                val_loss_sum += loss.item() * labels.size(0)
                val_n += labels.size(0)
                val_logits_list.append(logits.cpu())
                val_labels_list.append(labels.cpu())

        train_stats = torch.tensor(
            [train_loss_sum, float(train_n)], dtype=torch.float64, device=device
        )
        train_stats = reduce_sum(train_stats, world_size)
        train_loss_mean = (train_stats[0] / train_stats[1].clamp(min=1)).item()

        val_stats = torch.tensor(
            [val_loss_sum, float(val_n)], dtype=torch.float64, device=device
        )
        val_stats = reduce_sum(val_stats, world_size)
        g_val_loss = (val_stats[0] / val_stats[1].clamp(min=1)).item()

        v_logits = (
            torch.cat(val_logits_list).numpy()
            if val_logits_list
            else np.zeros((0, num_classes))
        )
        v_labels = (
            torch.cat(val_labels_list).numpy()
            if val_labels_list
            else np.zeros((0, num_classes))
        )
        v_logits = all_gather_numpy(v_logits, world_size)
        v_labels = all_gather_numpy(v_labels, world_size)

        if is_main:
            _, preds = multilabel_predict(
                v_logits, threshold=config.prediction_threshold, fallback_top1=True
            )
            macro_f1 = float(f1_score(v_labels, preds, average="macro", zero_division=0))
            micro_f1 = float(f1_score(v_labels, preds, average="micro", zero_division=0))

            if maximize:
                score = macro_f1
                improved = score > best_score + config.early_stop_min_delta
            else:
                score = g_val_loss
                improved = score < best_score - config.early_stop_min_delta

            if improved:
                best_score = score
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(raw_model.state_dict(), best_model_path)
                save_training_checkpoint(
                    best_ckpt_path,
                    raw_model,
                    fold_name=fold_name,
                    best_epoch_1based=epoch + 1,
                    best_score=score,
                    classes=classes,
                )
            else:
                epochs_no_improve += 1

            # Always persist last-epoch checkpoint (best/last split).
            torch.save(raw_model.state_dict(), last_model_path)
            save_training_checkpoint(
                last_ckpt_path,
                raw_model,
                fold_name=fold_name,
                best_epoch_1based=epoch + 1,
                best_score=score,
                classes=classes,
            )

            # Per-epoch history (append to CSV).
            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss_mean,
                "val_loss": g_val_loss,
                "val_macro_f1": macro_f1,
                "val_micro_f1": micro_f1,
                "improved": int(improved),
                "best_epoch": best_epoch + 1 if best_epoch >= 0 else 0,
                "best_score": best_score,
                "epochs_no_improve": epochs_no_improve,
                "lr_esmc": optimizer.param_groups[0]["lr"],
                "lr_classifier": optimizer.param_groups[1]["lr"],
            }
            pd.DataFrame([row]).to_csv(
                history_path,
                mode="a",
                header=not history_path.exists(),
                index=False,
            )

            log_payload: Dict[str, object] = {
                f"{fold_name}/train_loss": train_loss_mean,
                f"{fold_name}/val_loss": g_val_loss,
                f"{fold_name}/val_macro_f1": macro_f1,
                f"{fold_name}/val_micro_f1": micro_f1,
                f"{fold_name}/epochs_no_improve": epochs_no_improve,
                f"{fold_name}/best_epoch": best_epoch + 1 if best_epoch >= 0 else 0,
                "epoch": epoch,
            }
            log_scalars, log_hists = should_log_monitor(
                epoch,
                getattr(config, "wandb_log_params_every", 1),
                getattr(config, "wandb_log_param_histogram_every", 10),
                getattr(config, "wandb_log_param_histograms", False),
            )
            if log_monitor and log_scalars:
                monitor = collect_training_monitor_metrics(
                    raw_model,
                    optimizer,
                    use_lora=config.use_lora_model,
                    total_grad_norm=last_grad_norm,
                    log_histograms=log_hists,
                )
                log_payload.update(
                    emit_monitor_metrics(
                        monitor,
                        fold_name,
                        epoch,
                        as_charts=monitor_as_charts,
                        table_logger=monitor_table,
                    )
                )
            wandb.log(log_payload)
            metric_tag = (
                f"macro_f1={macro_f1:.4f}"
                if maximize
                else f"val_loss={g_val_loss:.4f}"
            )
            tag = "  best" if improved else f"  ({epochs_no_improve}/{config.patience})"
            print(
                f"[{fold_name}] Epoch {epoch+1}: train_loss={train_loss_mean:.4f} "
                f"val_loss={g_val_loss:.4f} macro_f1={macro_f1:.4f} "
                f"micro_f1={micro_f1:.4f} [{metric_tag}]{tag}",
                flush=True,
            )
        else:
            macro_f1 = 0.0
            score = 0.0

        score = broadcast_scalar(score, src=0, device=device, world_size=world_size)
        scheduler.step(score)
        epochs_no_improve = broadcast_int(
            epochs_no_improve, src=0, device=device, world_size=world_size
        )
        if epochs_no_improve >= config.patience:
            if is_main:
                print(
                    f"[{fold_name}] Early stopping at epoch {epoch+1} "
                    f"(best {config.early_stop_metric} at epoch {best_epoch + 1}, "
                    f"patience={config.patience})",
                    flush=True,
                )
            break

    # ---------------- Test (rank 0 only) ----------------
    results: Dict[str, Dict[str, float]] = {}
    if is_main:
        if best_model_path.exists():
            raw_model.load_state_dict(torch.load(best_model_path, map_location=device))
        raw_model.eval()

        sep = config.label_sep
        ext_csv_name = (
            Path(config.test_csv).name if external_test_manifest is not None else None
        )

        # --- Per-class thresholds (F1 on val or MCC on train — see config) ---
        tuned_thresholds = None
        if config.tune_thresholds:
            thr_metric = (getattr(config, "threshold_tune_metric", "f1") or "f1").lower()
            thr_split = (getattr(config, "threshold_tune_split", "val") or "val").lower()
            if thr_split == "train":
                thr_loader = train_loader
                thr_desc = f"{fold_name} train (threshold tuning)"
            else:
                thr_loader = val_loader
                thr_desc = f"{fold_name} val (threshold tuning)"
            thr_logits_eval, thr_labels_eval, _ = _forward_collect(
                raw_model,
                thr_loader,
                device,
                num_classes,
                desc=thr_desc,
            )
            tuned_thresholds = tune_per_class_thresholds(
                thr_logits_eval,
                thr_labels_eval,
                default=config.prediction_threshold,
                metric=thr_metric,
            )
            with open(fold_out / "tuned_thresholds.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "classes": list(classes),
                        "threshold_tune_metric": thr_metric,
                        "threshold_tune_split": thr_split,
                        "thresholds_default": [config.prediction_threshold] * len(classes),
                        "thresholds_tuned": [float(x) for x in tuned_thresholds.tolist()],
                    },
                    f,
                    indent=2,
                )
            print(
                f"[{fold_name}] tuned per-class thresholds ({thr_metric} on {thr_split}): "
                + ", ".join(
                    f"{c}={t:.2f}" for c, t in zip(classes, tuned_thresholds.tolist())
                ),
                flush=True,
            )

        # HPA-aligned class subsets (see config docstring + Thumuluri et al. 2022).
        hpa_present_classes = list(
            getattr(config, "hpa_eval_classes_present", []) or []
        )
        hpa_paper_classes = list(
            getattr(config, "hpa_eval_classes_paper", []) or []
        )

        def _add_restricted(
            split_key: str,
            labels_eval: np.ndarray,
            preds: np.ndarray,
            subset_classes: list,
            subset_tag: str,
            desc_prefix: str,
        ) -> Dict[str, float] | None:
            """Compute + log paper-aligned restricted-class metrics."""
            if not subset_classes:
                return None
            sub = compute_restricted_metrics(
                labels_eval, preds, classes, subset_classes
            )
            results[f"{split_key}_{subset_tag}"] = sub
            print(
                format_restricted_report(
                    labels_eval,
                    preds,
                    classes,
                    subset_classes,
                    title=f"{desc_prefix} restricted to {subset_tag} ({sub['n_classes']} classes)",
                ),
                flush=True,
            )
            return sub

        def _eval_split(loader, *, split_key: str, desc_prefix: str, out_name: str | None = None):
            logits_eval, labels_eval, ids_eval = _forward_collect(
                raw_model, loader, device, num_classes, desc=desc_prefix
            )
            metrics_default, preds_default = _metrics_from(
                labels_eval,
                logits_eval,
                config.prediction_threshold,
                classes,
                desc=f"{desc_prefix} [thr=0.5]",
                print_report=True,
            )
            results[split_key] = metrics_default
            metrics_tuned = None
            preds_tuned = None
            if tuned_thresholds is not None:
                metrics_tuned, preds_tuned = _metrics_from(
                    labels_eval,
                    logits_eval,
                    tuned_thresholds,
                    classes,
                    desc=f"{desc_prefix} [tuned per-class thresholds]",
                    print_report=True,
                )
                results[f"{split_key}_tuned"] = metrics_tuned

            # Paper-aligned restricted metrics — only for HPA splits.
            sub_present_default = sub_present_tuned = None
            sub_paper_default = sub_paper_tuned = None
            if split_key.startswith("hpa"):
                sub_present_default = _add_restricted(
                    split_key, labels_eval, preds_default,
                    hpa_present_classes, "hpa8",
                    desc_prefix=f"{desc_prefix} [thr=0.5]",
                )
                sub_paper_default = _add_restricted(
                    split_key, labels_eval, preds_default,
                    hpa_paper_classes, "hpa6_paper",
                    desc_prefix=f"{desc_prefix} [thr=0.5]",
                )
                if preds_tuned is not None:
                    sub_present_tuned = _add_restricted(
                        f"{split_key}_tuned", labels_eval, preds_tuned,
                        hpa_present_classes, "hpa8",
                        desc_prefix=f"{desc_prefix} [tuned]",
                    )
                    sub_paper_tuned = _add_restricted(
                        f"{split_key}_tuned", labels_eval, preds_tuned,
                        hpa_paper_classes, "hpa6_paper",
                        desc_prefix=f"{desc_prefix} [tuned]",
                    )

            pred_rows = {
                "accession": ids_eval,
                "true_labels": [
                    _join_row(labels_eval[i], classes, sep) for i in range(len(labels_eval))
                ],
                "pred_labels_default": [
                    _join_row(preds_default[i], classes, sep) for i in range(len(preds_default))
                ],
            }
            if preds_tuned is not None:
                pred_rows["pred_labels_tuned"] = [
                    _join_row(preds_tuned[i], classes, sep) for i in range(len(preds_tuned))
                ]
            csv_name = out_name or f"{split_key}_predictions.csv"
            pd.DataFrame(pred_rows).to_csv(fold_out / csv_name, index=False)

            log_payload = {
                f"{fold_name}/{split_key}_macro_f1": metrics_default["macro_f1"],
                f"{fold_name}/{split_key}_micro_f1": metrics_default["micro_f1"],
                f"{fold_name}/{split_key}_subset_acc": metrics_default["subset_acc"],
            }
            if metrics_tuned is not None:
                log_payload.update({
                    f"{fold_name}/{split_key}_tuned_macro_f1": metrics_tuned["macro_f1"],
                    f"{fold_name}/{split_key}_tuned_micro_f1": metrics_tuned["micro_f1"],
                    f"{fold_name}/{split_key}_tuned_subset_acc": metrics_tuned["subset_acc"],
                })

            def _add_sub(prefix: str, sub: Dict[str, float] | None) -> None:
                if not sub:
                    return
                log_payload[f"{fold_name}/{prefix}_macro_f1"] = sub["macro_f1"]
                log_payload[f"{fold_name}/{prefix}_micro_f1"] = sub["micro_f1"]
                log_payload[f"{fold_name}/{prefix}_macro_mcc"] = sub["macro_mcc"]

            _add_sub(f"{split_key}_hpa8", sub_present_default)
            _add_sub(f"{split_key}_hpa6_paper", sub_paper_default)
            _add_sub(f"{split_key}_tuned_hpa8", sub_present_tuned)
            _add_sub(f"{split_key}_tuned_hpa6_paper", sub_paper_tuned)

            wandb.log(log_payload)

        # Val partition (for CV model selection — not reported as "test").
        _eval_split(
            val_loader,
            split_key="val",
            desc_prefix=f"{fold_name} val partition",
            out_name="val_predictions.csv",
        )

        # External test: HPA only (never used during training or threshold tuning).
        if external_test_manifest is not None and external_test_manifest.exists():
            ext_loader, _ = create_external_test_loader(
                external_test_manifest, class_to_idx=dict(CLASS_TO_IDX)
            )
            _eval_split(
                ext_loader,
                split_key="hpa_test",
                desc_prefix=f"{fold_name} HPA external test ({ext_csv_name})",
                out_name="hpa_test_predictions.csv",
            )

        # Persist per-fold summary so the unique run dir is self-contained.
        with open(fold_out / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fold_name": fold_name,
                    "best_epoch": best_epoch + 1 if best_epoch >= 0 else 0,
                    "best_score": float(best_score) if best_score != float("inf") else None,
                    "early_stop_metric": config.early_stop_metric,
                    "tune_thresholds": bool(config.tune_thresholds),
                    "external_test_csv": str(config.test_csv),
                    "metrics": results,
                },
                f,
                indent=2,
            )

        if monitor_table is not None and monitor_table.rows:
            pd.DataFrame(
                monitor_table.rows, columns=list(MonitorTableLogger._COLUMNS)
            ).to_csv(fold_out / "monitor_history.csv", index=False)
        finalize_monitor_table(
            fold_name, monitor_table, as_charts=monitor_as_charts
        )

    if world_size > 1:
        dist.barrier()
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regenerate-splits",
        action="store_true",
        help="Rewrite manifests under config.splits_root before training.",
    )
    parser.add_argument(
        "--data-summary-only",
        action="store_true",
        help="Only print per-fold diagnostics, then exit (no training).",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Use PEFT LoRA adapters on the ESMC backbone (only adapters + head train).",
    )
    parser.add_argument(
        "--lora-last-n",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override config.lora_last_n_layers: adapt only the last N transformer blocks "
            "(explicit module paths). Use 0 for all blocks (short target_modules patterns)."
        ),
    )
    parser.add_argument(
        "--pooling",
        choices=("attention", "average", "bom", "bom_attn"),
        default=None,
        help="Override config.pooling_type.",
    )
    parser.add_argument(
        "--membrane-only",
        action="store_true",
        help="Filter both train CSV and external test CSV to Membrane>=0.5 rows.",
    )
    parser.add_argument(
        "--no-external-test",
        action="store_true",
        help="Skip evaluation on config.test_csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override config.output_dir for checkpoints and predictions.",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Short experiment tag for WandB run names (set by run_multilabel_cv.sh).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early-stop patience in epochs (default: config.patience).",
    )
    parser.add_argument(
        "--early-stop-metric",
        choices=("val_loss", "macro_f1"),
        default=None,
        help="Metric for checkpointing and early stopping (default: val_loss).",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=None,
        help="Minimum improvement to reset early-stop counter.",
    )
    parser.add_argument(
        "--layer-aggregation",
        choices=("all", "last_n", "band"),
        default=None,
        help="Override config.layer_aggregation for residue-vector pooling.",
    )
    parser.add_argument(
        "--layer-agg-n",
        type=int,
        default=None,
        help="Last-N block outputs to mean over when --layer-aggregation last_n.",
    )
    parser.add_argument(
        "--layer-agg-band",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Block-output range [START END] (inclusive, 0-indexed) for --layer-aggregation band.",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Comma-separated outer fold indices (1-based) to run, e.g. '1' or '1,3'.",
    )
    parser.add_argument(
        "--cv-scheme",
        choices=("paper", "holdout", "random"),
        default=None,
        help=(
            "CV split scheme. 'paper' (default): train 4 partitions, val 1 "
            "whole partition, HPA-only external test. 'holdout'/'random' are legacy."
        ),
    )
    parser.add_argument(
        "--inner-val-strategy",
        choices=("paper", "holdout", "random", "partition"),
        default=None,
        help="Deprecated alias for --cv-scheme (partition -> holdout).",
    )
    parser.add_argument(
        "--no-tune-thresholds",
        action="store_true",
        help="Disable per-class threshold tuning on val after training.",
    )
    parser.add_argument(
        "--backbone",
        choices=tuple(ESMC_BACKBONE_PRESETS.keys()),
        default=None,
        help=(
            "ESMC variant preset (sets model_name_or_path + embed_dim). "
            "600M outputs go to cv_results_deeploc_esmc600_*."
        ),
    )
    parser.add_argument(
        "--loss",
        choices=("bce", "focal"),
        default=None,
        help="Multilabel loss (default: config.loss_type).",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=None,
        help="Focal loss gamma when --loss focal (default: config.focal_gamma).",
    )
    parser.add_argument(
        "--no-focal-alpha",
        action="store_true",
        help="Disable per-class inverse-frequency alpha for focal loss.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Explicit run id (subfolder under output_dir). Default: auto "
            "(timestamp + short hash). Each launch gets a unique dir."
        ),
    )
    args = parser.parse_args()

    # Optional CLI overrides (debug / one-off). Normal Slurm runs use config.py only.
    if args.backbone is not None:
        apply_esmc_backbone(args.backbone)
    if args.loss is not None:
        config.loss_type = args.loss
    if args.focal_gamma is not None:
        config.focal_gamma = float(args.focal_gamma)
    if args.no_focal_alpha:
        config.focal_use_alpha = False
    if args.lora:
        config.use_lora_model = True
    if args.lora_last_n is not None:
        config.lora_last_n_layers = None if args.lora_last_n <= 0 else int(args.lora_last_n)
    if args.pooling is not None:
        config.pooling_type = args.pooling
    if args.membrane_only:
        config.membrane_only = True
    if args.patience is not None:
        config.patience = int(args.patience)
    if args.early_stop_metric is not None:
        config.early_stop_metric = args.early_stop_metric
    if args.early_stop_min_delta is not None:
        config.early_stop_min_delta = float(args.early_stop_min_delta)
    if args.layer_aggregation is not None:
        config.layer_aggregation = args.layer_aggregation
    if args.layer_agg_n is not None:
        config.layer_agg_n = int(args.layer_agg_n)
    if args.layer_agg_band is not None:
        config.layer_agg_band = (int(args.layer_agg_band[0]), int(args.layer_agg_band[1]))
    if args.folds is not None:
        config.folds_to_run = [int(x) for x in args.folds.split(",") if x.strip()]
    if args.cv_scheme is not None:
        config.cv_scheme = args.cv_scheme
    elif args.inner_val_strategy is not None:
        legacy = args.inner_val_strategy
        config.cv_scheme = "holdout" if legacy == "partition" else legacy
    if args.no_tune_thresholds:
        config.tune_thresholds = False

    if apply_run_snapshot_from_env():
        pass
    elif os.environ.get("RUN_TAG", "").strip() and os.environ.get("OUTPUT_DIR", "").strip():
        config.run_tag = os.environ["RUN_TAG"].strip()
        config.output_dir = os.environ["OUTPUT_DIR"].strip()
    else:
        if args.run_tag:
            config.run_tag = args.run_tag
        if args.output_dir is not None:
            config.output_dir = args.output_dir
        else:
            sync_run_paths()

    regenerate_splits = config.regenerate_splits or args.regenerate_splits
    data_summary_only = config.data_summary_only or args.data_summary_only
    no_external_test = (not config.eval_external_test) or args.no_external_test

    set_seed(config.random_state)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    local_rank, rank, world_size, device = setup_ddp()

    # Generate a per-launch unique run id, then broadcast to all ranks so they
    # all land in the same output subfolder. Each new launch -> new folder,
    # so concurrent / repeated runs never overwrite each other's checkpoints.
    run_id = _resolve_run_id(args.run_id, rank, world_size, device)
    config.run_id = run_id
    base_out = Path(config.output_dir)
    config.output_dir = str(base_out / run_id)

    if rank == 0:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        # Convenience pointer to the most recent launch for this run_tag.
        latest_link = base_out / "LATEST"
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(run_id)
        except OSError:
            pass
        run_meta = run_settings_dict()
        run_meta["started_at_utc"] = datetime.now(timezone.utc).isoformat()
        run_meta["cli_argv"] = " ".join(sys.argv[1:])
        run_meta["run_id"] = run_id
        run_meta["output_dir"] = config.output_dir
        run_meta["task"] = "multilabel-classification"
        with open(Path(config.output_dir) / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)
        print(
            f"[run] tag={run_meta['run_tag']} run_id={run_id} "
            f"membrane_only={config.membrane_only} lora={config.use_lora_model} "
            f"pooling={config.pooling_type} cv_scheme={config.cv_scheme} "
            f"tune_thr={config.tune_thresholds} "
            f"external_test={Path(config.test_csv).name} -> {config.output_dir}",
            flush=True,
        )
    if world_size > 1:
        dist.barrier()

    splits_root = Path(config.splits_root)
    marker = splits_root / "Outer_Fold_1" / "Inner_Fold_1" / "train_manifest.csv"
    splits_ok = splits_meta_matches(
        splits_root,
        cv_scheme=config.cv_scheme,
        n_outer=config.n_outer_folds,
        inner_val_fraction=config.inner_val_fraction,
        random_state=config.random_state,
        membrane_only=config.membrane_only,
    )
    need_regen = regenerate_splits or (not marker.exists()) or (not splits_ok)
    if rank == 0 and need_regen:
        if not splits_ok and marker.exists():
            print(
                f"[splits] existing manifests don't match cv_scheme={config.cv_scheme} / "
                f"membrane_only={config.membrane_only}; regenerating.",
                flush=True,
            )
        print(f"Generating partition manifests under {splits_root} ...", flush=True)
        generate_partition_manifests(
            Path(config.label_csv),
            splits_root,
            n_outer=config.n_outer_folds,
            inner_val_fraction=config.inner_val_fraction,
            random_state=config.random_state,
            cv_scheme=config.cv_scheme,
        )
    if world_size > 1:
        dist.barrier()

    external_manifest: Path | None = None
    if not no_external_test:
        external_manifest = splits_root / "external_test_manifest.csv"
        if rank == 0 and (regenerate_splits or not external_manifest.exists()):
            write_external_test_manifest(Path(config.test_csv), external_manifest)
        if world_size > 1:
            dist.barrier()

    fold_dirs = sorted(
        d for d in splits_root.iterdir() if d.is_dir() and d.name.startswith("Outer_Fold_")
    )
    if config.folds_to_run:
        keep = {int(i) for i in config.folds_to_run}
        fold_dirs = [d for d in fold_dirs if int(d.name.split("_")[-1]) in keep]
    if rank == 0:
        print(f"Folds: {[d.name for d in fold_dirs]}", flush=True)

    if data_summary_only:
        if rank == 0:
            for fold_dir in fold_dirs:
                train_csv = fold_dir / "Inner_Fold_1" / "train_manifest.csv"
                ds = ProteinMultilabelDataset(
                    str(train_csv),
                    class_to_idx=dict(CLASS_TO_IDX),
                    max_len=config.max_len,
                    test_mode=config.test_mode,
                    label_sep=config.label_sep,
                )
                print_multilabel_train_diag(ds, fold_dir.name, config.entropy_weight)
        if world_size > 1:
            dist.barrier()
        cleanup_ddp()
        return

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        if rank == 0:
            print(f"\n{'='*60}\n{fold_name}\n{'='*60}", flush=True)
            run_tag = config.run_tag or build_run_tag()
            wandb_run_name = f"{fold_name}_{run_tag}"
            wandb.init(
                project=config.wandb_project,
                name=wandb_run_name,
                config={
                    k: v
                    for k, v in vars(config).items()
                    if not k.startswith("_") and not callable(v)
                },
                reinit=True,
                mode=getattr(config, "wandb_mode", "online"),
            )
            setup_wandb_chart_metrics(fold_name)
        if world_size > 1:
            dist.barrier()

        fold_results = train_and_evaluate_fold(
            fold_name,
            fold_dir,
            external_test_manifest=external_manifest,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        if rank == 0:
            results[fold_name] = fold_results
            wandb.finish()

    if rank == 0 and results:
        print("\n=== CV SUMMARY (paper scheme) ===", flush=True)
        print(f"External test: {Path(config.test_csv).name} (never used in training)", flush=True)
        for fname, r in results.items():
            parts = []
            for key, m in sorted(r.items()):
                parts.append(
                    f"{key}: macro_f1={m['macro_f1']:.4f} "
                    f"micro_f1={m['micro_f1']:.4f} subset_acc={m['subset_acc']:.4f}"
                )
            print(f"{fname}: " + " | ".join(parts))

        def _mean_std(key: str) -> None:
            macros = [r[key]["macro_f1"] for r in results.values() if key in r]
            micros = [r[key]["micro_f1"] for r in results.values() if key in r]
            if not macros:
                return
            print(
                f"Mean {key} macro F1: {np.mean(macros):.4f} ± {np.std(macros):.4f}  "
                f"(micro F1: {np.mean(micros):.4f} ± {np.std(micros):.4f}, n={len(macros)} folds)",
                flush=True,
            )

        _mean_std("hpa_test")
        _mean_std("hpa_test_tuned")
        _mean_std("hpa_test_hpa8")
        _mean_std("hpa_test_tuned_hpa8")
        _mean_std("hpa_test_hpa6_paper")
        _mean_std("hpa_test_tuned_hpa6_paper")
        _mean_std("val")
        _mean_std("val_tuned")

        with open(Path(config.output_dir) / "cv_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": config.run_id,
                    "tag": build_run_tag(),
                    "cv_scheme": config.cv_scheme,
                    "external_test_csv": str(config.test_csv),
                    "tune_thresholds": bool(config.tune_thresholds),
                    "folds": results,
                    "hpa_test_macro_f1_mean": float(np.mean(
                        [r["hpa_test"]["macro_f1"] for r in results.values() if "hpa_test" in r]
                    )) if any("hpa_test" in r for r in results.values()) else None,
                    "hpa_test_macro_f1_std": float(np.std(
                        [r["hpa_test"]["macro_f1"] for r in results.values() if "hpa_test" in r]
                    )) if any("hpa_test" in r for r in results.values()) else None,
                },
                f,
                indent=2,
            )

    cleanup_ddp()


if __name__ == "__main__":
    main()
