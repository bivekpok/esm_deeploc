"""
Evaluate a trained checkpoint on ``hpa_testset.csv`` (or any DeepLoc-style test CSV).

Best weights from ``train.py`` are saved per fold as:
  {output_dir}/{fold_name}/best_model.pth   — state_dict only
  {output_dir}/{fold_name}/checkpoint.pt    — state_dict + training meta (preferred)

Example:
  python test_hpa.py \\
    --checkpoint cv_results_deeploc_lora_loran5_bom_attn/Outer_Fold_1/checkpoint.pt \\
    --test-csv hpa_testset.csv \\
    --output-dir hpa_eval_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from config import config
from dataset import create_external_test_loader, write_external_test_manifest
from model import build_model
from test_checkpoint import _join_row
from train import apply_checkpoint_meta
from utils import (  # noqa: F401
    compute_restricted_metrics,
    format_restricted_report,
    multilabel_predict,
    set_seed,
    tune_per_class_thresholds,
)


def load_checkpoint(ckpt_path: Path, device: torch.device, weights_only: bool) -> torch.nn.Module:
    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)

    if weights_only or not isinstance(blob, dict) or "state_dict" not in blob:
        state_dict = blob if not isinstance(blob, dict) else blob.get("state_dict", blob)
        print(
            "[test_hpa] Loading weights only; ensure config matches training.",
            flush=True,
        )
    else:
        meta = blob["meta"]
        apply_checkpoint_meta(meta)
        if "deeploc_classes" in meta:
            config.deeploc_classes = list(meta["deeploc_classes"])
        state_dict = blob["state_dict"]
        print(
            f"[test_hpa] fold={blob.get('fold_name')} "
            f"best_epoch={blob.get('best_epoch')} "
            f"best_score={blob.get('best_score')} "
            f"pooling={config.pooling_type}",
            flush=True,
        )

    classes = list(config.deeploc_classes)
    model = build_model(len(classes), config.model_name_or_path, device=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[test_hpa] load_state_dict: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Return overall metrics, per-class table, and per-class 2x2 confusion matrices."""
    overall = {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "samples_jaccard": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
    }

    report = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    per_class_rows = []
    for cls in classes:
        stats = report.get(cls, {})
        per_class_rows.append(
            {
                "class": cls,
                "support": int(stats.get("support", 0)),
                "precision": float(stats.get("precision", 0.0)),
                "recall": float(stats.get("recall", 0.0)),
                "f1": float(stats.get("f1-score", 0.0)),
            }
        )
    per_class_df = pd.DataFrame(per_class_rows)

    cm_rows = []
    for i, cls in enumerate(classes):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        cm_rows.append(
            {
                "class": cls,
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            }
        )
    confusion_df = pd.DataFrame(cm_rows)
    return overall, per_class_df, confusion_df


def run_inference(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    threshold=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    model.eval()
    ids, lbls, lgs = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="HPA test inference"):
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

    thr = config.prediction_threshold if threshold is None else threshold
    probs, preds = multilabel_predict(t_logits, threshold=thr, fallback_top1=True)
    return t_labels, preds, probs, ids


def maybe_load_tuned_thresholds(checkpoint_path: Path, n_classes: int) -> Optional[np.ndarray]:
    """Look for ``tuned_thresholds.json`` next to the checkpoint."""
    cand = checkpoint_path.parent / "tuned_thresholds.json"
    if not cand.is_file():
        return None
    try:
        with open(cand, "r", encoding="utf-8") as f:
            payload = json.load(f)
        thr = np.asarray(payload["thresholds_tuned"], dtype=np.float64)
        if thr.size != n_classes:
            print(
                f"[test_hpa] {cand} has {thr.size} thresholds, expected {n_classes}; ignoring."
            )
            return None
        return thr
    except Exception as e:
        print(f"[test_hpa] could not parse {cand}: {e}")
        return None


def print_results(
    overall: Dict[str, float],
    per_class_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    classes: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    print("\n=== Overall metrics ===")
    for key, value in overall.items():
        print(f"  {key}: {value:.4f}")

    print("\n=== Per-class metrics ===")
    print(per_class_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Per-class confusion matrices (TN, FP, FN, TP) ===")
    print(confusion_df.to_string(index=False))

    print("\n=== sklearn classification report ===")
    print(
        classification_report(
            y_true, y_pred, target_names=classes, zero_division=0, digits=4
        )
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on HPA / external test CSV.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint.pt (preferred) or best_model.pth.",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Treat checkpoint as raw state_dict (no meta).",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=str(root / "hpa_testset.csv"),
        help="DeepLoc-style test CSV (default: hpa_testset.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(root / "hpa_eval_results"),
        help="Directory for metrics CSV/JSON and predictions.",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto).")
    parser.add_argument(
        "--use-tuned-thresholds",
        action="store_true",
        help=(
            "If ``tuned_thresholds.json`` exists next to the checkpoint, use "
            "those per-class thresholds instead of the default 0.5."
        ),
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    test_csv = Path(args.test_csv)
    if not test_csv.is_file():
        raise FileNotFoundError(test_csv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    set_seed(config.random_state)

    model = load_checkpoint(ckpt_path, device, weights_only=args.weights_only)
    classes = list(config.deeploc_classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    manifest_path = Path(config.splits_root) / "hpa_test_manifest.csv"
    write_external_test_manifest(test_csv, manifest_path)
    loader, _ = create_external_test_loader(
        manifest_path,
        class_to_idx=class_to_idx,
        max_len=config.max_len,
        label_sep=config.label_sep,
    )

    threshold = None
    if args.use_tuned_thresholds:
        tuned = maybe_load_tuned_thresholds(ckpt_path, len(classes))
        if tuned is not None:
            threshold = tuned
            print(
                "[test_hpa] using tuned per-class thresholds: "
                + ", ".join(f"{c}={t:.2f}" for c, t in zip(classes, tuned.tolist()))
            )
        else:
            print(
                "[test_hpa] --use-tuned-thresholds set but no tuned_thresholds.json "
                "found next to the checkpoint; falling back to default 0.5."
            )

    y_true, y_pred, probs, ids = run_inference(
        model, loader, device, len(classes), threshold=threshold
    )
    overall, per_class_df, confusion_df = compute_multilabel_metrics(y_true, y_pred, classes)
    print_results(overall, per_class_df, confusion_df, classes, y_true, y_pred)

    # ---- Paper-aligned restricted-class evaluation (HPA-only) -----------
    # DeepLoc 2.0 (Thumuluri et al. 2022, Table 2) trains on 10 classes but
    # reports HPA metrics over present classes only — Plastid (no plastids
    # in animals) and Extracellular (washed away in IF prep) are absent
    # from the HPA assay, so including them as 0-support drags macro F1.
    present = list(getattr(config, "hpa_eval_classes_present", []) or [])
    paper = list(getattr(config, "hpa_eval_classes_paper", []) or [])

    restricted_overall: Dict[str, Dict[str, float]] = {
        "all10": overall,
    }
    if present:
        sub8 = compute_restricted_metrics(y_true, y_pred, classes, present)
        restricted_overall["hpa8_present"] = sub8
        print(
            format_restricted_report(
                y_true, y_pred, classes, present,
                title="HPA — 8 present classes (Plastid & Extracellular excluded)",
            ),
            flush=True,
        )
    if paper:
        sub6 = compute_restricted_metrics(y_true, y_pred, classes, paper)
        restricted_overall["hpa6_paper"] = sub6
        print(
            format_restricted_report(
                y_true, y_pred, classes, paper,
                title="HPA — 6 paper classes (DeepLoc 2.0 Table 2 subset)",
            ),
            flush=True,
        )

    sep = config.label_sep
    pd.DataFrame(
        {
            "accession": ids,
            "true_labels": [_join_row(y_true[i], classes, sep) for i in range(len(y_true))],
            "pred_labels": [_join_row(y_pred[i], classes, sep) for i in range(len(y_pred))],
        }
    ).to_csv(out_dir / "predictions.csv", index=False)

    per_class_df.to_csv(out_dir / "per_class_metrics.csv", index=False)
    confusion_df.to_csv(out_dir / "per_class_confusion_matrices.csv", index=False)
    with open(out_dir / "overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)
    with open(out_dir / "restricted_metrics.json", "w", encoding="utf-8") as f:
        json.dump(restricted_overall, f, indent=2)

    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
