"""
Load a training checkpoint (``checkpoint.pt`` with ``state_dict`` + ``meta``)
or a raw ``best_model.pth`` state_dict, then evaluate on the external test manifest.

Typical flow (after CV training):
  1. Train writes per-fold ``Outer_Fold_*/checkpoint.pt`` (full dict) and
     ``best_model.pth`` (weights only).
  2. Run this script pointing at ``checkpoint.pt`` so pooling / LoRA flags match.

Examples:
  # eval on bundled test manifest (from ``config.test_csv`` via splits)
  python test_checkpoint.py \\
    --checkpoint cv_results_deeploc_full_bom_attn/Outer_Fold_1/checkpoint.pt

  # raw state dict — you must match training ``config.py`` / CLI yourself
  python test_checkpoint.py --checkpoint path/to/best_model.pth --weights-only

  # custom manifest CSV (same schema as ``external_test_manifest.csv``)
  python test_checkpoint.py --checkpoint checkpoint.pt \\
    --manifest cv_splits_deeploc/external_test_manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

from config import config
from dataset import create_external_test_loader, write_external_test_manifest
from model import build_model
from train import apply_checkpoint_meta
from utils import multilabel_predict, set_seed


def _join_row(row: np.ndarray, classes: List[str], sep: str) -> str:
    idxs = np.flatnonzero(row)
    return sep.join(classes[i] for i in idxs) if len(idxs) else ""


def run_eval(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int,
    classes: List[str],
    desc: str,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, list]:
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

    _, preds = multilabel_predict(
        t_logits, threshold=config.prediction_threshold, fallback_top1=True
    )
    metrics = {
        "macro_f1": float(f1_score(t_labels, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(t_labels, preds, average="micro", zero_division=0)),
        "subset_acc": float(accuracy_score(t_labels, preds)),
    }
    print(f"\n--- {desc} ---")
    print(
        f"macro_f1={metrics['macro_f1']:.4f} micro_f1={metrics['micro_f1']:.4f} "
        f"subset_acc={metrics['subset_acc']:.4f}"
    )
    print(classification_report(t_labels, preds, target_names=classes, zero_division=0))
    return metrics, t_labels, preds, ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Load checkpoint and eval on test manifest.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint.pt (dict) or best_model.pth (state_dict only).",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Treat file as a raw state_dict; do not load meta (use current config).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to external test manifest CSV. Default: splits_root/external_test_manifest.csv",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=None,
        help="If set, rebuild manifest from this CSV (e.g. test_label.csv) then evaluate.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save predictions CSV.",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto).")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)
    if args.weights_only or not isinstance(blob, dict) or "state_dict" not in blob:
        state_dict = blob if not isinstance(blob, dict) else blob.get("state_dict", blob)
        print(
            "[test_checkpoint] Loading weights only; using *current* config.py — "
            "ensure pooling_type / LoRA match training.",
            flush=True,
        )
    else:
        meta = blob["meta"]
        apply_checkpoint_meta(meta)
        if "deeploc_classes" in meta:
            config.deeploc_classes = list(meta["deeploc_classes"])
        state_dict = blob["state_dict"]
        print(
            f"[test_checkpoint] Loaded checkpoint meta: "
            f"fold={blob.get('fold_name')}, best_epoch={blob.get('best_epoch')}, "
            f"pooling={config.pooling_type}",
            flush=True,
        )

    set_seed(config.random_state)

    classes = list(config.deeploc_classes)
    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    model = build_model(num_classes, config.model_name_or_path, device=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[test_checkpoint] load_state_dict: missing={missing}, unexpected={unexpected}")
    model.eval()

    splits_root = Path(config.splits_root)
    if args.test_csv:
        manifest_path = splits_root / "external_test_manifest_infer.csv"
        write_external_test_manifest(Path(args.test_csv), manifest_path)
    elif args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = splits_root / "external_test_manifest.csv"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} not found. Pass --manifest or --test-csv, or run train.py once."
        )

    loader, _ = create_external_test_loader(
        manifest_path,
        class_to_idx=class_to_idx,
        max_len=config.max_len,
        label_sep=config.label_sep,
    )

    metrics, t_labels, preds, ids = run_eval(
        model,
        loader,
        device,
        num_classes,
        classes,
        desc=f"checkpoint eval ({manifest_path.name})",
    )

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        sep = config.label_sep
        pd.DataFrame(
            {
                "accession": ids,
                "true_labels": [
                    _join_row(t_labels[i], classes, sep) for i in range(len(t_labels))
                ],
                "pred_labels": [
                    _join_row(preds[i], classes, sep) for i in range(len(preds))
                ],
            }
        ).to_csv(out, index=False)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
