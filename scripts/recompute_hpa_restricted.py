"""
Recompute paper-aligned (8-class + 6-class) HPA metrics from already-saved
prediction CSVs — no GPU / model reload needed.

Why: DeepLoc 2.0 (Thumuluri et al. 2022) trains on 10 classes but reports
HPA metrics over the classes physically present in the HPA assay. HPA is
human-only so Plastid is biologically absent, and Extracellular (secreted)
proteins are washed away in immunofluorescence prep. Computing macro F1
over all 10 classes therefore unfairly drags the score with structural
zeros. This script slices saved predictions to the present-class subsets
and reports the comparable metrics, leaving full 10-class numbers intact.

Usage:
    python scripts/recompute_hpa_restricted.py            # all runs under cv_results_*
    python scripts/recompute_hpa_restricted.py PATH ...   # one or more run/Outer_Fold_* dirs
    python scripts/recompute_hpa_restricted.py --csv path/to/predictions.csv

Outputs a comparison table to stdout AND writes
``hpa_restricted_metrics.json`` next to each predictions CSV it inspects.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import config  # noqa: E402
from dataset import parse_label_field  # noqa: E402
from utils import compute_restricted_metrics  # noqa: E402


PRED_NAMES = (
    "hpa_test_predictions.csv",
    "external_test_predictions.csv",  # legacy filename
)


def find_prediction_csvs(targets: List[Path]) -> List[Path]:
    found: List[Path] = []
    for t in targets:
        if t.is_file() and t.suffix.lower() == ".csv":
            found.append(t)
            continue
        if not t.is_dir():
            continue
        for name in PRED_NAMES:
            found.extend(t.rglob(name))
    seen = set()
    out: List[Path] = []
    for p in found:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return sorted(out)


def predictions_to_multihot(
    df: pd.DataFrame,
    classes: List[str],
    pred_col: str,
) -> Optional[tuple]:
    if "true_labels" not in df.columns or pred_col not in df.columns:
        return None
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    C = len(classes)
    y_true = np.zeros((len(df), C), dtype=np.int8)
    y_pred = np.zeros((len(df), C), dtype=np.int8)
    sep = config.label_sep
    for i, row in enumerate(df.itertuples(index=False)):
        rd = row._asdict()
        for c in parse_label_field(rd["true_labels"], sep=sep):
            if c in cls_to_idx:
                y_true[i, cls_to_idx[c]] = 1
        for c in parse_label_field(rd[pred_col], sep=sep):
            if c in cls_to_idx:
                y_pred[i, cls_to_idx[c]] = 1
    return y_true, y_pred


def evaluate_csv(csv_path: Path) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(csv_path)
    classes = list(config.deeploc_classes)
    present = list(getattr(config, "hpa_eval_classes_present", []) or [])
    paper = list(getattr(config, "hpa_eval_classes_paper", []) or [])

    result: Dict[str, Dict[str, float]] = {}
    pred_cols = [c for c in ("pred_labels_default", "pred_labels_tuned", "pred_labels")
                 if c in df.columns]
    if not pred_cols:
        print(f"[skip] {csv_path} has no pred_labels* column", file=sys.stderr)
        return {}

    for pc in pred_cols:
        decoded = predictions_to_multihot(df, classes, pc)
        if decoded is None:
            continue
        y_true, y_pred = decoded
        bucket: Dict[str, Dict[str, float]] = {}
        if present:
            bucket["hpa8_present"] = compute_restricted_metrics(
                y_true, y_pred, classes, present
            )
        if paper:
            bucket["hpa6_paper"] = compute_restricted_metrics(
                y_true, y_pred, classes, paper
            )
        # Full 10-class — for direct comparison.
        from sklearn.metrics import f1_score, accuracy_score
        bucket["all10"] = {
            "n_classes": len(classes),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "subset_acc": float(accuracy_score(y_true, y_pred)),
        }
        result[pc] = bucket
    return result


def short_summary(name: str, buckets: Dict[str, Dict[str, float]]) -> str:
    parts = [name]
    for subset, m in buckets.items():
        parts.append(
            f"{subset}: macro_f1={m['macro_f1']:.4f} micro_f1={m['micro_f1']:.4f}"
            + (f" macro_mcc={m['macro_mcc']:.4f}" if "macro_mcc" in m else "")
        )
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "targets",
        nargs="*",
        type=Path,
        help="Run dirs (cv_results_*/<run_id>) or fold dirs or prediction CSVs. "
             "If empty, all cv_results_* dirs under the project root are scanned.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        default=[],
        help="Add a specific predictions CSV (repeatable).",
    )
    args = parser.parse_args()

    targets = list(args.targets) + list(args.csv)
    if not targets:
        targets = sorted(ROOT.glob("cv_results_*"))

    pred_csvs = find_prediction_csvs(targets)
    if not pred_csvs:
        print("No HPA prediction CSVs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pred_csvs)} prediction CSV(s).\n")
    for csv_path in pred_csvs:
        buckets_per_col = evaluate_csv(csv_path)
        if not buckets_per_col:
            continue
        rel = csv_path.relative_to(ROOT) if csv_path.is_absolute() else csv_path
        for pred_col, buckets in buckets_per_col.items():
            print(short_summary(f"{rel} [{pred_col}]", buckets))
        out_json = csv_path.with_name("hpa_restricted_metrics.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(buckets_per_col, f, indent=2)
        print(f"  -> wrote {out_json.relative_to(ROOT) if out_json.is_absolute() else out_json}\n")


if __name__ == "__main__":
    main()
