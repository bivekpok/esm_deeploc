#!/usr/bin/env python3
"""
Count membrane-related proteins in DeepLoc CV manifests (per fold, per split).

Two notions (reported separately):

1. **membrane_flag** — DeepLoc ``Membrane`` metadata score >= threshold (default 0.8).
   In Swissprot this column is 0.0 or 1.0, so >= 0.8 is equivalent to == 1.0.

2. **localization_membrane_only** — multilabel string is exactly ``Cell membrane``
   (no Cytoplasm, Nucleus, etc.).

3. **strict_membrane_only** — both of the above (flag high *and* single membrane localization).

Splits per ``Outer_Fold_*``:
  - train  → ``Inner_Fold_1/train_manifest.csv``
  - val    → ``Inner_Fold_1/valid_manifest.csv``
  - test   → ``test_manifest.csv`` (held-out homology partition)

External benchmark (same for every fold):
  - ``cv_splits_deeploc/external_test_manifest.csv`` (no ``membrane`` column; localization only)

Usage:
  python scripts/count_membrane_proteins.py
  python scripts/count_membrane_proteins.py --threshold 0.8 --splits-root cv_splits_deeploc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Project root on path for optional imports
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import config  # noqa: E402
from dataset import DEEPLOC_CLASSES, parse_label_field  # noqa: E402

CELL_MEMBRANE = "Cell membrane"


def _is_localization_membrane_only(label_value: object, sep: str) -> bool:
    labs = parse_label_field(label_value, sep=sep)
    return labs == [CELL_MEMBRANE]


def _membrane_flag_series(df: pd.DataFrame, threshold: float) -> pd.Series | None:
    if "membrane" not in df.columns:
        return None
    return df["membrane"].astype(float) >= threshold


def count_split(
    csv_path: Path,
    threshold: float,
    label_sep: str,
) -> dict:
    df = pd.read_csv(csv_path)
    n_total = len(df)

    loc_only = df["label"].map(lambda v: _is_localization_membrane_only(v, label_sep))
    n_loc_only = int(loc_only.sum())

    mem_flag = _membrane_flag_series(df, threshold)
    if mem_flag is not None:
        n_mem_flag = int(mem_flag.sum())
        n_strict = int((mem_flag & loc_only).sum())
        n_mem_not_loc_only = int((mem_flag & ~loc_only).sum())
    else:
        n_mem_flag = None
        n_strict = None
        n_mem_not_loc_only = None

    has_cm = df["label"].astype(str).str.contains(CELL_MEMBRANE, regex=False, na=False)
    n_has_cm = int(has_cm.sum())

    return {
        "path": str(csv_path),
        "n_total": n_total,
        "n_membrane_flag": n_mem_flag,
        "n_localization_membrane_only": n_loc_only,
        "n_strict_membrane_only": n_strict,
        "n_membrane_flag_not_loc_only": n_mem_not_loc_only,
        "n_has_cell_membrane_label": n_has_cm,
    }


def _pct(n: int | None, total: int) -> str:
    if n is None or total == 0:
        return "—"
    return f"{100.0 * n / total:.1f}%"


def print_table(rows: list[dict], threshold: float) -> None:
    print(f"\nMembrane metadata threshold: >= {threshold}")
    print(
        "Columns: total | membrane_flag | loc_only (Cell membrane only) | "
        "strict (both) | flag but multi-loc | has Cell membrane in label\n"
    )
    header = (
        f"{'fold':<14} {'split':<10} {'total':>7} "
        f"{'mem_flag':>9} {'loc_only':>9} {'strict':>8} "
        f"{'flag≠only':>9} {'has_CM':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        t = r["n_total"]
        print(
            f"{r['fold']:<14} {r['split']:<10} {t:>7} "
            f"{r['n_membrane_flag'] if r['n_membrane_flag'] is not None else '—':>9} "
            f"{r['n_localization_membrane_only']:>9} "
            f"{r['n_strict_membrane_only'] if r['n_strict_membrane_only'] is not None else '—':>8} "
            f"{r['n_membrane_flag_not_loc_only'] if r['n_membrane_flag_not_loc_only'] is not None else '—':>9} "
            f"{r['n_has_cell_membrane_label']:>8}"
        )
        if t:
            print(
                f"{'':14} {'(%)':<10} {'':7} "
                f"{_pct(r['n_membrane_flag'], t):>9} "
                f"{_pct(r['n_localization_membrane_only'], t):>9} "
                f"{_pct(r['n_strict_membrane_only'], t):>8} "
                f"{_pct(r['n_membrane_flag_not_loc_only'], t):>9} "
                f"{_pct(r['n_has_cell_membrane_label'], t):>8}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=Path(config.splits_root),
        help="CV manifest root (default: config.splits_root)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Membrane metadata cutoff (default: 0.8)",
    )
    parser.add_argument(
        "--label-sep",
        type=str,
        default=config.label_sep,
        help="Label separator in manifest 'label' column",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional path to write counts as CSV",
    )
    args = parser.parse_args()

    splits_root: Path = args.splits_root
    if not splits_root.is_dir():
        raise SystemExit(f"splits root not found: {splits_root}")

    fold_dirs = sorted(
        d for d in splits_root.iterdir() if d.is_dir() and d.name.startswith("Outer_Fold_")
    )
    if not fold_dirs:
        raise SystemExit(f"no Outer_Fold_* under {splits_root}")

    rows: list[dict] = []
    split_files = {
        "train": "Inner_Fold_1/train_manifest.csv",
        "val": "Inner_Fold_1/valid_manifest.csv",
        "test": "test_manifest.csv",
    }

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        for split_name, rel in split_files.items():
            path = fold_dir / rel
            if not path.exists():
                print(f"warning: missing {path}", file=sys.stderr)
                continue
            stats = count_split(path, args.threshold, args.label_sep)
            rows.append({"fold": fold_name, "split": split_name, **stats})

    ext_path = splits_root / "external_test_manifest.csv"
    if ext_path.exists():
        stats = count_split(ext_path, args.threshold, args.label_sep)
        rows.append({"fold": "all_folds", "split": "external", **stats})
        print(
            "\nNote: external manifest has no 'membrane' column — "
            "membrane_flag / strict counts are localization-only on that split.",
            file=sys.stderr,
        )
    else:
        print(f"warning: no {ext_path}", file=sys.stderr)

    print_table(rows, args.threshold)

    if args.csv_out:
        out_df = pd.DataFrame(rows)
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.csv_out, index=False)
        print(f"\nWrote {args.csv_out}")


if __name__ == "__main__":
    main()
