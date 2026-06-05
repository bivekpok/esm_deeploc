"""
Lightweight per-fold diagnostics: class distribution + BCE ``pos_weight``.

Imported by ``train.py`` so the rank-0 process can log a quick sanity check
before each fold starts training.
"""

from __future__ import annotations


def print_multilabel_train_diag(train_ds, fold_name: str, entropy_weight: float) -> None:
    """Pretty-print fold-level multilabel stats to stdout."""
    classes = list(train_ds.classes)
    targets = train_ds.targets
    pos = targets.sum(dim=0)
    N = int(targets.shape[0])

    print(f"\n[{fold_name}] train rows={N}  classes={len(classes)}", flush=True)
    print(f"[{fold_name}] entropy regularizer weight = {entropy_weight}", flush=True)

    sqrt_pw = train_ds.pos_weight.sqrt().tolist()
    pw = train_ds.pos_weight.tolist()
    width = max((len(c) for c in classes), default=0)
    print(
        f"[{fold_name}] {'class'.ljust(width)}  n_pos  frac    pos_weight  sqrt(pw)",
        flush=True,
    )
    for c, p, w, sw in zip(classes, pos.tolist(), pw, sqrt_pw):
        frac = (p / N) if N > 0 else 0.0
        print(
            f"[{fold_name}] {c.ljust(width)}  "
            f"{int(p):>5d}  {frac:>5.3f}  {w:>10.3f}  {sw:>8.3f}",
            flush=True,
        )
