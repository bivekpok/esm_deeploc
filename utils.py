"""
Shared utilities: DDP setup/teardown, all-gather helpers, multilabel decoding,
deterministic seeding.

These are deliberately lightweight (no project-specific logic) so they can be
shared across ``train.py`` and any future evaluation scripts.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    jaccard_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


# ---------------------------------------------------------------------------
# DDP lifecycle
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def setup_ddp() -> Tuple[int, int, int, torch.device]:
    """
    Return ``(local_rank, rank, world_size, device)``.

    - If torchrun env vars are present we initialize the process group.
    - Otherwise we run in single-process mode (rank=0, world_size=1).
    """
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)

    if world_size > 1 and dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return local_rank, rank, world_size, device


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Collective helpers
# ---------------------------------------------------------------------------

def reduce_sum(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """In-place sum-reduce across ranks (no-op when single-process)."""
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def broadcast_scalar(
    value: float, src: int, device: torch.device, world_size: int
) -> float:
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return float(value)
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.broadcast(t, src=src)
    return float(t.item())


def broadcast_int(
    value: int, src: int, device: torch.device, world_size: int
) -> int:
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return int(value)
    t = torch.tensor([int(value)], device=device, dtype=torch.int64)
    dist.broadcast(t, src=src)
    return int(t.item())


def all_gather_list(local_obj, world_size: int) -> list:
    """Gather a picklable Python object from every rank onto every rank."""
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return [local_obj]
    out: List = [None] * world_size
    dist.all_gather_object(out, local_obj)
    return out


def all_gather_numpy(local: np.ndarray, world_size: int) -> np.ndarray:
    """Concatenate per-rank ndarrays along axis 0."""
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return local
    parts = all_gather_list(local, world_size)
    if not parts:
        return local
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Multilabel decoding
# ---------------------------------------------------------------------------

def multilabel_predict(
    logits: np.ndarray,
    threshold=0.5,
    fallback_top1: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sigmoid -> threshold; if a row has no positive prediction and
    ``fallback_top1`` is True, the argmax class is forced on so the per-row
    macro-F1 stays meaningful.

    ``threshold`` may be a scalar (one value for every class) or an array of
    length ``C`` (per-class thresholds tuned on the validation set).

    Returns ``(probs, preds)`` with ``preds`` shaped ``(N, C)`` uint8.
    """
    if logits.size == 0:
        return logits.copy(), np.zeros_like(logits, dtype=np.uint8)
    probs = 1.0 / (1.0 + np.exp(-logits))
    thr = np.asarray(threshold, dtype=np.float64).reshape(-1)
    if thr.size == 1:
        thr = float(thr[0])
        preds = (probs >= thr).astype(np.uint8)
    else:
        if thr.size != probs.shape[1]:
            raise ValueError(
                f"per-class threshold length {thr.size} != n_classes {probs.shape[1]}"
            )
        preds = (probs >= thr[None, :]).astype(np.uint8)
    if fallback_top1:
        empties = preds.sum(axis=1) == 0
        if np.any(empties):
            top = probs[empties].argmax(axis=1)
            preds[empties, top] = 1
    return probs, preds


def _binary_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if den == 0:
        return 0.0
    return float(num / np.sqrt(den))


def best_threshold_mcc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Threshold maximizing binary MCC by scanning sorted probabilities.
    Matches DeepLoc 2.0 ``get_best_threshold_mcc`` (Kaggle MCC trick).
    """
    y = y_true.astype(np.int8).reshape(-1)
    p = y_prob.astype(np.float64).reshape(-1)
    n = y.shape[0]
    n_pos = int(y.sum())
    if n == 0 or n_pos == 0 or n_pos == n:
        return 0.5

    order = np.argsort(p)
    y_sort = y[order]
    tp = float(n_pos)
    tn = 0.0
    fp = float(n - n_pos)
    fn = 0.0
    best_mcc = -1.0
    best_thr = 0.5
    prev_p = -1.0

    for i in range(n):
        prob = float(p[order[i]])
        if prob != prev_p:
            prev_p = prob
            mcc = _binary_mcc(int(tp), int(tn), int(fp), int(fn))
            if mcc >= best_mcc:
                best_mcc = mcc
                best_thr = prob
        if y_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    return float(best_thr)


def tune_per_class_thresholds(
    logits: np.ndarray,
    labels: np.ndarray,
    grid: Optional[np.ndarray] = None,
    default: float = 0.5,
    min_positives: int = 1,
    metric: str = "f1",
) -> np.ndarray:
    """
    Per-class threshold tuning on sigmoid probabilities.

    ``metric``:
      - ``"f1"`` — grid search maximizing binary F1 (default, val-friendly).
      - ``"mcc"`` — sorted scan maximizing MCC (DeepLoc 2.0 paper / official repo).
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    if logits.size == 0:
        n_classes = labels.shape[1] if labels.ndim == 2 else 0
        return np.full(n_classes, float(default), dtype=np.float64)

    probs = 1.0 / (1.0 + np.exp(-logits))
    n_classes = probs.shape[1]
    thresholds = np.full(n_classes, float(default), dtype=np.float64)
    key = (metric or "f1").lower()

    for c in range(n_classes):
        y = labels[:, c].astype(np.int8)
        if int(y.sum()) < min_positives:
            continue
        p = probs[:, c]
        if key == "mcc":
            thresholds[c] = best_threshold_mcc(y, p)
        elif key == "f1":
            best_t, best_f1 = float(default), -1.0
            for t in grid:
                pred = (p >= t).astype(np.int8)
                tp = int(((pred == 1) & (y == 1)).sum())
                fp = int(((pred == 1) & (y == 0)).sum())
                fn = int(((pred == 0) & (y == 1)).sum())
                denom = 2 * tp + fp + fn
                f1 = (2 * tp) / denom if denom > 0 else 0.0
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            thresholds[c] = best_t
        else:
            raise ValueError(f"Unknown threshold metric {metric!r}; use 'f1' or 'mcc'.")
    return thresholds


# ---------------------------------------------------------------------------
# Paper-aligned HPA evaluation (restricted-class metrics)
# ---------------------------------------------------------------------------
#
# HPA is human-only: Plastid is biologically absent and Extracellular is not
# captured by HPA's immunofluorescence assay. DeepLoc 2.0 (Thumuluri et al.
# 2022, Table 2) reports HPA metrics over present classes only, not all 10
# training classes. These helpers slice the 10-class predictions to a given
# subset so the paper-style numbers can be reported alongside the full
# 10-class scores.

def restrict_to_classes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    all_classes: Sequence[str],
    keep_classes: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Slice (N, C) multi-hot arrays to columns in ``keep_classes``.

    Returns ``(y_true_sub, y_pred_sub, ordered_kept)`` where ``ordered_kept``
    keeps the order of ``keep_classes`` and skips names not in ``all_classes``.
    """
    idx_map = {c: i for i, c in enumerate(all_classes)}
    cols = [idx_map[c] for c in keep_classes if c in idx_map]
    kept = [c for c in keep_classes if c in idx_map]
    if not cols:
        raise ValueError(
            "restrict_to_classes: none of keep_classes are in all_classes"
        )
    return y_true[:, cols], y_pred[:, cols], kept


def compute_restricted_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    all_classes: Sequence[str],
    keep_classes: Sequence[str],
) -> Dict[str, float]:
    """Macro/micro F1 + per-class F1/MCC over a class subset (HPA-aligned).

    All averaging happens only over ``keep_classes`` so missing/empty classes
    (Extracellular, Plastid, etc.) do not drag macro F1 down.
    """
    yt, yp, kept = restrict_to_classes(y_true, y_pred, all_classes, keep_classes)
    out: Dict[str, float] = {
        "n_classes": int(len(kept)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(yt, yp, average="micro", zero_division=0)),
        "macro_precision": float(
            precision_score(yt, yp, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(yt, yp, average="macro", zero_division=0)),
        "subset_acc": float(accuracy_score(yt, yp)),
        "jaccard_samples": float(jaccard_score(yt, yp, average="samples", zero_division=0)),
    }
    mccs: List[float] = []
    for i, cls in enumerate(kept):
        col_true = yt[:, i]
        col_pred = yp[:, i]
        if col_true.sum() + col_pred.sum() == 0:
            mcc_i = 0.0
        else:
            mcc_i = float(matthews_corrcoef(col_true, col_pred))
        out[f"f1/{cls}"] = float(
            f1_score(col_true, col_pred, zero_division=0)
        )
        out[f"mcc/{cls}"] = mcc_i
        out[f"support/{cls}"] = int(col_true.sum())
        mccs.append(mcc_i)
    out["macro_mcc"] = float(np.mean(mccs)) if mccs else 0.0
    return out


def format_restricted_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    all_classes: Sequence[str],
    keep_classes: Sequence[str],
    *,
    title: str,
) -> str:
    """Pretty-print restricted-class classification_report + MCC column."""
    yt, yp, kept = restrict_to_classes(y_true, y_pred, all_classes, keep_classes)
    report = classification_report(yt, yp, target_names=kept, zero_division=0)
    mcc_lines = ["Per-class MCC:"]
    for i, cls in enumerate(kept):
        col_true = yt[:, i]
        col_pred = yp[:, i]
        if col_true.sum() + col_pred.sum() == 0:
            mcc_i = 0.0
        else:
            mcc_i = float(matthews_corrcoef(col_true, col_pred))
        mcc_lines.append(f"  {cls:<25} support={int(col_true.sum()):>5d}  mcc={mcc_i:+.4f}")
    return f"\n=== {title} (n_classes={len(kept)}) ===\n{report}\n" + "\n".join(mcc_lines)
