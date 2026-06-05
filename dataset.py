"""
Dataset, manifest generation, and DataLoader factories for DeepLoc 2.0.

Train/Val CSV (``config.label_csv``):
    Columns: ACC, Kingdom, Partition, Membrane, <10 location columns>, Sequence
    - ``Partition`` (0..4) is the homology-partitioned 5-fold split provided by
      the DeepLoc 2.0 authors; we use it directly instead of any clustering.
    - The 10 location columns are 0/1 multi-label targets (see ``DEEPLOC_CLASSES``).
    - ``Membrane`` is a metadata flag (not a target). If
      ``config.membrane_only=True``, we filter to ``Membrane == 1.0`` rows only.

External test CSV (``config.test_csv``):
    Columns may use ``sid`` instead of ``ACC`` and may omit some location columns
    (e.g. Extracellular/Plastid are absent from the published HPA test CSV).
    Missing class columns are treated as all-zero so the label dimension still
    matches the fixed ``DEEPLOC_CLASSES`` ordering.

Per-fold layout written by ``generate_partition_manifests`` (``config.cv_scheme``)::

    Outer_Fold_1/                       # val partition = 0
        Inner_Fold_1/
            train_manifest.csv          # partitions != 0  (4 partitions)
            valid_manifest.csv          # partition == 0   (1 partition)
    Outer_Fold_2/                       # val partition = 1
        ...
    Outer_Fold_5/                       # val partition = 4

**Paper scheme (``cv_scheme="paper"``, default):**

    For fold *i* with validation partition *p*:
        Train = all rows with ``Partition != p``  (4 homology partitions)
        Val   = all rows with ``Partition == p``    (1 whole partition)
        No internal test split — ``hpa_testset.csv`` is the ONLY test set.

    This matches DeepLoc 2.0 cross-validation: whole partitions as val folds,
    no random row splits, no Swiss-Prot proteins held out as a separate
    "partition test". Threshold tuning (if enabled) uses the val partition only;
    HPA is never used during training or tuning.

Legacy schemes (``cv_scheme="holdout"`` | ``"random"``) still write
``test_manifest.csv`` for backwards compatibility but are not recommended.

All manifests share the same schema:
    accession, sequence, label, partition[, kingdom, membrane]
where ``label`` is the chosen ``config.label_sep``-joined string of positive
location names (sorted) for that row.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from config import config


DEEPLOC_CLASSES: List[str] = list(config.deeploc_classes)
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(DEEPLOC_CLASSES)}


# ---------------------------------------------------------------------------
# Label parsing helpers
# ---------------------------------------------------------------------------

def parse_label_field(value, sep: str = "|") -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    if not s:
        return []
    return [p.strip() for p in s.split(sep) if p.strip()]


def _row_positive_labels(row: pd.Series, class_cols: List[str]) -> List[str]:
    out = []
    for c in class_cols:
        if c not in row:
            continue
        try:
            v = float(row[c])
        except (TypeError, ValueError):
            continue
        if v >= 0.5:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProteinMultilabelDataset(Dataset):
    """
    Loads a manifest CSV (accession, sequence, label, ...) into a multi-hot
    dataset over ``DEEPLOC_CLASSES`` (fixed ordering shared by every loader).
    """

    def __init__(
        self,
        csv_path: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        max_len: int = 1000,
        test_mode: bool = False,
        label_sep: str = "|",
    ):
        self.max_len = max_len
        self.label_sep = label_sep
        raw = pd.read_csv(csv_path)
        if test_mode:
            raw = raw.head(200)

        if "accession" not in raw.columns:
            raise ValueError(f"{csv_path} missing required column 'accession'")
        if "sequence" not in raw.columns:
            raise ValueError(f"{csv_path} missing required column 'sequence'")
        if "label" not in raw.columns:
            raise ValueError(f"{csv_path} missing required column 'label'")

        self.ids = raw["accession"].astype(str).tolist()
        self.sequences = [self._truncate(str(s)) for s in raw["sequence"].tolist()]
        self.lengths = [len(s) for s in self.sequences]

        raw_lists = [parse_label_field(v, sep=label_sep) for v in raw["label"].tolist()]

        if class_to_idx is None:
            class_to_idx = dict(CLASS_TO_IDX)

        self.class_to_idx = dict(class_to_idx)
        self.classes = [k for k, _ in sorted(self.class_to_idx.items(), key=lambda x: x[1])]

        C = len(self.classes)
        N = len(raw_lists)
        targets = torch.zeros(N, C, dtype=torch.float32)
        unseen: Counter = Counter()
        for i, lst in enumerate(raw_lists):
            for c in lst:
                if c in self.class_to_idx:
                    targets[i, self.class_to_idx[c]] = 1.0
                else:
                    unseen[c] += 1
        if unseen:
            print(
                f"[ProteinMultilabelDataset] labels not in class_to_idx (ignored): {dict(unseen)}",
                flush=True,
            )

        self.targets = targets

        n_pos = targets.sum(dim=0)
        n_neg = N - n_pos
        self.pos_weight = (n_neg / n_pos.clamp(min=1.0)).clamp(min=1.0)

    def _truncate(self, sequence: str) -> str:
        if len(sequence) <= self.max_len:
            return sequence
        front_len = self.max_len // 2
        back_len = self.max_len - front_len
        return sequence[:front_len] + sequence[-back_len:]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        return (
            self.sequences[idx],
            self.targets[idx],
            self.lengths[idx],
            self.ids[idx],
        )


def collate_fn(batch):
    sequences, labels, lengths, ids = zip(*batch)
    return {
        "sequences": sequences,
        "labels": torch.stack(labels, dim=0),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "ids": ids,
    }


# ---------------------------------------------------------------------------
# DeepLoc 2.0 manifest generation
# ---------------------------------------------------------------------------

def _load_deeploc_train_csv(csv_path: Path) -> pd.DataFrame:
    """Load DeepLoc 2.0 train/val CSV, build canonical label string, optionally
    filter to membrane-only proteins."""
    df = pd.read_csv(csv_path)

    acc_col = "ACC" if "ACC" in df.columns else ("accession" if "accession" in df.columns else None)
    if acc_col is None:
        raise ValueError(f"{csv_path}: no 'ACC' or 'accession' column")
    if "Sequence" not in df.columns and "sequence" not in df.columns:
        raise ValueError(f"{csv_path}: no 'Sequence' / 'sequence' column")
    if "Partition" not in df.columns:
        raise ValueError(f"{csv_path}: no 'Partition' column (DeepLoc 2.0 5-fold split)")

    seq_col = "Sequence" if "Sequence" in df.columns else "sequence"

    if getattr(config, "membrane_only", False) and "Membrane" in df.columns:
        before = len(df)
        df = df[df["Membrane"].astype(float) >= 0.5].copy()
        print(
            f"[deeploc] membrane_only=True: kept {len(df)} / {before} rows "
            "(Membrane>=0.5).",
            flush=True,
        )

    sep = config.label_sep
    present_class_cols = [c for c in DEEPLOC_CLASSES if c in df.columns]
    missing = [c for c in DEEPLOC_CLASSES if c not in df.columns]
    if missing:
        print(
            f"[deeploc][train] note: classes absent from CSV (will be 0): {missing}",
            flush=True,
        )

    def _join(row: pd.Series) -> str:
        labs = _row_positive_labels(row, present_class_cols)
        return sep.join(sorted(labs))

    out = pd.DataFrame(
        {
            "accession": df[acc_col].astype(str).values,
            "sequence": df[seq_col].astype(str).values,
            "label": df.apply(_join, axis=1).values,
            "partition": df["Partition"].astype(int).values,
        }
    )
    if "Kingdom" in df.columns:
        out["kingdom"] = df["Kingdom"].astype(str).values
    if "Membrane" in df.columns:
        out["membrane"] = df["Membrane"].astype(float).values

    no_label_mask = out["label"].str.len() == 0
    if int(no_label_mask.sum()) > 0:
        print(
            f"[deeploc][train] dropping {int(no_label_mask.sum())} rows with no positive label",
            flush=True,
        )
        out = out[~no_label_mask].reset_index(drop=True)
    return out


def load_deeploc_test_csv(csv_path: Path) -> pd.DataFrame:
    """Load DeepLoc 2.0 independent test CSV (test_label.csv / hpa_testset.csv)
    into the same ``accession,sequence,label`` schema, padding missing class
    columns with zeros."""
    df = pd.read_csv(csv_path)

    acc_col = None
    for cand in ("ACC", "accession", "sid", "id"):
        if cand in df.columns:
            acc_col = cand
            break
    if acc_col is None:
        raise ValueError(f"{csv_path}: no accession-like column (ACC/accession/sid/id)")

    seq_col = None
    for cand in ("Sequence", "sequence", "fasta", "FASTA"):
        if cand in df.columns:
            seq_col = cand
            break
    if seq_col is None:
        raise ValueError(
            f"{csv_path}: no sequence column (Sequence / sequence / fasta / FASTA)"
        )

    sep = config.label_sep
    present_class_cols = [c for c in DEEPLOC_CLASSES if c in df.columns]
    missing = [c for c in DEEPLOC_CLASSES if c not in df.columns]
    if missing:
        print(
            f"[deeploc][test] classes absent from test CSV (treated as 0): {missing}",
            flush=True,
        )

    def _join(row: pd.Series) -> str:
        labs = _row_positive_labels(row, present_class_cols)
        return sep.join(sorted(labs))

    out = pd.DataFrame(
        {
            "accession": df[acc_col].astype(str).values,
            "sequence": df[seq_col].astype(str).values,
            "label": df.apply(_join, axis=1).values,
        }
    )
    if getattr(config, "membrane_only", False) and "Membrane" in df.columns:
        before = len(out)
        keep = df["Membrane"].astype(float).values >= 0.5
        out = out.loc[keep].reset_index(drop=True)
        print(
            f"[deeploc][test] membrane_only=True: kept {len(out)} / {before} rows.",
            flush=True,
        )
    return out


SPLITS_META_NAME = "splits_meta.json"


def read_splits_meta(splits_root: Path) -> Optional[dict]:
    """Return the splits metadata dict, or None if absent / unreadable."""
    meta_path = Path(splits_root) / SPLITS_META_NAME
    if not meta_path.is_file():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def splits_meta_matches(
    splits_root: Path,
    *,
    cv_scheme: str,
    n_outer: int,
    inner_val_fraction: float,
    random_state: int,
    membrane_only: bool,
) -> bool:
    """True iff existing manifests were built with the same options."""
    meta = read_splits_meta(splits_root)
    if not meta:
        return False
    scheme = meta.get("cv_scheme") or meta.get("inner_val_strategy")
    return (
        scheme == cv_scheme
        and int(meta.get("n_outer", 0)) == int(n_outer)
        and bool(meta.get("membrane_only", False)) == bool(membrane_only)
        and (cv_scheme != "random"
             or (float(meta.get("inner_val_fraction", -1)) == float(inner_val_fraction)
                 and int(meta.get("random_state", -1)) == int(random_state)))
    )


def generate_partition_manifests(
    csv_path: Path,
    output_root: Path,
    n_outer: int = 5,
    inner_val_fraction: float = 0.10,
    random_state: int = 42,
    cv_scheme: str = "paper",
) -> List[str]:
    """
    Write per-fold manifests using the DeepLoc 2.0 ``Partition`` column.

    ``cv_scheme="paper"`` (DeepLoc 2.0 CV — recommended):
        For fold *k* with validation partition ``p``:
            Train = all rows with ``Partition != p``  (4 partitions)
            Val   = all rows with ``Partition == p``    (1 whole partition)
        No ``test_manifest.csv`` — evaluate only on ``hpa_testset.csv``.

    Legacy schemes (not recommended):
        ``holdout``: outer test = ``p``, inner val = next partition, train = rest.
        ``random``:  outer test = ``p``, random 10% val from remaining rows.
    """
    df = _load_deeploc_train_csv(csv_path)
    if df.empty:
        raise RuntimeError(
            f"No usable rows in {csv_path} (every row had empty label after filtering)."
        )
    partitions = sorted(df["partition"].unique().tolist())
    n_part = len(partitions)
    if n_part < n_outer:
        print(
            f"[generate_partition_manifests] only {n_part} partitions present "
            f"({partitions}); n_outer_folds capped to {n_part}.",
            flush=True,
        )
        n_outer = n_part

    if cv_scheme not in ("paper", "holdout", "random"):
        raise ValueError(
            f"cv_scheme must be 'paper', 'holdout', or 'random' (got {cv_scheme!r})"
        )
    if cv_scheme in ("holdout", "random") and n_part < 2:
        raise RuntimeError(f"cv_scheme={cv_scheme!r} needs >=2 partitions in the CSV.")

    fold_names: List[str] = []
    fold_meta: List[dict] = []
    for fold_idx, p_val in enumerate(partitions[:n_outer], start=1):
        outer_path = output_root / f"Outer_Fold_{fold_idx}"
        outer_path.mkdir(parents=True, exist_ok=True)

        if cv_scheme == "paper":
            train_df = df[df["partition"] != p_val].reset_index(drop=True)
            val_df = df[df["partition"] == p_val].reset_index(drop=True)
            test_df = None
            msg = f"P_val={p_val} (paper CV: train=4 partitions, val=1 partition)"
        elif cv_scheme == "holdout":
            p_test = p_val
            p_inner_val = partitions[(partitions.index(p_test) + 1) % n_part]
            test_df = df[df["partition"] == p_test].reset_index(drop=True)
            val_df = df[df["partition"] == p_inner_val].reset_index(drop=True)
            train_df = df[
                (df["partition"] != p_test) & (df["partition"] != p_inner_val)
            ].reset_index(drop=True)
            msg = f"P_test={p_test} P_val={p_inner_val} (legacy holdout)"
        else:  # random
            p_test = p_val
            rest_df = df[df["partition"] != p_test].reset_index(drop=True)
            test_df = df[df["partition"] == p_test].reset_index(drop=True)
            sep = config.label_sep
            strat_key = (
                rest_df["label"]
                .map(lambda s: s.split(sep)[0] if isinstance(s, str) and s else "_none_")
                .values
            )
            try:
                train_df, val_df = train_test_split(
                    rest_df,
                    test_size=inner_val_fraction,
                    random_state=random_state,
                    stratify=strat_key,
                )
            except ValueError:
                train_df, val_df = train_test_split(
                    rest_df,
                    test_size=inner_val_fraction,
                    random_state=random_state,
                )
            msg = f"P_test={p_test} (legacy random inner val)"

        inner_path = outer_path / "Inner_Fold_1"
        inner_path.mkdir(parents=True, exist_ok=True)

        out_cols = ["accession", "sequence", "label", "partition"]
        for extra in ("kingdom", "membrane"):
            if extra in df.columns:
                out_cols.append(extra)

        train_df[out_cols].to_csv(inner_path / "train_manifest.csv", index=False)
        val_df[out_cols].to_csv(inner_path / "valid_manifest.csv", index=False)
        if test_df is not None:
            test_df[out_cols].to_csv(outer_path / "test_manifest.csv", index=False)
        elif (outer_path / "test_manifest.csv").exists():
            (outer_path / "test_manifest.csv").unlink()

        n_test = len(test_df) if test_df is not None else 0
        print(
            f"[generate_partition_manifests] Fold {fold_idx} ({msg}): "
            f"train={len(train_df)} val={len(val_df)}"
            + (f" test={n_test}" if n_test else ""),
            flush=True,
        )
        fold_names.append(outer_path.name)
        fold_meta.append({
            "fold": fold_idx,
            "p_val": int(p_val),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(n_test),
            "cv_scheme": cv_scheme,
        })

    meta_payload = {
        "cv_scheme": cv_scheme,
        "inner_val_strategy": cv_scheme,  # backwards compat for old readers
        "n_outer": int(n_outer),
        "inner_val_fraction": float(inner_val_fraction),
        "random_state": int(random_state),
        "membrane_only": bool(getattr(config, "membrane_only", False)),
        "label_csv": str(csv_path),
        "external_test_csv": str(getattr(config, "test_csv", "")),
        "folds": fold_meta,
    }
    with open(Path(output_root) / SPLITS_META_NAME, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)
    return fold_names


def write_external_test_manifest(test_csv_path: Path, output_path: Path) -> Path:
    """Convert the external DeepLoc test CSV to the standard manifest schema
    on disk so any rank/process can re-load it cheaply."""
    df = load_deeploc_test_csv(test_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(
        f"[deeploc] wrote external test manifest -> {output_path} ({len(df)} rows)",
        flush=True,
    )
    return output_path


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def create_fold_loaders(
    fold_dir: Path,
    rank: int,
    world_size: int,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    max_len: Optional[int] = None,
    test_mode: Optional[bool] = None,
    label_sep: Optional[str] = None,
):
    """Train + valid loaders for one ``Outer_Fold_*/Inner_Fold_1``."""
    batch_size = batch_size if batch_size is not None else config.batch_size
    num_workers = num_workers if num_workers is not None else config.num_workers
    max_len = max_len if max_len is not None else config.max_len
    test_mode = test_mode if test_mode is not None else config.test_mode
    label_sep = label_sep if label_sep is not None else config.label_sep

    train_csv = fold_dir / "Inner_Fold_1" / "train_manifest.csv"
    val_csv = fold_dir / "Inner_Fold_1" / "valid_manifest.csv"

    train_ds = ProteinMultilabelDataset(
        str(train_csv),
        class_to_idx=dict(CLASS_TO_IDX),
        max_len=max_len,
        test_mode=test_mode,
        label_sep=label_sep,
    )
    val_ds = ProteinMultilabelDataset(
        str(val_csv),
        class_to_idx=train_ds.class_to_idx,
        max_len=max_len,
        test_mode=test_mode,
        label_sep=label_sep,
    )

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        train_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader, train_ds


def create_test_loader(
    fold_dir: Path,
    class_to_idx: Dict[str, int],
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    max_len: Optional[int] = None,
    test_mode: Optional[bool] = None,
    label_sep: Optional[str] = None,
):
    """Single-process held-out partition test loader (rank 0 only)."""
    batch_size = batch_size if batch_size is not None else config.batch_size
    num_workers = num_workers if num_workers is not None else config.num_workers
    max_len = max_len if max_len is not None else config.max_len
    test_mode = test_mode if test_mode is not None else config.test_mode
    label_sep = label_sep if label_sep is not None else config.label_sep

    test_csv = fold_dir / "test_manifest.csv"
    test_ds = ProteinMultilabelDataset(
        str(test_csv),
        class_to_idx=class_to_idx,
        max_len=max_len,
        test_mode=test_mode,
        label_sep=label_sep,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader, test_ds


def create_external_test_loader(
    manifest_path: Path,
    class_to_idx: Dict[str, int],
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    max_len: Optional[int] = None,
    test_mode: Optional[bool] = None,
    label_sep: Optional[str] = None,
):
    """Independent (HPA-style) test loader; rank 0 only."""
    batch_size = batch_size if batch_size is not None else config.batch_size
    num_workers = num_workers if num_workers is not None else config.num_workers
    max_len = max_len if max_len is not None else config.max_len
    test_mode = test_mode if test_mode is not None else config.test_mode
    label_sep = label_sep if label_sep is not None else config.label_sep

    test_ds = ProteinMultilabelDataset(
        str(manifest_path),
        class_to_idx=class_to_idx,
        max_len=max_len,
        test_mode=test_mode,
        label_sep=label_sep,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader, test_ds
