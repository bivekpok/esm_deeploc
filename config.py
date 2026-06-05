"""
Central configuration for the ESMC multilabel DeepLoc 2.0 training project.

This config is shared by ``train.py``, ``dataset.py``, ``model.py``,
``multilabel_train_diagnostics.py`` and ``utils.py``.

Data:
- ``label_csv``: DeepLoc 2.0 Swissprot Train/Validation CSV. Uses the
  ``Partition`` column (0..4) for 5-fold cross-validation (homology-partitioned
  at <=30% global identity by the DeepLoc authors).
- ``test_csv``: DeepLoc 2.0 independent test set (e.g. HPA / hpa_testset.csv
  or ``test_label.csv``). Evaluated *after* each fold finishes training.
- ``deeploc_classes``: canonical 10 sub-cellular location classes; used as a
  fixed ``class_to_idx`` for both train and test so that label dimensions stay
  aligned even when the test CSV doesn't contain every column.

Model variants:
- ``use_lora_model``: True -> PEFT LoRA on the ESMC backbone (adapters + head train).
  With ``lora_last_n_layers`` > 0, only the **last N** ``*.blocks.*`` layers get
  explicit LoRA targets; ``None`` uses short names on **all** matching layers.
- ``pooling_type``: ``"attention"`` | ``"average"`` | ``"bom"`` (default ``bom``) |
  ``"bom_attn"``.
    * "attention": N-term + C-term + global localization-attention pooling
      (``LocalizationAttention`` in model.py, returns a 3*embed_dim vector).
    * "average":   masked mean over the sequence (1*embed_dim vector).
    * "bom":       Bag-of-(k)Mer pooling - mean per k-mer window, then summary
      pool (mean+max) across windows (1*embed_dim vector).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


_ROOT = os.path.dirname(os.path.abspath(__file__))


# ESMC backbone presets (EvolutionaryScale ``esm`` package). Switch via the fields
# below, ``apply_esmc_backbone()``, or ``train.py --backbone esmc_600m``.
# 300M: d_model=960, 30 blocks (outputs 0..29). 600M: d_model=1152, 36 blocks (0..35).
ESMC_BACKBONE_PRESETS: dict[str, dict[str, int | str | tuple[int, int]]] = {
    "esmc_300m": {
        "model_name_or_path": "esmc_300m",
        "embed_dim": 960,
        "default_layer_agg_band": (15, 29),
    },
    "esmc_600m": {
        "model_name_or_path": "esmc_600m",
        "embed_dim": 1152,
        "default_layer_agg_band": (20, 35),
    },
}


def backbone_slug(cfg: "Config | None" = None) -> str | None:
    """Filesystem tag for non-default backbones; None for 300M (legacy dir names)."""
    c = cfg if cfg is not None else config
    key = (c.model_name_or_path or "").lower()
    if "600" in key:
        return "esmc600"
    return None


def apply_esmc_backbone(backbone_key: str, cfg: "Config | None" = None) -> None:
    """Set ``model_name_or_path`` and ``embed_dim`` for an ESMC variant."""
    if backbone_key not in ESMC_BACKBONE_PRESETS:
        raise ValueError(
            f"Unknown backbone {backbone_key!r}; choose from {list(ESMC_BACKBONE_PRESETS)}"
        )
    c = cfg if cfg is not None else config
    preset = ESMC_BACKBONE_PRESETS[backbone_key]
    c.model_name_or_path = str(preset["model_name_or_path"])
    c.embed_dim = int(preset["embed_dim"])
    # If ``layer_aggregation == "band"`` on 600M, also set:
    # c.layer_agg_band = tuple(preset["default_layer_agg_band"])  # type: ignore[arg-type]


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Backbone (ESMC). Outputs -> cv_results_deeploc_<run_tag>/.
    # 300M: 960-dim, 30 blocks (0..29).  600M: 1152-dim, 36 blocks (0..35).
    # -------------------------------------------------------------------------
    model_name_or_path = "esmc_300m"  # active: 300M | alt: "esmc_600m"
    embed_dim = 960                   # active: 960   | alt: 1152 (required for 600M)
    # model_name_or_path = "esmc_600m"
    # embed_dim = 1152

    label_csv = os.path.join(_ROOT, "Swissprot_Train_Validation_dataset.csv")
    # Independent HPA test set (1717 proteins). ``test_label.csv`` is the same
    # data with an extra (unused) integer ``class_label`` column; both files
    # carry identical 0/1 location labels.
    test_csv = os.path.join(_ROOT, "hpa_testset.csv")
    # test_csv = os.path.join(_ROOT, "test_label.csv")  # same labels + extra class_label col
    splits_root = os.path.join(_ROOT, "cv_splits_deeploc")
    output_dir = os.path.join(_ROOT, "cv_results_deeploc")
    # Filled in by ``train.py`` at launch time (UTC timestamp + short hash).
    # Each run lands under ``output_dir/<run_id>/`` so re-runs never overwrite.
    run_id: str | None = None

    deeploc_classes = [
        "Cytoplasm",
        "Nucleus",
        "Extracellular",
        "Cell membrane",
        "Mitochondrion",
        "Plastid",
        "Endoplasmic reticulum",
        "Lysosome/Vacuole",
        "Golgi apparatus",
        "Peroxisome",
    ]
    # HPA test set is human-only and structurally lacks Plastid (no plastids
    # in animals) and Extracellular (secreted proteins are washed away in
    # immunofluorescence sample prep). DeepLoc 2.0 trains on all 10 but
    # reports HPA results restricted to the present classes. We keep the
    # 10-class model output and only restrict the metric averaging.
    #
    #   hpa_eval_classes_present (8):  classes physically present in
    #     ``hpa_testset.csv`` — use this for the main HPA macro/micro F1 in
    #     the paper-aligned comparison.
    #   hpa_eval_classes_paper (6):    the subset that DeepLoc 2.0's Table 2
    #     reports MCC for (drops Lysosome/Vacuole and Peroxisome which have
    #     <10 positives in HPA).
    hpa_eval_classes_present = [
        "Cytoplasm",
        "Nucleus",
        "Cell membrane",
        "Mitochondrion",
        "Endoplasmic reticulum",
        "Lysosome/Vacuole",
        "Golgi apparatus",
        "Peroxisome",
    ]
    hpa_eval_classes_paper = [
        "Cytoplasm",
        "Nucleus",
        "Cell membrane",
        "Mitochondrion",
        "Endoplasmic reticulum",
        "Golgi apparatus",
    ]
    # -------------------------------------------------------------------------
    # Experiment / run settings — edit ONLY here, then:  ./submit_multilabel_cv.sh
    #
    # RUN 1 (active): 5-fold CV, ESMC-300M, BCE, MCC thresholds on train split
    #   wandb_project = "deeploc2_5fold_bce300_mcc"
    #   ./submit_multilabel_cv.sh
    #
    # RUN 2 (swap comments): 5-fold CV, ESMC-600M, same loss/thresholds
    #   Uncomment esmc_600m + embed_dim=1152 below; comment esmc_300m lines
    #   wandb_project = "deeploc2_5fold_bce600_mcc"
    #   NPROC_PER_NODE=1 ./submit_multilabel_cv.sh   # if OOM on 600M
    # -------------------------------------------------------------------------
    membrane_only = False       # alt: True  (DeepLoc Membrane >= 0.5 rows only)
    regenerate_splits = False   # alt: True  (rewrite cv_splits_deeploc/ manifests)
    data_summary_only = False   # alt: True  (print fold stats, no training)
    eval_external_test = True   # alt: False (skip HPA eval after each fold)
    nproc_per_node = 2          # alt: 1     (use if GPU OOM, e.g. ESMC-600M)
    run_tag: str | None = None  # alt: "my_run" (fixed tag; None = auto from settings)

    label_sep = "|"               # DeepLoc multilabel delimiter in CSV
    prediction_threshold = 0.5    # alt: tune_thresholds=True overrides per-class on val

    # -------------------------------------------------------------------------
    # Cross-validation (DeepLoc 2.0 paper protocol).
    # -------------------------------------------------------------------------
    cv_scheme = "paper"         # active: paper | alt: "holdout" | "random" (legacy)
    # cv_scheme = "holdout"
    # cv_scheme = "random"
    tune_thresholds = True      # alt: False (fixed 0.5 threshold on val + HPA)
    threshold_tune_metric = "mcc"   # active: mcc (paper) | alt: "f1" (legacy val F1)
    # threshold_tune_metric = "f1"
    threshold_tune_split = "train"  # active: train (paper) | alt: "val" (no train leakage)
    # threshold_tune_split = "val"
    folds_to_run: list[int] | None = None  # active: all 5 folds | alt: [1] (single fold)
    # folds_to_run = [1]

    # -------------------------------------------------------------------------
    # Training loop.
    # -------------------------------------------------------------------------
    batch_size = 16             # alt: 8 or 4 if OOM
    num_epochs = 150
    patience = 7
    early_stop_metric = "val_loss"  # active: val_loss | alt: "macro_f1" (better for focal)
    # early_stop_metric = "macro_f1"
    early_stop_min_delta = 1e-4
    lr_esmc = 5e-6              # alt: 1e-5
    lr_classifier = 1e-5        # alt: 3e-5
    weight_decay = 1e-3
    max_len = 1000
    entropy_weight = 0.25       # bom_attn pooling regularizer (see model.py)
    num_workers = 4

    # -------------------------------------------------------------------------
    # Loss — multilabel BCE vs focal.
    #
    # BCE baseline (paper-aligned, sqrt pos_weight on positives):
    #   loss_type = "bce"
    #
    # Focal loss (see losses.py MultilabelFocalLoss):
    #   loss_type = "focal"
    #   focal_gamma = 2.0
    #
    # Alpha options when loss_type == "focal":
    #   A) Manual per-class weights (recommended) — edit MANUAL_FOCAL_ALPHA in losses.py:
    #        focal_use_alpha = True
    #        focal_alpha_mode = "manual"
    #      Rare classes ~0.75, frequent ~0.25 (Cytoplasm/Nucleus), rest ~0.50.
    #
    #   B) No alpha (gamma-only focal):
    #        focal_use_alpha = False
    #      Run tag gets suffix ``noalpha``.
    #
    #   C) Auto inverse-frequency alpha (legacy, not recommended — crushes Cytoplasm):
    #        focal_use_alpha = True
    #        focal_alpha_mode = "inv_freq"
    #      Run tag gets suffix ``alphainvfreq``.
    # -------------------------------------------------------------------------
    loss_type = "bce"             # active: bce (5-fold production) | alt: "focal"
    # loss_type = "focal"
    focal_gamma = 2.0             # alt: 1.0 (milder) | 3.0 (stronger hard-example down-weight)
    focal_use_alpha = True        # active: True (manual/inv_freq) | alt: False (gamma-only focal)
    # focal_use_alpha = False
    focal_alpha_mode = "manual"   # active: manual (edit MANUAL_FOCAL_ALPHA in losses.py)
    # focal_alpha_mode = "inv_freq"   # auto 1/freq — not recommended (Cytoplasm ~0.03)
    focal_use_pos_weight = False  # alt: True (usually avoid — double-counts with alpha)

    # -------------------------------------------------------------------------
    # Backbone fine-tuning mode.
    # -------------------------------------------------------------------------
    use_lora_model = True         # active: LoRA | alt: False (full ESMC fine-tune)
    # use_lora_model = False
    lora_last_n_layers = 5        # active: last 5 blocks | alt: None or 0 (LoRA on all blocks)
    # lora_last_n_layers = None
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_total_blocks: int | None = None  # alt: 30 or 36 to override auto-detect
    lora_target_modules = (
        "attn.layernorm_qkv.1",
        "attn.out_proj",
        "ffn.1",
        "ffn.3",
    )

    # -------------------------------------------------------------------------
    # Pooling head (sequence -> fixed vector -> 10-class logits).
    # -------------------------------------------------------------------------
    pooling_type = "bom_attn"     # active: bom_attn | alt: "attention" | "average" | "bom"
    # pooling_type = "attention"    # N/C-term + global attn (3 * embed_dim)
    # pooling_type = "average"      # masked mean pooling
    # pooling_type = "bom"          # bag-of-mer mean+max (uses k_mer_size below)
    classify_dropout = 0.4
    classifier_hidden_dim = 512

    # -------------------------------------------------------------------------
    # Layer aggregation — mean over which ESMC hidden-state blocks feed pooling.
    # Tag suffix: none for "all"; ``agglast5``; ``aggband15-29`` (300M example).
    # -------------------------------------------------------------------------
    layer_aggregation = "all"       # active: all | alt: "last_n" | "band"
    # layer_aggregation = "last_n"  # mean over last layer_agg_n block outputs
    # layer_aggregation = "band"      # mean over layer_agg_band inclusive range
    layer_agg_n = 5                 # used only when layer_aggregation == "last_n"
    layer_agg_band: tuple[int, int] = (15, 29)  # 300M band | alt for 600M: (20, 35)
    # layer_agg_band = (20, 35)     # ESMC-600M band (36 blocks, indices 0..35)

    # -------------------------------------------------------------------------
    # bom_attn pooling knobs (used when pooling_type == "bom_attn").
    # Classic ``bom`` uses k_mer_size / bom_stride / bom_summary instead.
    # -------------------------------------------------------------------------
    bom_attn_k_mer_size = 7       # k-mer window size for self-attention over windows
    bom_attn_stride: int | None = None  # None -> bom_stride (below)
    bom_attn_inner_dim = 256      # Q/K projection dim
    bom_attn_value_dim = 1024     # V projection dim / default output dim
    bom_attn_output_dim: int | None = None  # alt: int (defaults to bom_attn_value_dim)
    bom_attn_dropout = 0.0

    # Classic bom settings (pooling_type == "bom" only).
    k_mer_size = 5
    bom_stride = 1
    bom_summary = "mean_max"      # alt: "mean" | "max"
    bom_inner_dim: int | None = None
    bom_output_dim: int | None = None

    n_outer_folds = 5
    inner_val_fraction = 0.10   # used only when cv_scheme == "random"
    random_state = 42

    test_mode = False             # alt: True (quick smoke run in train.py)
    wandb_project = "deeploc2_5fold_bce300_mcc"  # RUN 1: 300M | RUN 2: deeploc2_5fold_bce600_mcc
    # wandb_project = "deeploc2_5fold_bce600_mcc"
    wandb_mode = "online"                      # active: online | alt: "offline" | "disabled"
    # wandb_mode = "offline"
    # wandb_mode = "disabled"
    wandb_log_monitor = True
    wandb_log_monitor_as_charts = False        # alt: True (many W&B line charts)
    wandb_log_params_every = 1
    wandb_log_param_histograms = False
    wandb_log_param_histogram_every = 10


def build_run_tag(nproc_per_node: int | None = None, cfg: "Config | None" = None) -> str:
    """Short filesystem-safe tag for logs, Slurm job names, and result dirs."""
    c = cfg if cfg is not None else config
    nproc = c.nproc_per_node if nproc_per_node is None else nproc_per_node
    if c.run_tag:
        return str(c.run_tag).strip()

    parts: list[str] = []
    bslug = backbone_slug(c)
    if bslug:
        parts.append(bslug)
    if c.use_lora_model:
        parts.append("lora")
        n = c.lora_last_n_layers
        if n is None or int(n) <= 0:
            parts.append("loraall")
        else:
            parts.append(f"loran{int(n)}")
    else:
        parts.append("full")
    parts.append(str(c.pooling_type))
    agg = (c.layer_aggregation or "all").lower()
    if agg == "last_n":
        parts.append(f"agglast{int(c.layer_agg_n)}")
    elif agg == "band":
        a, b = c.layer_agg_band
        parts.append(f"aggband{int(a)}-{int(b)}")
    if c.membrane_only:
        parts.append("membrane")
    if c.folds_to_run:
        fids = "".join(str(int(i)) for i in c.folds_to_run)
        parts.append(f"folds{fids}")
    if not c.eval_external_test:
        parts.append("noext")
    if c.regenerate_splits:
        parts.append("regen")
    if c.data_summary_only:
        parts.append("summary")
    loss = (getattr(c, "loss_type", "bce") or "bce").lower()
    if loss == "focal":
        g = float(getattr(c, "focal_gamma", 2.0))
        gtag = str(int(g)) if g == int(g) else str(g).replace(".", "p")
        parts.append(f"focalg{gtag}")
        if not getattr(c, "focal_use_alpha", True):
            parts.append("noalpha")
        elif (getattr(c, "focal_alpha_mode", "manual") or "manual").lower() == "inv_freq":
            parts.append("alphainvfreq")
    elif loss == "bce":
        parts.append("bce")
    if getattr(c, "tune_thresholds", True):
        met = (getattr(c, "threshold_tune_metric", "f1") or "f1").lower()
        spl = (getattr(c, "threshold_tune_split", "val") or "val").lower()
        if met != "f1" or spl != "val":
            parts.append(f"thr{met}_{spl}")
    if int(nproc) != 2:
        parts.append(f"gpu{int(nproc)}")
    return "_".join(parts)


def resolve_output_dir(run_tag: str | None = None, cfg: "Config | None" = None) -> str:
    """``cv_results_deeploc_<run_tag>`` under the project root."""
    c = cfg if cfg is not None else config
    tag = run_tag or build_run_tag(cfg=c)
    return os.path.join(_ROOT, f"cv_results_deeploc_{tag}")


def sync_run_paths(cfg: "Config | None" = None) -> tuple[str, str]:
    """Set ``run_tag`` and ``output_dir`` on *cfg* from current experiment fields."""
    c = cfg if cfg is not None else config
    if not c.run_tag:
        c.run_tag = build_run_tag(cfg=c)
    c.output_dir = resolve_output_dir(c.run_tag, cfg=c)
    return c.run_tag, c.output_dir


def apply_run_snapshot(data: dict, cfg: "Config | None" = None) -> None:
    """Apply a frozen snapshot written at ``submit_multilabel_cv.sh`` time."""
    c = cfg if cfg is not None else config
    skip = {"run_id", "snapshot_at_utc", "inner_val_strategy"}
    for key, val in data.items():
        if key in skip or not hasattr(c, key):
            continue
        if key == "layer_agg_band" and val is not None:
            c.layer_agg_band = (int(val[0]), int(val[1]))
        elif key == "folds_to_run":
            c.folds_to_run = list(val) if val else None
        else:
            setattr(c, key, val)
    if data.get("run_tag"):
        c.run_tag = str(data["run_tag"])
    if data.get("output_dir"):
        c.output_dir = str(data["output_dir"])


def apply_run_snapshot_from_env(cfg: "Config | None" = None) -> bool:
    """Load ``log/run_snapshots/<RUN_TAG>.json`` if ``RUN_TAG`` is in the environment."""
    import os

    tag = os.environ.get("RUN_TAG", "").strip()
    if not tag:
        return False
    snap = Path(_ROOT) / "log" / "run_snapshots" / f"{tag}.json"
    if not snap.is_file():
        return False
    with open(snap, encoding="utf-8") as f:
        apply_run_snapshot(json.load(f), cfg=cfg)
    return True


def training_cli_argv(cfg: "Config | None" = None) -> list[str]:
    """Deprecated: Slurm launches ``train.py`` with no CLI flags; config.py is authoritative."""
    return []


def run_settings_dict(cfg: "Config | None" = None) -> dict:
    """Snapshot of experiment fields for logging / ``run_config.json``."""
    c = cfg if cfg is not None else config
    tag = build_run_tag(cfg=c)
    return {
        "run_tag": tag,
        "output_dir": resolve_output_dir(tag, cfg=c),
        "label_csv": c.label_csv,
        "test_csv": c.test_csv,
        "membrane_only": c.membrane_only,
        "regenerate_splits": c.regenerate_splits,
        "data_summary_only": c.data_summary_only,
        "eval_external_test": c.eval_external_test,
        "use_lora_model": c.use_lora_model,
        "lora_last_n_layers": c.lora_last_n_layers,
        "pooling_type": c.pooling_type,
        "layer_aggregation": c.layer_aggregation,
        "layer_agg_n": c.layer_agg_n,
        "layer_agg_band": list(c.layer_agg_band),
        "folds_to_run": list(c.folds_to_run) if c.folds_to_run else None,
        "nproc_per_node": c.nproc_per_node,
        "patience": c.patience,
        "early_stop_metric": c.early_stop_metric,
        "early_stop_min_delta": c.early_stop_min_delta,
        "batch_size": c.batch_size,
        "lr_esmc": c.lr_esmc,
        "lr_classifier": c.lr_classifier,
        "model_name_or_path": c.model_name_or_path,
        "embed_dim": c.embed_dim,
        "loss_type": c.loss_type,
        "focal_gamma": c.focal_gamma,
        "focal_use_alpha": c.focal_use_alpha,
        "focal_alpha_mode": c.focal_alpha_mode,
        "focal_use_pos_weight": c.focal_use_pos_weight,
        "inner_val_strategy": c.cv_scheme,
        "cv_scheme": c.cv_scheme,
        "tune_thresholds": c.tune_thresholds,
        "threshold_tune_metric": getattr(c, "threshold_tune_metric", "f1"),
        "threshold_tune_split": getattr(c, "threshold_tune_split", "val"),
        "wandb_project": c.wandb_project,
        "wandb_mode": c.wandb_mode,
        "wandb_log_monitor": c.wandb_log_monitor,
        "wandb_log_monitor_as_charts": c.wandb_log_monitor_as_charts,
        "run_id": c.run_id,
    }


config = Config()
sync_run_paths(cfg=config)
Path(config.splits_root).mkdir(parents=True, exist_ok=True)
