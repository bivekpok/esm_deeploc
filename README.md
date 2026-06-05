# DeepLoc 2.0 Multilabel (ESMC + LoRA)

Repository: [github.com/bivekpok/esm_deeploc](https://github.com/bivekpok/esm_deeploc)

Paper-aligned 5-fold cross-validation for multilabel protein subcellular localization using ESMC-300M / ESMC-600M with LoRA and bag-of-motifs attention pooling.

## Data (not in repo)

Place these files in the project root (from [DeepLoc 2.0](https://services.healthtech.dtu.dk/service.php?DeepLoc-2.0)):

- `Swissprot_Train_Validation_dataset.csv`
- `hpa_testset.csv`

CV split manifests are included under `cv_splits_deeploc/` (paper scheme).

## Environment

```bash
conda activate esm   # /u/bpokhrel/miniconda3/envs/esm
```

Requires: `torch`, `esm`, `peft`, `wandb`, `scikit-learn`, `pandas`.

## Run

Edit `config.py`, then from the login node:

```bash
./submit_multilabel_cv.sh
```

**ESMC-300M (default):** `model_name_or_path = "esmc_300m"`, `embed_dim = 960`

**ESMC-600M:** swap to `esmc_600m` / `embed_dim = 1152`, set `nproc_per_node = 1` if OOM.

Config is frozen at submit time via `log/run_snapshots/<RUN_TAG>.json`.

## Successful 5-fold runs (reference)

| Backbone | Run tag | Run ID | HPA metric |
|----------|---------|--------|------------|
| ESMC-300M | `lora_loran5_bom_attn_bce_thrmcc_train` | `20260601-021931Z-80cf86` | BCE + MCC thresholds on train |
| ESMC-600M | `esmc600_lora_loran5_bom_attn_bce_thrmcc_train` | `20260529-233846Z-870119` | same protocol |

Results: `cv_results_deeploc_<run_tag>/<run_id>/`

## Key files

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters (single source of truth) |
| `train.py` | 5-fold CV training + HPA eval |
| `model.py` | ESMC + LoRA + pooling + classifier |
| `dataset.py` | Manifests, loaders, paper CV splits |
| `utils.py` | Metrics, MCC threshold tuning |
| `test_hpa.py` | Standalone HPA checkpoint evaluation |
