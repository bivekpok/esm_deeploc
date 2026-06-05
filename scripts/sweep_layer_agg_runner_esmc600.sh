#!/bin/bash
# W&B sweep wrapper — ESMC-600M layer-aggregation ablation, Outer_Fold_1 only.
# Same protocol as sweep_layer_agg_runner.sh (300M) but adds --backbone esmc_600m
# so outputs land under cv_results_deeploc_esmc600_* (same splits + hpa_testset.csv).
#
# Accepts W&B ${args}:
#   --layer_aggregation {all|last_n|band}
#   --layer_agg_n N
#   --layer_agg_band_start S
#   --layer_agg_band_end E

set -euo pipefail

LAYER_AGG="all"
LAYER_AGG_N=5
# ESMC-600M has 36 blocks (0..35); 300M sweep used (20, 29) on 30 blocks.
BAND_START=20
BAND_END=35

while [[ $# -gt 0 ]]; do
    case "$1" in
        --layer_aggregation=*) LAYER_AGG="${1#*=}"; shift ;;
        --layer_aggregation)   LAYER_AGG="$2"; shift 2 ;;
        --layer_agg_n=*)       LAYER_AGG_N="${1#*=}"; shift ;;
        --layer_agg_n)         LAYER_AGG_N="$2"; shift 2 ;;
        --layer_agg_band_start=*) BAND_START="${1#*=}"; shift ;;
        --layer_agg_band_start)   BAND_START="$2"; shift 2 ;;
        --layer_agg_band_end=*)   BAND_END="${1#*=}"; shift ;;
        --layer_agg_band_end)     BAND_END="$2"; shift 2 ;;
        *) echo "unknown sweep arg: $1" >&2; exit 2 ;;
    esac
done

CMD=(python -u train.py
     --backbone esmc_600m
     --lora --lora-last-n 5
     --pooling bom_attn
     --folds 1
     --layer-aggregation "${LAYER_AGG}"
     --layer-agg-n "${LAYER_AGG_N}"
     --layer-agg-band "${BAND_START}" "${BAND_END}")

echo "[sweep_runner_esmc600] ${CMD[*]}"
exec "${CMD[@]}"
