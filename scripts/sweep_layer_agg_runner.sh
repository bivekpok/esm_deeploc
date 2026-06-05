#!/bin/bash
# Wrapper used by the W&B sweep. The agent invokes this script and passes the
# sampled parameters as CLI flags (because we use ${args} in the sweep yaml).
# We accept:
#   --layer_aggregation {all|last_n|band}
#   --layer_agg_n N
#   --layer_agg_band_start S
#   --layer_agg_band_end E
# and translate them to the flags train.py actually expects.
#
# All other train.py flags (LoRA, pooling, fold) are added here as constants,
# so the sweep config only varies the three things we want to ablate.

set -euo pipefail

LAYER_AGG="all"
LAYER_AGG_N=5
BAND_START=20
BAND_END=29

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
     --lora --lora-last-n 5
     --pooling bom_attn
     --folds 1
     --layer-aggregation "${LAYER_AGG}"
     --layer-agg-n "${LAYER_AGG_N}"
     --layer-agg-band "${BAND_START}" "${BAND_END}")

echo "[sweep_runner] ${CMD[*]}"
exec "${CMD[@]}"
