#!/bin/bash
# Run ONE layer-aggregation trial (no W&B sweep). Outer_Fold_1, paper CV, HPA test.
#
# Usage:
#   bash scripts/run_layer_agg_single.sh 300m all
#   bash scripts/run_layer_agg_single.sh 300m last_n
#   bash scripts/run_layer_agg_single.sh 300m band
#   bash scripts/run_layer_agg_single.sh 600m all
#   bash scripts/run_layer_agg_single.sh 600m last_n
#   bash scripts/run_layer_agg_single.sh 600m band
#
# Optional env:
#   NPROC_PER_NODE=2   # default 2; use 1 if OOM on 600M
#
# Outputs:
#   300M -> cv_results_deeploc_lora_loran5_bom_attn_folds1_<agg>/<run_id>/
#   600M -> cv_results_deeploc_esmc600_lora_loran5_bom_attn_folds1_<agg>/<run_id>/

set -euo pipefail

BACKBONE="${1:?Usage: $0 300m|600m all|last_n|band}"
AGG="${2:?Usage: $0 300m|600m all|last_n|band}"
NPROC="${NPROC_PER_NODE:-2}"

case "${BACKBONE}" in
    300m|300M|esmc_300m|esmc300) BACKBONE="300m" ;;
    600m|600M|esmc_600m|esmc600) BACKBONE="600m" ;;
    *) echo "Unknown backbone: ${BACKBONE} (use 300m or 600m)" >&2; exit 2 ;;
esac

case "${AGG}" in
    all|last_n|band) ;;
    *) echo "Unknown aggregation: ${AGG} (use all, last_n, or band)" >&2; exit 2 ;;
esac

PROJECT_DIR="${PROJECT_DIR:-/work/hdd/bdja/bpokhrel/deeploc}"
cd "${PROJECT_DIR}"

source /u/bpokhrel/miniconda3/etc/profile.d/conda.sh
conda activate /u/bpokhrel/miniconda3/envs/esm

LAYER_AGG_N=5
if [[ "${BACKBONE}" == "300m" ]]; then
    BAND_START=20
    BAND_END=29
    RUNNER_TAG="300m"
    EXTRA=( )
else
    BAND_START=20
    BAND_END=35
    RUNNER_TAG="600m"
    EXTRA=(--backbone esmc_600m)
fi

CMD=(python -u train.py
     "${EXTRA[@]}"
     --lora --lora-last-n 5
     --pooling bom_attn
     --folds 1
     --layer-aggregation "${AGG}"
     --layer-agg-n "${LAYER_AGG_N}"
     --layer-agg-band "${BAND_START}" "${BAND_END}")

echo "[run_layer_agg_single ${RUNNER_TAG} ${AGG}] NPROC=${NPROC}"
echo "[run_layer_agg_single] ${CMD[*]}"

if [[ "${NPROC}" -gt 1 ]]; then
    exec torchrun --standalone --nproc_per_node="${NPROC}" "${CMD[@]}"
else
    exec "${CMD[@]}"
fi
