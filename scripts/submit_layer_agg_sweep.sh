#!/bin/bash
# Create a W&B layer-aggregation sweep and submit Slurm agents (3 trials: all/last_n/band).
#
# Usage:
#   bash scripts/submit_layer_agg_sweep.sh 300m
#   bash scripts/submit_layer_agg_sweep.sh 600m
#   bash scripts/submit_layer_agg_sweep.sh 600m 1   # one agent at a time
#
# Change the W&B sweep display name: edit the ``name:`` field in
#   sweeps/layer_aggregation.yaml          (300M)
#   sweeps/layer_aggregation_esmc600.yaml  (600M)

set -euo pipefail

BACKBONE="${1:?Usage: $0 300m|600m [num_agents]}"
NUM_AGENTS="${2:-3}"
PROJECT_DIR="${PROJECT_DIR:-/work/hdd/bdja/bpokhrel/deeploc}"

case "${BACKBONE}" in
    300m|300M|esmc_300m) SWEEP_YAML="sweeps/layer_aggregation.yaml" ;;
    600m|600M|esmc_600m) SWEEP_YAML="sweeps/layer_aggregation_esmc600.yaml" ;;
    *) echo "Unknown backbone: ${BACKBONE} (use 300m or 600m)" >&2; exit 2 ;;
esac

cd "${PROJECT_DIR}"
source /u/bpokhrel/miniconda3/etc/profile.d/conda.sh
conda activate /u/bpokhrel/miniconda3/envs/esm

echo "Creating W&B sweep from ${SWEEP_YAML} ..."
SWEEP_OUT="$(wandb sweep --project deeploc2_paper_cv_sweep "${SWEEP_YAML}" 2>&1 | tee /dev/stderr)"
SWEEP_ID="$(echo "${SWEEP_OUT}" | sed -n 's/.*Run sweep agent with: wandb agent //p' | tr -d '[:space:]')"

if [[ -z "${SWEEP_ID}" ]]; then
    echo "Could not parse sweep id from wandb output." >&2
    exit 1
fi

echo "Submitting ${NUM_AGENTS} Slurm agent(s) for ${SWEEP_ID} ..."
bash scripts/launch_sweep_slurm.sh "${SWEEP_ID}" "${NUM_AGENTS}"
