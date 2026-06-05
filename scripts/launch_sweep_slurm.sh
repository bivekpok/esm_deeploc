#!/bin/bash
# Submit N Slurm agents for a W&B sweep. Each agent runs one trial (train.py)
# on 2 GPUs / 1 node. Each trial trains a single outer fold (fold 1) and finishes.
#
# Usage:
#   bash scripts/launch_sweep_slurm.sh <sweep_id> [num_agents]
#
#   sweep_id    : full sweep id printed by `wandb sweep ...`,
#                 e.g. udel/deeploc2_paper_cv_sweep/abc123
#   num_agents  : optional, defaults to 1
#
# Each agent occupies one Slurm job. For a 3-value grid sweep, use 3 agents
# (one per trial) so they run in parallel.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <sweep_id> [num_agents]" >&2
    exit 1
fi

SWEEP_ID="$1"
NUM_AGENTS="${2:-1}"
PROJECT_DIR="${PROJECT_DIR:-/work/hdd/bdja/bpokhrel/deeploc}"
LOG_DIR="${PROJECT_DIR}/log"
mkdir -p "${LOG_DIR}"

for ((i = 1; i <= NUM_AGENTS; i++)); do
    JOB_NAME="wandb_sweep_$(echo "${SWEEP_ID}" | awk -F/ '{print $NF}')_a${i}"
    sbatch \
        --job-name="${JOB_NAME}" \
        --output="${LOG_DIR}/${JOB_NAME}.%j.%N.out" \
        --error="${LOG_DIR}/${JOB_NAME}.%j.%N.err" \
        --partition=gpuA100x4 \
        --account=bdja-delta-gpu \
        --mem=120G \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=16 \
        --gpus-per-node=2 \
        --time=24:00:00 \
        --chdir="${PROJECT_DIR}" \
        --wrap="\
            module purge 2>/dev/null || true; \
            export NUMEXPR_MAX_THREADS=8 TOKENIZERS_PARALLELISM=false; \
            export WANDB_DIR=/tmp WANDB_CACHE_DIR=/tmp; \
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
            source /u/bpokhrel/miniconda3/etc/profile.d/conda.sh; \
            conda activate /u/bpokhrel/miniconda3/envs/esm; \
            cd ${PROJECT_DIR}; \
            wandb agent --count 1 ${SWEEP_ID}"
done

echo "Submitted ${NUM_AGENTS} agent job(s) for sweep ${SWEEP_ID}."
