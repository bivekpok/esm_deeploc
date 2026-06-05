#!/bin/bash
# =============================================================================
# submit_multilabel_cv.sh — submit a Slurm GPU job (run on the LOGIN node)
# =============================================================================
#
# WHAT THIS SCRIPT DOES
#   1. Reads experiment settings from config.py (via scripts/emit_run_env.sh)
#   2. Derives RUN_TAG, OUTPUT_DIR, and NPROC_PER_NODE
#   3. Calls sbatch on run_multilabel_cv.sh with a job name and log paths
#
# WHAT THIS SCRIPT DOES NOT DO
#   - Does not train models itself (that happens inside the Slurm job)
#   - Does not accept CLI experiment flags — edit config.py instead
#
# RELATIONSHIP TO run_multilabel_cv.sh
#   submit_multilabel_cv.sh  →  sbatch  →  run_multilabel_cv.sh  →  train.py
#   (login node)                (queue)     (compute node)           (CV loop)
#
# TYPICAL WORKFLOW
#   1. Edit config.py (loss, folds, backbone, pooling, layer agg, …)
#   2. ./submit_multilabel_cv.sh   (freezes config snapshot — safe to edit config.py while job queues)
#   3. Monitor:  squeue -u $USER
#   4. Logs:
#        Slurm wrapper:  log/deeploc_<RUN_TAG>.<jobid>.<node>.{out,err}
#        Training:       log/train_<RUN_TAG>_main.{out,err}
#   5. Results:  cv_results_deeploc_<RUN_TAG>/<run_id>/Outer_Fold_*/
#
# EXAMPLES
#   ./submit_multilabel_cv.sh
#   NPROC_PER_NODE=1 ./submit_multilabel_cv.sh    # single GPU (OOM / 600M)
#
# OPTIONAL OVERRIDES (env vars before ./submit_multilabel_cv.sh)
#   PROJECT_DIR=/path/to/deeploc
#   NPROC_PER_NODE=1|2   — GPUs for torchrun (default from config.nproc_per_node)
#
# DIRECT sbatch (advanced — skips submit wrapper; job name may be generic):
#   sbatch run_multilabel_cv.sh
#   Prefer ./submit_multilabel_cv.sh so RUN_TAG / OUTPUT_DIR match config.py.
# =============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/work/hdd/bdja/bpokhrel/deeploc}"
LOG_DIR="${PROJECT_DIR}/log"
RUN_SCRIPT="${PROJECT_DIR}/run_multilabel_cv.sh"

# Resolve RUN_TAG / OUTPUT_DIR / NPROC_PER_NODE from config.py (single source of truth).
# shellcheck source=scripts/emit_run_env.sh
eval "$(PROJECT_DIR="${PROJECT_DIR}" bash "${PROJECT_DIR}/scripts/emit_run_env.sh")"

JOB_NAME="deeploc_${RUN_TAG}"

mkdir -p "${LOG_DIR}"
SNAPSHOT_PATH="${LOG_DIR}/run_snapshots/${RUN_TAG}.json"
mkdir -p "${LOG_DIR}/run_snapshots"
"${PYTHON}" "${PROJECT_DIR}/scripts/write_run_snapshot.py" "${SNAPSHOT_PATH}"
export RUN_SNAPSHOT="${SNAPSHOT_PATH}"

# Passed into the Slurm job environment for run_multilabel_cv.sh
export RUN_TAG OUTPUT_DIR NPROC_PER_NODE RUN_SNAPSHOT

echo "Submitting: job-name=${JOB_NAME}"
echo "  (edit config.py to change experiment — no EXTRA_ARGS)"
echo "  RUN_TAG=${RUN_TAG}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  RUN_SNAPSHOT=${SNAPSHOT_PATH}"
echo "  Slurm logs: ${LOG_DIR}/${JOB_NAME}.<jobid>.<node>.{out,err}"
echo "  Train logs: ${LOG_DIR}/train_${RUN_TAG}_main.{out,err}"

sbatch \
    --job-name="${JOB_NAME}" \
    --output="${LOG_DIR}/${JOB_NAME}.%j.%N.out" \
    --error="${LOG_DIR}/${JOB_NAME}.%j.%N.err" \
    "${RUN_SCRIPT}"
