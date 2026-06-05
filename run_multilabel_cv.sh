#!/bin/bash
# =============================================================================
# run_multilabel_cv.sh — Slurm GPU job script (runs on a COMPUTE node)
# =============================================================================
#
# WHAT THIS SCRIPT DOES
#   - Loads modules / conda (esm env)
#   - Runs one CV session: train.py loops Outer_Fold_* internally (paper 5-fold)
#   - Uses torchrun when NPROC_PER_NODE > 1, else plain python
#   - Writes training stdout/stderr to log/train_<RUN_TAG>_main.{out,err}
#
# HOW YOU USUALLY INVOKE IT
#   Do NOT run this directly on the login node for normal experiments.
#   Use:  ./submit_multilabel_cv.sh
#   submit_multilabel_cv.sh calls sbatch on this file and passes RUN_TAG,
#   OUTPUT_DIR, NPROC_PER_NODE from config.py.
#
# DIFFERENCE FROM submit_multilabel_cv.sh
#   submit_multilabel_cv.sh     run_multilabel_cv.sh
#   -------------------------   ---------------------
#   Login node                  Compute node (inside Slurm allocation)
#   Reads config.py             Receives env vars from submit (or re-reads config)
#   sbatch + job naming         conda, torchrun, train.py
#   Short — queues the job      Long — actual GPU training
#
# EXPERIMENT SETTINGS
#   All hyperparameters live in config.py. train.py reads config at startup.
#   No CLI flags are passed from these shell scripts (no EXTRA_ARGS).
#
# ENV VARS (set by submit_multilabel_cv.sh, or derived here if missing)
#   RUN_TAG        — e.g. lora_loran5_bom_attn_folds1_focalg2
#   OUTPUT_DIR     — e.g. cv_results_deeploc_<RUN_TAG>/
#   NPROC_PER_NODE — GPUs for DDP (default 2 from config; use 1 if OOM)
#   PROJECT_DIR    — repo root (default below)
#
# OUTPUT LAYOUT
#   ${OUTPUT_DIR}/<run_id>/Outer_Fold_k/checkpoint.pt
#   ${OUTPUT_DIR}/<run_id>/Outer_Fold_k/...  (metrics, W&B, HPA eval)
#
# DIRECT sbatch (if submit wrapper not used):
#   sbatch run_multilabel_cv.sh
#   Job name falls back to #SBATCH defaults unless you pass --job-name.
# =============================================================================
#
# ---------------------------------------------------------------------------
# SBATCH directives (defaults; submit_multilabel_cv.sh overrides job-name + log paths)
# ---------------------------------------------------------------------------
#SBATCH --job-name="deeploc_multilabel_cv"
#SBATCH --output="/work/hdd/bdja/bpokhrel/deeploc/log/deeploc_multilabel_cv.%j.%N.out"
#SBATCH --error="/work/hdd/bdja/bpokhrel/deeploc/log/deeploc_multilabel_cv.%j.%N.err"
# Prefer:  ./submit_multilabel_cv.sh  (reads experiment flags from config.py)
#   -> job-name deeploc_<run_tag>, matching Slurm + train logs + cv_results_* dirs
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdja-delta-gpu
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mail-user='bivekpok@udel.edu'
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH -t 48:00:00
# Slurm may execute a *copy* of this script from /var/spool/slurmd/... — do not rely on
# BASH_SOURCE for the repo path. Force job cwd to the project (Slurm 20.11+).
#SBATCH --chdir=/work/hdd/bdja/bpokhrel/deeploc

set -euo pipefail

# ---------------------------------------------------------------------------
# Project root (see SLURM chdir note above)
# ---------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-/work/hdd/bdja/bpokhrel/deeploc}"
cd "${PROJECT_DIR}" || exit 1
echo "PROJECT_DIR=${PROJECT_DIR}"

ENTRY="train.py"
if [ ! -f "${PROJECT_DIR}/${ENTRY}" ]; then
    echo "ERROR: ${PROJECT_DIR}/${ENTRY} not found." >&2
    exit 1
fi
echo "ENTRY=${ENTRY}"

# ---------------------------------------------------------------------------
# Environment (compute node)
# ---------------------------------------------------------------------------
module purge 2>/dev/null || true
export NUMEXPR_MAX_THREADS=8
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR=/tmp
export WANDB_CACHE_DIR=/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

source /u/bpokhrel/miniconda3/etc/profile.d/conda.sh
conda activate /u/bpokhrel/miniconda3/envs/esm

LOG_DIR="${PROJECT_DIR}/log"
mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Resolve RUN_TAG / OUTPUT_DIR if submit_multilabel_cv.sh did not export them
# (e.g. direct sbatch run_multilabel_cv.sh). Same source: config.py via emit_run_env.sh
# ---------------------------------------------------------------------------
if [[ -z "${RUN_TAG:-}" || -z "${OUTPUT_DIR:-}" ]]; then
    echo "NOTE: RUN_TAG/OUTPUT_DIR not in env — reading from config.py"
    # shellcheck source=scripts/emit_run_env.sh
    eval "$(NPROC_PER_NODE="${NPROC_PER_NODE:-}" PROJECT_DIR="${PROJECT_DIR}" bash "${PROJECT_DIR}/scripts/emit_run_env.sh")"
fi
NPROC="${NPROC_PER_NODE:-2}"
MAIN_LOG="${LOG_DIR}/train_${RUN_TAG}_main.out"
MAIN_ERR="${LOG_DIR}/train_${RUN_TAG}_main.err"

echo "Starting ${ENTRY} (DeepLoc 2.0 multilabel CV) with torchrun --nproc_per_node=${NPROC} ..."
echo "RUN_TAG=${RUN_TAG}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "Config: config.py (no CLI overrides)"
echo "Logs: ${MAIN_LOG} / ${MAIN_ERR}"
echo "Per-fold outputs: ${OUTPUT_DIR}/<run_id>/Outer_Fold_*/"
echo "=========================================="

if [ "${NPROC}" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="${NPROC}" "${ENTRY}" \
        >"${MAIN_LOG}" 2>"${MAIN_ERR}"
else
    python -u "${ENTRY}" \
        >"${MAIN_LOG}" 2>"${MAIN_ERR}"
fi

echo "=========================================="
echo "${ENTRY} finished. Check ${MAIN_LOG} and ${MAIN_ERR};"
echo "predictions under ${OUTPUT_DIR}/<run_id>/Outer_Fold_*/ (partition val + HPA test)."
