#!/bin/bash
# Print shell variables from config.py for submit/run scripts.
# Usage:  eval "$(scripts/emit_run_env.sh)"

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" && -f /u/bpokhrel/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck source=/dev/null
    source /u/bpokhrel/miniconda3/etc/profile.d/conda.sh
    conda activate /u/bpokhrel/miniconda3/envs/esm 2>/dev/null || true
    PYTHON="$(command -v python)"
fi
PYTHON="${PYTHON:-python3}"

"${PYTHON}" - <<'PY'
import shlex
from config import (
    build_run_tag,
    config,
    resolve_output_dir,
    run_settings_dict,
)

nproc_raw = __import__("os").environ.get("NPROC_PER_NODE", "").strip()
nproc = int(nproc_raw) if nproc_raw else int(config.nproc_per_node)
tag = build_run_tag(nproc_per_node=nproc)
out = resolve_output_dir(tag)

print(f"RUN_TAG={shlex.quote(tag)}")
print(f"OUTPUT_DIR={shlex.quote(out)}")
print(f"NPROC_PER_NODE={nproc}")
# Human-readable one-liner for Slurm .out
parts = [f"{k}={v!r}" for k, v in run_settings_dict().items()]
print(f"CONFIG_SUMMARY={shlex.quote('; '.join(parts[:12]) + '...')}")
PY

# Expose the same interpreter for write_run_snapshot.py on the login node.
printf 'PYTHON=%q\n' "${PYTHON}"
