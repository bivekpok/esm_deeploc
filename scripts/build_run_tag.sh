#!/bin/bash
# Build a short, filesystem-safe tag from EXTRA_ARGS (and optional GPU count).
# Source this file:  source "${PROJECT_DIR}/scripts/build_run_tag.sh"
#
#   RUN_TAG="$(build_run_tag "${EXTRA_ARGS}" "${NPROC_PER_NODE:-2}")"

build_run_tag() {
    local args="${1:-}"
    local nproc="${2:-2}"
    local parts=()

    if [[ "${args}" == *"--lora"* ]]; then
        parts+=("lora")
        if [[ "${args}" =~ --lora-last-n[[:space:]]+([0-9]+) ]]; then
            local n="${BASH_REMATCH[1]}"
            if [[ "${n}" == "0" ]]; then
                parts+=("loraall")
            else
                parts+=("loran${n}")
            fi
        else
            parts+=("loran5")
        fi
    else
        parts+=("full")
    fi

    if [[ "${args}" =~ --pooling[[:space:]]+(attention|average|bom|bom_attn) ]]; then
        parts+=("${BASH_REMATCH[1]}")
    else
        parts+=("bom")
    fi

    if [[ "${args}" == *"--membrane-only"* ]]; then
        parts+=("membrane")
    fi
    if [[ "${args}" == *"--no-external-test"* ]]; then
        parts+=("noext")
    fi
    if [[ "${args}" == *"--regenerate-splits"* ]]; then
        parts+=("regen")
    fi
    if [[ "${args}" == *"--data-summary-only"* ]]; then
        parts+=("summary")
    fi
    if [[ "${nproc}" != "2" ]]; then
        parts+=("gpu${nproc}")
    fi

    local IFS=_
    echo "${parts[*]}"
}
