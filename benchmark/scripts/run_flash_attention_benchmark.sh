#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DATASET="${DATASET:-swebench}"
CASE_COUNT="${CASE_COUNT:-1}"
PREFIX_TOKENS="${PREFIX_TOKENS:-2048}"
BRANCHES="${BRANCHES:-2}"
BRANCH_GROUP_SIZE="${BRANCH_GROUP_SIZE:-1}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"

export BACKENDS="${BACKENDS:-FLASH_ATTN}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/flash_attention_${DATASET}_c${CASE_COUNT}_p${PREFIX_TOKENS}_b${BRANCHES}_g${BRANCH_GROUP_SIZE}_o${OUTPUT_TOKENS}}"

exec "${SCRIPT_DIR}/run_vllm_benchmark.sh" "$@"
