#!/usr/bin/env bash
set -Eeuo pipefail

# Produce one rich, UI-oriented Nsight Compute report per attention arm. These
# reports intentionally use the full section set and kernel replay. Use the
# split metric reports from run_fork_cascade_ncu.sh for low-interference exact
# cross-backend counter aggregation.

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
VLLM_PYTHON="${VLLM_PYTHON:-${REPO_ROOT}/vllm/.venv/bin/python}"
NCU_BIN="${NCU_BIN:-/usr/local/cuda-13.1/bin/ncu}"
OUTPUT_DIR="${OUTPUT_DIR:-${BENCHMARK_DIR}/results/fork_cascade_ncu_ui}"
PREFIX_TOKENS="${PREFIX_TOKENS:-4096}"
PRIVATE_SUFFIX_TOKENS="${PRIVATE_SUFFIX_TOKENS:-128}"
PREFIX_CHUNK_TOKENS="${PREFIX_CHUNK_TOKENS:-0}"
BRANCHES="${BRANCHES:-8}"
KERNEL_REGEX=".*(flash_fwd|FlashAttnFwd|prepare_varlen_num_blocks_kernel|fork_fwd|gather_kernel|merge_attn_states_kernel).*"

export PYTHONPATH="${REPO_ROOT}/vllm${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}/vllm"

backends=(FLASH_ATTN CASCADE_ATTN FORK_ATTN)
report_names=(flash_attention_full cascade_attention_full fork_attention_full)
for index in "${!backends[@]}"; do
  backend="${backends[index]}"
  report_name="${report_names[index]}"
  report="${OUTPUT_DIR}/${report_name}"
  log="${OUTPUT_DIR}/${report_name}.log"
  echo "Capturing rich Nsight Compute UI report for ${backend}"
  "${NCU_BIN}" \
    --force-overwrite \
    --export "${report}" \
    --target-processes application-only \
    --profile-from-start off \
    --replay-mode kernel \
    --kernel-name-base function \
    --kernel-name "regex:${KERNEL_REGEX}" \
    --set full \
    "${VLLM_PYTHON}" \
      "${BENCHMARK_DIR}/scripts/fork_attention_operator_ncu.py" \
      --attention-backend "${backend}" \
      --prefix-tokens "${PREFIX_TOKENS}" \
      --private-suffix-tokens "${PRIVATE_SUFFIX_TOKENS}" \
      --prefix-chunk-tokens "${PREFIX_CHUNK_TOKENS}" \
      --branches "${BRANCHES}" \
    >"${log}" 2>&1
done

echo "Rich Nsight Compute UI reports: ${OUTPUT_DIR}"
