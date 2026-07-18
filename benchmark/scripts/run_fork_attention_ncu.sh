#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
VLLM_PYTHON="${VLLM_PYTHON:-${REPO_ROOT}/vllm/.venv/bin/python}"
NCU_BIN="${NCU_BIN:-/usr/local/cuda-13.1/bin/ncu}"
BACKENDS="${BACKENDS:-FLASH_ATTN,FORK_ATTN}"
PREFIX_TOKENS="${PREFIX_TOKENS:-8192}"
PRIVATE_SUFFIX_TOKENS="${PRIVATE_SUFFIX_TOKENS:-128}"
PREFIX_CHUNK_TOKENS="${PREFIX_CHUNK_TOKENS:-0}"
BRANCHES="${BRANCHES:-16}"
OUTPUT_DIR="${OUTPUT_DIR:-${BENCHMARK_DIR}/results/ncu_forkattention_operator_8k16}"
KERNEL_REGEX=".*(flash_fwd|fork_fwd|gather_kernel|merge_attn_states_kernel).*"

group_names=(dram l2_total l2_hit l2_miss instructions)
group_metrics=(
  "dram__bytes_op_read.sum,dram__sectors_op_read.sum"
  "lts__t_sectors_op_read.sum"
  "lts__t_sectors_op_read_lookup_hit.sum"
  "lts__t_sectors_op_read_lookup_miss.sum"
  "smsp__inst_executed_pipe_tensor.sum,smsp__inst_executed_pipe_fma.sum"
)

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}/vllm"
IFS=',' read -r -a backend_list <<<"${BACKENDS}"
for backend in "${backend_list[@]}"; do
  backend_lower="$(tr '[:upper:]' '[:lower:]' <<<"${backend}")"
  raw_csvs=()
  for index in "${!group_names[@]}"; do
    group="${group_names[index]}"
    metrics="${group_metrics[index]}"
    report="${OUTPUT_DIR}/${backend_lower}_${group}"
    log="${OUTPUT_DIR}/${backend_lower}_${group}.log"
    raw_csv="${OUTPUT_DIR}/${backend_lower}_${group}.csv"
    echo "Profiling ${backend} operator counters: ${group}"
    "${NCU_BIN}" \
      --force-overwrite \
      --export "${report}" \
      --target-processes application-only \
      --profile-from-start off \
      --replay-mode kernel \
      --kernel-name-base function \
      --kernel-name "regex:${KERNEL_REGEX}" \
      --metrics "${metrics}" \
      "${VLLM_PYTHON}" \
        "${BENCHMARK_DIR}/scripts/fork_attention_operator_ncu.py" \
        --attention-backend "${backend}" \
        --prefix-tokens "${PREFIX_TOKENS}" \
        --private-suffix-tokens "${PRIVATE_SUFFIX_TOKENS}" \
        --prefix-chunk-tokens "${PREFIX_CHUNK_TOKENS}" \
        --branches "${BRANCHES}" \
      >"${log}" 2>&1
    "${NCU_BIN}" \
      --import "${report}.ncu-rep" \
      --page raw \
      --csv \
      --print-units base \
      --print-fp \
      >"${raw_csv}"
    raw_csvs+=("${raw_csv}")
  done

  "${VLLM_PYTHON}" \
    "${BENCHMARK_DIR}/scripts/summarize_fork_attention_ncu.py" \
    "${raw_csvs[@]}" \
    --backend "${backend}" \
    --output-tokens "${BRANCHES}" \
    --output "${OUTPUT_DIR}/${backend_lower}_summary.json"
done

echo "Nsight Compute operator artifacts: ${OUTPUT_DIR}"
