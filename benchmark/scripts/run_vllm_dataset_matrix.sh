#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="${BENCHMARK_DIR}/scripts/run_vllm_benchmark.sh"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-1.7b-local}"
BACKENDS="${BACKENDS:-FORK_ATTN}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/fork_dataset_matrix_$(date +%Y%m%d_%H%M%S)}"
AGENTBOARD_PATH="${AGENTBOARD_PATH:-${BENCHMARK_DIR}/data/agentboard.jsonl}"
APPWORLD_PATH="${APPWORLD_PATH:-${BENCHMARK_DIR}/data/appworld.jsonl}"
CAPTURE_BUCKETS="${CAPTURE_BUCKETS:-common:4,8;forest:256,512,1024}"

DATASETS=(
  "agentboard|${AGENTBOARD_PATH}"
  "appworld|${APPWORLD_PATH}"
)

for spec in "${DATASETS[@]}"; do
  IFS="|" read -r dataset data_path <<<"${spec}"
  if [[ ! -d "${data_path}" ]]; then
    echo "Dataset directory does not exist: ${data_path}" >&2
    exit 1
  fi
  for capture_mode in dense sparse; do
    capture_config=""
    if [[ "${capture_mode}" == "sparse" ]]; then
      capture_config="${CAPTURE_BUCKETS}"
    fi
    DATASET="${dataset}" \
    DATA_PATH="${data_path}" \
    MODEL_PATH="${MODEL_PATH}" \
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
    BACKENDS="${BACKENDS}" \
    SAMPLE_COUNT=4 \
    CASE_COUNT=4 \
    PREFIX_TOKENS=8192 \
    BRANCHES=8 \
    BRANCH_GROUP_SIZE=8 \
    CONCURRENCY=32 \
    SUFFIX_DISTRIBUTION=equal \
    SUFFIX_MEAN=128 \
    OUTPUT_TOKENS=128 \
    COMMON_ANALYSIS_TOKENS=32 \
    MAX_NUM_SEQS=32 \
    MAX_MODEL_LEN=32768 \
    GPU_MEMORY_UTILIZATION=0.70 \
    PROFILE_FORK=1 \
    VLLM_FORK_ATTN_ENABLE_FOREST=1 \
    VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH=1 \
    VLLM_FORK_ATTN_FOREST_CTA_BUCKETS=64,128,256,384,512,768,1024 \
    VLLM_FORK_ATTN_CUDAGRAPH_CAPTURE_BUCKETS="${capture_config}" \
    OUTPUT_DIR="${OUTPUT_ROOT}/${dataset}/${capture_mode}" \
    "${RUN_SCRIPT}"
  done
done

echo "Dataset matrix complete: ${BENCHMARK_DIR}/${OUTPUT_ROOT}"
