#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
PYTHON="${VLLM_PYTHON:-${REPO_ROOT}/vllm/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
CASE_FILE="${CASE_FILE:?CASE_FILE is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPU_BLOCKS="${NUM_GPU_BLOCKS:-2600}"
DP_REPLICAS="${DP_REPLICAS:-4}"
QUESTION_LIMIT="${QUESTION_LIMIT:-68}"
CONCURRENCY="${CONCURRENCY:-32}"

export PYTHONPATH="${BENCHMARK_DIR}/src:${REPO_ROOT}/application/src:${REPO_ROOT}/vllm${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}"

if [[ "${ALLOW_BUSY_GPUS:-0}" != 1 ]]; then
  busy="$(nvidia-smi -i "${GPU_IDS}" --query-gpu=memory.used \
    --format=csv,noheader,nounits |
    awk '$1 > 1024 {count++} END {print count+0}')"
  if [[ "${busy}" != 0 ]]; then
    echo "Selected GPUs are busy; refusing a confounded run." >&2
    exit 2
  fi
fi

run_arm() {
  local arm="$1"
  local output="${OUTPUT_ROOT}/${arm}.json"
  if [[ -s "${output}" && "${RERUN_COMPLETED:-0}" != 1 ]]; then
    echo "Skipping completed ${arm}: ${output}"
    return
  fi
  echo "Running ${arm} with exactly ${NUM_GPU_BLOCKS} GPU KV blocks per rank"
  local routing=0
  if [[ "${arm}" == "full" ]]; then
    routing=1
  fi
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  PYTHONHASHSEED=0 \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_FORK_ATTN_ENABLE_FOREST=1 \
  VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH=1 \
  VLLM_FORK_ATTN_FANOUT_SCHEDULING_ENABLED="${routing}" \
  VLLM_FORK_ATTN_DP_PREFIX_ROUTING="${routing}" \
  VLLM_FORK_ATTN_DP_RELOAD_REBALANCE=0 \
  VLLM_FORK_ATTN_DP_ARRIVAL_WAVE_MS=10 \
  "${PYTHON}" "${BENCHMARK_DIR}/scripts/benchmark_full_stack_agent.py" \
    --arm "${arm}" \
    --model "${MODEL_PATH}" \
    --cases "${CASE_FILE}" \
    --output "${output}" \
    --question-limit "${QUESTION_LIMIT}" \
    --concurrency "${CONCURRENCY}" \
    --data-parallel-size "${DP_REPLICAS}" \
    --num-gpu-blocks-override "${NUM_GPU_BLOCKS}" \
    --max-model-len "${MAX_MODEL_LEN:-40960}" \
    --max-num-seqs "${MAX_NUM_SEQS:-64}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.70}" \
    --tool-rounds "${TOOL_ROUNDS:-4}" \
    --tool-delay-ms "${TOOL_DELAY_MS:-800}" \
    --trim-grace-ms "${TRIM_GRACE_MS:-100}" \
    --action-tokens "${ACTION_TOKENS:-32}" \
    --answer-tokens "${ANSWER_TOKENS:-96}" \
    --offload-cpu-gib "${OFFLOAD_CPU_GIB:-16}" \
    >"${OUTPUT_ROOT}/${arm}.log" 2>&1
}

run_arm baseline
run_arm full

"${PYTHON}" -m full_stack_agent_report \
  --baseline "${OUTPUT_ROOT}/baseline.json" \
  --full "${OUTPUT_ROOT}/full.json" \
  --output "${OUTPUT_ROOT}/comparison.json" \
  | tee "${OUTPUT_ROOT}/report.log"

echo "Equal-KV full-stack A/B complete: ${OUTPUT_ROOT}"
