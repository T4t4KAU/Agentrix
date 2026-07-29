#!/usr/bin/env bash
set -Eeuo pipefail
BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
PYTHON="${BENCHMARK_PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-${REPO_ROOT}/vllm/.venv/bin/python}"
SOURCE_CLI="${VLLM_SOURCE_CLI:-${BENCHMARK_DIR}/scripts/vllm_source_cli.py}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
CASE_FILE="${CASE_FILE:?CASE_FILE is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
MODEL_NAME="${SERVED_MODEL_NAME:-agentrix-longbench-qwen3}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"; DP_REPLICAS="${DP_REPLICAS:-4}"; PORT="${PORT:-9000}"
SERVER_PID=""; SAMPLER_PID=""
export PYTHONPATH="${BENCHMARK_DIR}/src:${REPO_ROOT}/application/src:${REPO_ROOT}/vllm${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}"

if [[ "${ALLOW_BUSY_GPUS:-0}" != 1 ]]; then
  busy="$(nvidia-smi -i "${GPU_IDS}" --query-gpu=memory.used --format=csv,noheader,nounits |
    awk '$1 > 1024 {count++} END {print count+0}')"
  [[ "${busy}" == 0 ]] || { echo "selected GPUs are busy; refusing a confounded run" >&2; exit 2; }
fi
stop_processes() {
  if [[ -n "${SAMPLER_PID}" ]] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then
    kill -TERM "${SAMPLER_PID}" 2>/dev/null || true; wait "${SAMPLER_PID}" 2>/dev/null || true
  fi
  SAMPLER_PID=""
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_processes EXIT INT TERM
wait_server() {
  local log="$1" deadline=$((SECONDS+900))
  until curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null; do
    kill -0 "${SERVER_PID}" 2>/dev/null || { tail -n 160 "${log}" >&2; return 1; }
    ((SECONDS < deadline)) || return 1; sleep 2
  done
}
run_arm() {
  local arm="$1" backend="$2" routing="$3" blocks="$4" arm_dir="${OUTPUT_ROOT}/$1"
  mkdir -p "${arm_dir}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" FLASHINFER_DISABLE_VERSION_CHECK=1 \
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_FORK_ATTN_ENABLE_FOREST=1 \
  VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH=1 \
  VLLM_FORK_ATTN_FANOUT_SCHEDULING_ENABLED="${routing}" \
  VLLM_FORK_ATTN_DP_PREFIX_ROUTING="${routing}" VLLM_FORK_ATTN_DP_RELOAD_REBALANCE=0 \
  VLLM_FORK_ATTN_DP_ARRIVAL_WAVE_MS=10 \
  "${VLLM_PYTHON}" "${SOURCE_CLI}" serve "${MODEL_PATH}" --host 127.0.0.1 --port "${PORT}" \
    --served-model-name "${MODEL_NAME}" --attention-backend "${backend}" --dtype bfloat16 \
    --generation-config vllm --enable-prefix-caching --no-async-scheduling \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --data-parallel-size "${DP_REPLICAS}" --api-server-count 1 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.70}" \
    --num-gpu-blocks-override "${blocks}" --max-model-len "${MAX_MODEL_LEN:-40960}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}" \
    --max-num-seqs "${MAX_NUM_SEQS:-64}" >"${arm_dir}/vllm_server.log" 2>&1 &
  SERVER_PID=$!; wait_server "${arm_dir}/vllm_server.log"
  nvidia-smi --query-compute-apps=timestamp,gpu_uuid,pid,used_memory \
    --format=csv,noheader,nounits -lms 200 >"${arm_dir}/gpu_process_memory.csv" &
  SAMPLER_PID=$!
  "${PYTHON}" -m longbench_qa_runner --model "${MODEL_NAME}" --cases "${CASE_FILE}" \
    --concurrency "${CONCURRENCY:-64}" --max-tokens "${MAX_OUTPUT_TOKENS:-64}" \
    --output "${arm_dir}/run.json" | tee "${arm_dir}/runner.log"
  stop_processes
}
[[ "${RUN_BASELINE:-1}" == 1 ]] && run_arm flash_ordinary_dp FLASH_ATTN 0 "${BASELINE_NUM_GPU_BLOCKS:-3852}"
[[ "${RUN_OPTIMIZED:-1}" == 1 ]] && run_arm agentrix FORK_ATTN 1 "${AGENTRIX_NUM_GPU_BLOCKS:-2600}"
"${PYTHON}" -m longbench_qa_report --baseline "${OUTPUT_ROOT}/flash_ordinary_dp" \
  --optimized "${OUTPUT_ROOT}/agentrix" --output "${OUTPUT_ROOT}/quality_performance_ab.json"
