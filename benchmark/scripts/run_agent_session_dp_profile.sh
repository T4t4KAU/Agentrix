#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
VLLM_BIN="${VLLM_BIN:-${REPO_ROOT}/vllm/.venv/bin/vllm}"
PYTHON="${PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/../models/Qwen3-VL-8B-Instruct}"
MODEL_NAME="${MODEL_NAME:-qwen3-vl}"
GPU_IDS="${GPU_IDS:-0,1}"
PORT="${PORT:-8135}"
POLICIES="${POLICIES:-prefix_aware session_aware}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BENCHMARK_DIR}/results/agent_session_dp_profile}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
SESSIONS="${SESSIONS:-12}"
TRIALS="${TRIALS:-5}"
SHARED_PREFIX_TOKENS="${SHARED_PREFIX_TOKENS:-2048}"
SESSION_TOKENS="${SESSION_TOKENS:-1024}"
FOLLOWUP_TOKENS="${FOLLOWUP_TOKENS:-64}"
server_args=()
if [[ -n "${NUM_GPU_BLOCKS:-}" ]]; then
  server_args+=(--num-gpu-blocks-override "${NUM_GPU_BLOCKS}")
fi
if [[ -n "${KV_TRANSFER_CONFIG:-}" ]]; then
  server_args+=(--kv-transfer-config "${KV_TRANSFER_CONFIG}")
fi
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  server_args+=(--enforce-eager)
fi
export PATH="$(dirname "${VLLM_BIN}"):${PATH}"

SERVER_PID=""
GPU_SAMPLER_PID=""

stop_processes() {
  if [[ -n "${GPU_SAMPLER_PID}" ]] && kill -0 "${GPU_SAMPLER_PID}" 2>/dev/null; then
    kill -TERM "${GPU_SAMPLER_PID}" 2>/dev/null || true
    wait "${GPU_SAMPLER_PID}" 2>/dev/null || true
  fi
  GPU_SAMPLER_PID=""
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 60); do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -KILL -- "-${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" || true
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

wait_for_server() {
  local log="$1"
  for _ in $(seq 1 "${STARTUP_TIMEOUT}"); do
    if curl --silent --fail --max-time 2 \
      "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -n 120 "${log}" >&2
      return 1
    fi
    sleep 1
  done
  tail -n 120 "${log}" >&2
  return 1
}

trap stop_processes EXIT INT TERM
mkdir -p "${OUTPUT_ROOT}"

for policy in ${POLICIES}; do
  output_dir="${OUTPUT_ROOT}/${policy}"
  mkdir -p "${output_dir}"
  server_log="${output_dir}/vllm_server.log"
  echo "Starting ${policy} on GPUs ${GPU_IDS}"
  setsid env \
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    PYTHONHASHSEED=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_SERVER_DEV_MODE=1 \
    VLLM_AGENTRIX_DP_ROUTING_POLICY="${policy}" \
    VLLM_FORK_ATTN_ENABLE_FOREST=1 \
    VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH=1 \
    VLLM_FORK_ATTN_FANOUT_SCHEDULING_ENABLED=0 \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --served-model-name "${MODEL_NAME}" \
      --attention-backend FORK_ATTN \
      --dtype bfloat16 \
      --generation-config vllm \
      --enable-prefix-caching \
      --no-async-scheduling \
      --data-parallel-size 2 \
      --api-server-count 1 \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      "${server_args[@]}" \
      >"${server_log}" 2>&1 &
  SERVER_PID=$!
  wait_for_server "${server_log}"

  nvidia-smi dmon -i "${GPU_IDS}" -s pucvmet -d 1 \
    >"${output_dir}/nvidia_dmon.log" 2>&1 &
  GPU_SAMPLER_PID=$!
  "${PYTHON}" "${BENCHMARK_DIR}/scripts/benchmark_agent_session_dp.py" \
    --base-url "http://127.0.0.1:${PORT}" \
    --model "${MODEL_NAME}" \
    --policy-label "${policy}" \
    --sessions "${SESSIONS}" \
    --shared-prefix-tokens "${SHARED_PREFIX_TOKENS}" \
    --session-tokens "${SESSION_TOKENS}" \
    --followup-tokens "${FOLLOWUP_TOKENS}" \
    --trials "${TRIALS}" \
    --output "${output_dir}/result.json"

  curl --silent --fail "http://127.0.0.1:${PORT}/metrics" \
    >"${output_dir}/metrics.prom" || true
  stop_processes
done

echo "Profiling complete: ${OUTPUT_ROOT}"
