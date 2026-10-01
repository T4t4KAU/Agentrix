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
POLICIES="${POLICIES:-native consistent_hash cache_aware}"
BACKEND_PORT="${BACKEND_PORT:-$((PORT + 1))}"
ROUTER_METRICS_PORT="${ROUTER_METRICS_PORT:-$((PORT + 2))}"
ROUTER_PYTHON="${ROUTER_PYTHON:-${REPO_ROOT}/.router-venv/bin/python}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to an experiment-server results directory}"
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
ROUTER_PID=""

stop_processes() {
  if [[ -n "${ROUTER_PID}" ]] && kill -0 "${ROUTER_PID}" 2>/dev/null; then
    kill -TERM "${ROUTER_PID}" 2>/dev/null || true
    wait "${ROUTER_PID}" 2>/dev/null || true
  fi
  ROUTER_PID=""
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
  local url="$2"
  local pid="$3"
  for _ in $(seq 1 "${STARTUP_TIMEOUT}"); do
    if curl --silent --fail --max-time 2 \
      "${url}/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -n 120 "${log}" >&2
      return 1
    fi
    sleep 1
  done
  tail -n 120 "${log}" >&2
  return 1
}

for policy in ${POLICIES}; do
  case "${policy}" in
    native|consistent_hash|cache_aware|round_robin) ;;
    *) echo "Unsupported routing policy: ${policy}" >&2; exit 2 ;;
  esac
done
"${ROUTER_PYTHON}" "${BENCHMARK_DIR}/scripts/serve_dp_router.py" --check
"${PYTHON}" - "${PORT}" "${BACKEND_PORT}" "${ROUTER_METRICS_PORT}" <<'PY'
import socket
import sys

ports = [int(value) for value in sys.argv[1:]]
if len(set(ports)) != len(ports):
    raise SystemExit("Router, backend and metrics ports must differ")
for port in ports:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
PY
trap stop_processes EXIT INT TERM
mkdir -p "${OUTPUT_ROOT}"
for policy in ${POLICIES}; do
  if [[ -e "${OUTPUT_ROOT}/${policy}" ]]; then
    echo "Results already exist for ${policy}; choose a new OUTPUT_ROOT." >&2
    exit 2
  fi
done

for policy in ${POLICIES}; do
  output_dir="${OUTPUT_ROOT}/${policy}"
  mkdir "${output_dir}"
  server_log="${output_dir}/vllm_server.log"
  echo "Starting ${policy} on GPUs ${GPU_IDS}"
  setsid env \
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    PYTHONHASHSEED=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_SERVER_DEV_MODE=1 \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --host 127.0.0.1 \
      --port "${BACKEND_PORT}" \
      --served-model-name "${MODEL_NAME}" \
      --attention-backend FLASH_ATTN \
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
  backend_url="http://127.0.0.1:${BACKEND_PORT}"
  wait_for_server "${server_log}" "${backend_url}" "${SERVER_PID}"
  base_url="${backend_url}"
  if [[ "${policy}" != "native" ]]; then
    base_url="http://127.0.0.1:${PORT}"
    "${ROUTER_PYTHON}" "${BENCHMARK_DIR}/scripts/serve_dp_router.py" \
      --worker-urls "${backend_url}" --policy "${policy}" \
      --intra-node-data-parallel-size 2 --host 127.0.0.1 --port "${PORT}" \
      --prometheus-port "${ROUTER_METRICS_PORT}" \
      >"${output_dir}/router.log" 2>&1 &
    ROUTER_PID=$!
    wait_for_server "${output_dir}/router.log" "${base_url}" "${ROUTER_PID}"
  fi

  nvidia-smi dmon -i "${GPU_IDS}" -s pucvmet -d 1 \
    >"${output_dir}/nvidia_dmon.log" 2>&1 &
  GPU_SAMPLER_PID=$!
  "${PYTHON}" "${BENCHMARK_DIR}/scripts/benchmark_agent_session_dp.py" \
    --base-url "${base_url}" \
    --control-url "${backend_url}" \
    --model "${MODEL_NAME}" \
    --policy-label "${policy}" \
    --sessions "${SESSIONS}" \
    --shared-prefix-tokens "${SHARED_PREFIX_TOKENS}" \
    --session-tokens "${SESSION_TOKENS}" \
    --followup-tokens "${FOLLOWUP_TOKENS}" \
    --trials "${TRIALS}" \
    --output "${output_dir}/result.json"

  curl --silent --fail "${backend_url}/metrics" \
    >"${output_dir}/metrics.prom" || true
  stop_processes
done

echo "Profiling complete: ${OUTPUT_ROOT}"
