#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-results/lmcache_mooncake_smoke}"
OUTPUT_ROOT="${BENCHMARK_DIR}/${OUTPUT_DIR}"
VLLM_BIN="${VLLM_BIN:-${REPO_ROOT}/vllm/.venv/bin/vllm}"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-${REPO_ROOT}/vllm/.venv/bin/python}"
BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-${REPO_ROOT}/benchmark/.venv/bin/python}"
LMCACHE_SOURCE="${LMCACHE_SOURCE:-${REPO_ROOT}/LMCache}"
BIN_DIR="$(dirname -- "${RUNTIME_PYTHON}")"
MOONCAKE_MASTER_BIN="${MOONCAKE_MASTER_BIN:-${BIN_DIR}/mooncake_master}"
MOONCAKE_METADATA_BIN="${MOONCAKE_METADATA_BIN:-${BIN_DIR}/mooncake_http_metadata_server}"

HOST="${HOST:-127.0.0.1}"
MOONCAKE_MASTER_PORT="${MOONCAKE_MASTER_PORT:-50051}"
MOONCAKE_METADATA_PORT="${MOONCAKE_METADATA_PORT:-8005}"
VLLM_PORT="${PORT:-9000}"
MOONCAKE_MEMORY_SIZE_GB="${MOONCAKE_MEMORY_SIZE_GB:-4}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-420}"
LMCACHE_CONFIG_FILE="${OUTPUT_ROOT}/lmcache_mooncake.yaml"

MOONCAKE_MASTER_PID=""
MOONCAKE_METADATA_PID=""

stop_process() {
  local pid="$1"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  stop_process "${MOONCAKE_MASTER_PID}"
  stop_process "${MOONCAKE_METADATA_PID}"
}
trap cleanup EXIT INT TERM

require_executable() {
  local path="$1"
  if [[ ! -x "${path}" ]]; then
    echo "Required executable does not exist: ${path}" >&2
    exit 1
  fi
}

port_is_open() {
  "${RUNTIME_PYTHON}" - "$1" "$2" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=0.25):
        pass
except OSError:
    raise SystemExit(1)
PY
}

wait_for_port() {
  local name="$1"
  local port="$2"
  local pid="$3"
  local log_path="$4"
  local deadline=$((SECONDS + 60))
  until port_is_open "${HOST}" "${port}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${name} exited during startup." >&2
      tail -n 100 "${log_path}" >&2 || true
      exit 1
    fi
    if ((SECONDS >= deadline)); then
      echo "Timed out waiting for ${name} on ${HOST}:${port}." >&2
      tail -n 100 "${log_path}" >&2 || true
      exit 1
    fi
    sleep 1
  done
}

require_executable "${RUNTIME_PYTHON}"
require_executable "${BENCHMARK_PYTHON}"
require_executable "${VLLM_BIN}"
require_executable "${MOONCAKE_MASTER_BIN}"
require_executable "${MOONCAKE_METADATA_BIN}"

mkdir -p "${OUTPUT_ROOT}"
for port in \
  "${MOONCAKE_MASTER_PORT}" \
  "${MOONCAKE_METADATA_PORT}" \
  "${VLLM_PORT}"; do
  if port_is_open "${HOST}" "${port}"; then
    echo "Port ${HOST}:${port} is already in use; refusing to use a stale service." >&2
    exit 1
  fi
done

mooncake_memory_bytes="$("${RUNTIME_PYTHON}" - "${MOONCAKE_MEMORY_SIZE_GB}" <<'PY'
import sys

print(int(float(sys.argv[1]) * 1024**3))
PY
)"

cat >"${LMCACHE_CONFIG_FILE}" <<EOF
chunk_size: ${LMCACHE_CHUNK_SIZE:-256}
remote_url: "mooncakestore://${HOST}:${MOONCAKE_MASTER_PORT}/"
remote_serde: "naive"
local_cpu: false
max_local_cpu_size: 1
extra_config:
  save_chunk_meta: false
  local_hostname: "${HOST}"
  metadata_server: "http://${HOST}:${MOONCAKE_METADATA_PORT}/metadata"
  protocol: "tcp"
  device_name: ""
  master_server_address: "${HOST}:${MOONCAKE_MASTER_PORT}"
  global_segment_size: ${mooncake_memory_bytes}
  local_buffer_size: ${mooncake_memory_bytes}
  transfer_timeout: 5
EOF

connector_config="$("${RUNTIME_PYTHON}" - <<'PY'
import json

print(json.dumps({
    "kv_connector": "LMCacheConnectorV1",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
}))
PY
)"

echo "Starting Mooncake metadata server on ${HOST}:${MOONCAKE_METADATA_PORT}"
"${MOONCAKE_METADATA_BIN}" \
  --port "${MOONCAKE_METADATA_PORT}" \
  >"${OUTPUT_ROOT}/mooncake_metadata.log" 2>&1 &
MOONCAKE_METADATA_PID="$!"
wait_for_port \
  "Mooncake metadata server" \
  "${MOONCAKE_METADATA_PORT}" \
  "${MOONCAKE_METADATA_PID}" \
  "${OUTPUT_ROOT}/mooncake_metadata.log"

echo "Starting Mooncake master on ${HOST}:${MOONCAKE_MASTER_PORT}"
"${MOONCAKE_MASTER_BIN}" \
  --rpc_address "${HOST}" \
  --rpc_port "${MOONCAKE_MASTER_PORT}" \
  -v=1 \
  >"${OUTPUT_ROOT}/mooncake_master.log" 2>&1 &
MOONCAKE_MASTER_PID="$!"
wait_for_port \
  "Mooncake master" \
  "${MOONCAKE_MASTER_PORT}" \
  "${MOONCAKE_MASTER_PID}" \
  "${OUTPUT_ROOT}/mooncake_master.log"

LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE}" \
PYTHONHASHSEED=0 \
RUNTIME_PYTHONPATH="${LMCACHE_SOURCE}" \
KV_TRANSFER_CONFIG="${connector_config}" \
VLLM_BIN="${VLLM_BIN}" \
BENCHMARK_PYTHON="${BENCHMARK_PYTHON}" \
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/../models/Qwen3-VL-8B-Instruct}" \
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-8b-mooncake-smoke}" \
BACKENDS="${BACKENDS:-FLASH_ATTN}" \
GPU_IDS="${GPU_IDS:-0}" \
HOST="${HOST}" \
PORT="${VLLM_PORT}" \
DTYPE="${DTYPE:-bfloat16}" \
PREFIX_TOKENS="${PREFIX_TOKENS:-2048}" \
BRANCHES="${BRANCHES:-4}" \
CASE_COUNT="${CASE_COUNT:-1}" \
SAMPLE_COUNT="${SAMPLE_COUNT:-1}" \
CONCURRENCY="${CONCURRENCY:-1}" \
OUTPUT_TOKENS="${OUTPUT_TOKENS:-8}" \
COMMON_ANALYSIS_TOKENS="${COMMON_ANALYSIS_TOKENS:-8}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}" \
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}" \
ENFORCE_EAGER="${ENFORCE_EAGER:-1}" \
ENABLE_TELEMETRY="${ENABLE_TELEMETRY:-0}" \
KV_BYTES_PER_TOKEN="${KV_BYTES_PER_TOKEN:-0}" \
STARTUP_TIMEOUT="${STARTUP_TIMEOUT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
  "${BENCHMARK_DIR}/scripts/run_vllm_benchmark.sh"

backend_name="${BACKENDS:-FLASH_ATTN}"
backend_name="${backend_name,,}"
server_log="${OUTPUT_ROOT}/${backend_name}/vllm_server.log"
if ! grep -q "Mooncake store setup completed successfully" "${server_log}"; then
  echo "Mooncake smoke failed: LMCache did not initialize Mooncake Store." >&2
  tail -n 120 "${server_log}" >&2
  exit 1
fi
if grep -Eq "Traceback|EngineCore encountered a fatal error|Segmentation fault" \
  "${server_log}"; then
  echo "Mooncake smoke failed: a runtime error was found in the logs." >&2
  tail -n 120 "${server_log}" >&2
  exit 1
fi

cat >"${OUTPUT_ROOT}/smoke_summary.txt" <<EOF
status=passed
transport=tcp
lmcache_mode=inprocess
lmcache_local_cpu=false
mooncake_memory_size_gb=${MOONCAKE_MEMORY_SIZE_GB}
model=${MODEL_PATH:-${REPO_ROOT}/../models/Qwen3-VL-8B-Instruct}
lmcache_config=${LMCACHE_CONFIG_FILE}
mooncake_master_log=${OUTPUT_ROOT}/mooncake_master.log
vllm_log=${server_log}
EOF

echo "LMCache + Mooncake smoke passed."
echo "Summary: ${OUTPUT_ROOT}/smoke_summary.txt"
