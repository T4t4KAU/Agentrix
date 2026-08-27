#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGENTRIX_ROOT="${AGENTRIX_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$(dirname "${AGENTRIX_ROOT}")}"
VLLM_ROOT="${VLLM_ROOT:-${AGENTRIX_ROOT}/vllm}"
MODEL="${MODEL:-${WORK_ROOT}/models/Qwen2.5-VL-7B-Instruct}"
PYTHON="${PYTHON:-${VLLM_ROOT}/.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-${WORK_ROOT}/epd_experiments/runs}"
MEDIA_ROOT="${MEDIA_ROOT:-${WORK_ROOT}/datasets}"
COMPAT_PATH="${COMPAT_PATH:-${WORK_ROOT}/vllm_runs/qwen2_5_vl_7b}"

ENCODER_PORT="${ENCODER_PORT:-19534}"
PD_PORT="${PD_PORT:-19535}"
PROXY_PORT="${PROXY_PORT:-10001}"
GPU_E="${GPU_E:-0}"
GPU_PD="${GPU_PD:-1}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_ROOT}/${RUN_ID}"
EC_SHARED_STORAGE_PATH="/dev/shm/agentrix_epd_ec_${RUN_ID}"

mkdir -p "${RUN_DIR}" "${EC_SHARED_STORAGE_PATH}"

if curl -fsS "http://127.0.0.1:${ENCODER_PORT}/health" >/dev/null 2>&1 || \
   curl -fsS "http://127.0.0.1:${PD_PORT}/health" >/dev/null 2>&1 || \
   curl -fsS "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1; then
  echo "An EPD endpoint is already running; stop it before starting another run." >&2
  exit 1
fi

COMMON_ENV=(
  "PYTHONPATH=${COMPAT_PATH}:${VLLM_ROOT}"
  "VLLM_LOGGING_LEVEL=INFO"
)

cd "${VLLM_ROOT}"

setsid env "${COMMON_ENV[@]}" CUDA_VISIBLE_DEVICES="${GPU_E}" \
  "${PYTHON}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
  --served-model-name Qwen2.5-VL-7B-Instruct \
  --host 127.0.0.1 \
  --port "${ENCODER_PORT}" \
  --dtype auto \
  --gpu-memory-utilization 0.05 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --mm-encoder-only \
  --max-num-batched-tokens 114688 \
  --max-num-seqs 128 \
  --allowed-local-media-path "${MEDIA_ROOT}" \
  --kernel-config '{"enable_cutedsl_warmup":false}' \
  --ec-transfer-config "{\
    \"ec_connector\": \"ECExampleConnector\",\
    \"ec_role\": \"ec_producer\",\
    \"ec_connector_extra_config\": {\
      \"shared_storage_path\": \"${EC_SHARED_STORAGE_PATH}\"\
    }\
  }" \
  >"${RUN_DIR}/encoder.log" 2>&1 &
ENCODER_PID=$!
echo "${ENCODER_PID}" >"${RUN_DIR}/encoder.pid"

setsid env "${COMMON_ENV[@]}" CUDA_VISIBLE_DEVICES="${GPU_PD}" \
  "${PYTHON}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
  --served-model-name Qwen2.5-VL-7B-Instruct \
  --host 127.0.0.1 \
  --port "${PD_PORT}" \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --max-num-seqs 128 \
  --allowed-local-media-path "${MEDIA_ROOT}" \
  --kernel-config '{"enable_cutedsl_warmup":false}' \
  --ec-transfer-config "{\
    \"ec_connector\": \"ECExampleConnector\",\
    \"ec_role\": \"ec_consumer\",\
    \"ec_connector_extra_config\": {\
      \"shared_storage_path\": \"${EC_SHARED_STORAGE_PATH}\"\
    }\
  }" \
  >"${RUN_DIR}/pd.log" 2>&1 &
PD_PID=$!
echo "${PD_PID}" >"${RUN_DIR}/pd.pid"

wait_for_health() {
  local port="$1"
  local name="$2"
  for _ in $(seq 1 600); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for ${name} on port ${port}." >&2
  return 1
}

wait_for_health "${ENCODER_PORT}" encoder
wait_for_health "${PD_PORT}" pd

setsid env "PYTHONPATH=${VLLM_ROOT}" \
  "${PYTHON}" \
  "${VLLM_ROOT}/examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py" \
  --host 0.0.0.0 \
  --port "${PROXY_PORT}" \
  --encode-servers-urls "http://127.0.0.1:${ENCODER_PORT}" \
  --prefill-servers-urls disable \
  --decode-servers-urls "http://127.0.0.1:${PD_PORT}" \
  >"${RUN_DIR}/proxy.log" 2>&1 &
PROXY_PID=$!
echo "${PROXY_PID}" >"${RUN_DIR}/proxy.pid"
wait_for_health "${PROXY_PORT}" proxy

ln -sfn "${RUN_DIR}" "${RUN_ROOT}/current"
cat >"${RUN_DIR}/run.env" <<EOF
RUN_ID=${RUN_ID}
RUN_DIR=${RUN_DIR}
MODEL=${MODEL}
GPU_E=${GPU_E}
GPU_PD=${GPU_PD}
ENCODER_PORT=${ENCODER_PORT}
PD_PORT=${PD_PORT}
PROXY_PORT=${PROXY_PORT}
EC_SHARED_STORAGE_PATH=${EC_SHARED_STORAGE_PATH}
VLLM_COMMIT=$(git -C "${VLLM_ROOT}" rev-parse HEAD)
EOF

echo "1E+1PD ready at http://127.0.0.1:${PROXY_PORT}/v1"
echo "run directory: ${RUN_DIR}"
