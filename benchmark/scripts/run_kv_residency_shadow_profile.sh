#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
VLLM_BIN="${VLLM_BIN:-${REPO_ROOT}/vllm/.venv/bin/vllm}"
PYTHON="${PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/../models/Qwen3-VL-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BENCHMARK_DIR}/results/kv_residency_shadow}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8136}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
SESSIONS="${SESSIONS:-8}"
TRIALS="${TRIALS:-5}"
MODES="${MODES:-0 1}"
export PATH="$(dirname "${VLLM_BIN}"):${PATH}"

SERVER_PID=""

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" || true
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
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -n 120 "${log}" >&2
      return 1
    fi
    sleep 1
  done
  tail -n 120 "${log}" >&2
  return 1
}

trap stop_server EXIT INT TERM
mkdir -p "${OUTPUT_ROOT}"

for shadow in ${MODES}; do
  label="disabled"
  if [[ "${shadow}" == "1" ]]; then
    label="enabled"
  fi
  output_dir="${OUTPUT_ROOT}/${label}"
  mkdir -p "${output_dir}"
  server_log="${output_dir}/server.log"

  setsid env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHONPATH="${REPO_ROOT}/vllm" \
    PYTHONHASHSEED=0 \
    VLLM_SERVER_DEV_MODE=1 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_AGENTRIX_KV_RESIDENCY_SHADOW="${shadow}" \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --served-model-name qwen3-vl \
      --attention-backend FORK_ATTN \
      --dtype bfloat16 \
      --generation-config vllm \
      --enable-prefix-caching \
      --no-async-scheduling \
      --gpu-memory-utilization 0.80 \
      --max-model-len 4096 \
      --max-num-batched-tokens 8192 \
      --max-num-seqs 16 \
      >"${server_log}" 2>&1 &
  SERVER_PID=$!
  wait_for_server "${server_log}"

  "${PYTHON}" "${BENCHMARK_DIR}/scripts/benchmark_agent_session_dp.py" \
    --base-url "http://127.0.0.1:${PORT}" \
    --model qwen3-vl \
    --policy-label "${label}-warmup" \
    --sessions 2 \
    --shared-prefix-tokens 256 \
    --session-tokens 128 \
    --followup-tokens 16 \
    --trials 1 \
    --output "${output_dir}/warmup.json"

  "${PYTHON}" "${BENCHMARK_DIR}/scripts/benchmark_agent_session_dp.py" \
    --base-url "http://127.0.0.1:${PORT}" \
    --model qwen3-vl \
    --policy-label "${label}" \
    --sessions "${SESSIONS}" \
    --shared-prefix-tokens 2048 \
    --session-tokens 1024 \
    --followup-tokens 64 \
    --trials "${TRIALS}" \
    --output "${output_dir}/result.json"

  stop_server
done

echo "Profiling complete: ${OUTPUT_ROOT}"
