#!/usr/bin/env bash
set -Eeuo pipefail

# Environment variables can override every expensive benchmark parameter.
BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
VLLM_BIN="${VLLM_BIN:-${REPO_ROOT}/vllm/.venv/bin/vllm}"
BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}"
export PATH="$(dirname "${VLLM_BIN}"):${PATH}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b-local}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-PAT_ATTN}"
DTYPE="${DTYPE:-float16}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9000}"
DATASET="${DATASET:-swebench}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
PREFIX_TOKENS="${PREFIX_TOKENS:-2048}"
BRANCHES="${BRANCHES:-2}"
SUFFIX_DISTRIBUTION="${SUFFIX_DISTRIBUTION:-lognormal}"
SUFFIX_MEAN="${SUFFIX_MEAN:-128}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
COMMON_ANALYSIS_TOKENS="${COMMON_ANALYSIS_TOKENS:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
OUTPUT_DIR="${OUTPUT_DIR:-results/${ATTENTION_BACKEND,,}_p${PREFIX_TOKENS}_b${BRANCHES}_o${OUTPUT_TOKENS}}"
KEEP_SERVER="${KEEP_SERVER:-0}"

LOG_DIR="${BENCHMARK_DIR}/${OUTPUT_DIR}"
SERVER_LOG="${LOG_DIR}/vllm_server.log"
BASE_URL="http://${HOST}:${PORT}"
SERVER_PID=""

cd "${BENCHMARK_DIR}"
mkdir -p "${LOG_DIR}"

if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "vLLM executable does not exist: ${VLLM_BIN}" >&2
  echo "Build the vllm submodule first; see the repository README." >&2
  exit 1
fi

if [[ ! -x "${BENCHMARK_PYTHON}" ]]; then
  echo "Benchmark Python does not exist: ${BENCHMARK_PYTHON}" >&2
  echo "Install the benchmark environment first; see the repository README." >&2
  exit 1
fi

if curl --silent --fail --max-time 2 "${BASE_URL}/health" >/dev/null 2>&1; then
  echo "Port ${PORT} already has a healthy server; refusing to replace it." >&2
  exit 1
fi

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    if [[ "${KEEP_SERVER}" == "1" ]]; then
      echo "vLLM server remains running (PID ${SERVER_PID})."
    else
      echo "Stopping vLLM server (PID ${SERVER_PID})..."
      kill -TERM "${SERVER_PID}" 2>/dev/null || true
      wait "${SERVER_PID}" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT INT TERM

vllm_args=(
  serve "${MODEL_PATH}"
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --dtype "${DTYPE}" \
  --generation-config vllm \
  --enable-prefix-caching \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}"
)
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  vllm_args+=(--enforce-eager)
fi

echo "Starting ${MODEL_PATH} with ${ATTENTION_BACKEND} on ${BASE_URL}..."
"${VLLM_BIN}" "${vllm_args[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl --silent --fail --max-time 2 "${BASE_URL}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "vLLM exited during startup. Last log lines:" >&2
    tail -n 80 "${SERVER_LOG}" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for vLLM after ${STARTUP_TIMEOUT}s." >&2
    tail -n 80 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 2
done

echo "vLLM is ready; available models:"
curl --silent --fail "${BASE_URL}/v1/models"
echo

echo "Running Agentrix benchmark..."
OPENAI_API_KEY="vllm-local" "${BENCHMARK_PYTHON}" -m cli run-api \
  --dataset "${DATASET}" \
  --sample-index "${SAMPLE_INDEX}" \
  --api-mode chat \
  --base-url "${BASE_URL}/v1" \
  --model "${SERVED_MODEL_NAME}" \
  --prefix-tokens "${PREFIX_TOKENS}" \
  --branches "${BRANCHES}" \
  --suffix-distribution "${SUFFIX_DISTRIBUTION}" \
  --suffix-mean "${SUFFIX_MEAN}" \
  --output-tokens "${OUTPUT_TOKENS}" \
  --common-analysis-tokens "${COMMON_ANALYSIS_TOKENS}" \
  --concurrency "${BRANCHES}" \
  --output-dir "${OUTPUT_DIR}"

echo "Benchmark complete: ${BENCHMARK_DIR}/${OUTPUT_DIR}"
