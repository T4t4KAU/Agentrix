#!/usr/bin/env bash
set -Eeuo pipefail

# Run a ForkAttention benchmark under Nsight Systems. The benchmark workload
# options intentionally mirror run_vllm_benchmark.sh so the resulting trace and
# CSV/Markdown outputs are comparable to regular benchmark runs.

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
VLLM_BIN="${VLLM_BIN:-${REPO_ROOT}/vllm/.venv/bin/vllm}"
BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}"
NSYS_BIN="${NSYS_BIN:-nsys}"
export PATH="$(dirname "${VLLM_BIN}"):${PATH}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b-local}"
DTYPE="${DTYPE:-float16}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9000}"
DATASET="${DATASET:-swebench}"
DATA_PATH="${DATA_PATH:-}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
SAMPLE_COUNT="${SAMPLE_COUNT:-1}"
FULL_DATASET="${FULL_DATASET:-0}"
PREFIX_TOKENS="${PREFIX_TOKENS:-2048}"
BRANCHES="${BRANCHES:-2}"
CASE_COUNT="${CASE_COUNT:-1}"
BRANCH_GROUP_SIZE="${BRANCH_GROUP_SIZE:-1}"
CONCURRENCY="${CONCURRENCY:-$((BRANCHES * CASE_COUNT))}"
SUFFIX_DISTRIBUTION="${SUFFIX_DISTRIBUTION:-lognormal}"
SUFFIX_MEAN="${SUFFIX_MEAN:-128}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
COMMON_ANALYSIS_TOKENS="${COMMON_ANALYSIS_TOKENS:-128}"
ARRIVAL_INTERVAL_MS="${ARRIVAL_INTERVAL_MS:-0}"
SEED="${SEED:-2026}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
OUTPUT_DIR="${OUTPUT_DIR:-results/fork_attention_nsys_${DATASET}_c${CASE_COUNT}_p${PREFIX_TOKENS}_b${BRANCHES}_g${BRANCH_GROUP_SIZE}_o${OUTPUT_TOKENS}}"
KEEP_SERVER="${KEEP_SERVER:-0}"
VLLM_SERVER_EXTRA_ARGS="${VLLM_SERVER_EXTRA_ARGS:-}"
BENCHMARK_EXTRA_ARGS="${BENCHMARK_EXTRA_ARGS:-}"
NSYS_OUTPUT_NAME="${NSYS_OUTPUT_NAME:-fork_attention}"
NSYS_EXTRA_ARGS="${NSYS_EXTRA_ARGS:-}"

BASE_URL="http://${HOST}:${PORT}"
BACKEND_OUTPUT_DIR="${OUTPUT_DIR}/fork_attn"
BACKEND_LOG_DIR="${BENCHMARK_DIR}/${BACKEND_OUTPUT_DIR}"
SERVER_LOG="${BACKEND_LOG_DIR}/vllm_server_nsys.log"
NSYS_OUTPUT="${BACKEND_LOG_DIR}/${NSYS_OUTPUT_NAME}"
SERVER_PID=""

cd "${BENCHMARK_DIR}"
mkdir -p "${BACKEND_LOG_DIR}"

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

if ! command -v "${NSYS_BIN}" >/dev/null 2>&1; then
  echo "Nsight Systems executable not found: ${NSYS_BIN}" >&2
  echo "Set NSYS_BIN=/path/to/nsys or add nsys to PATH." >&2
  exit 1
fi

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    if [[ "${KEEP_SERVER}" == "1" ]]; then
      echo "vLLM server remains running under nsys (PID ${SERVER_PID})."
    else
      echo "Stopping vLLM server under nsys (PID ${SERVER_PID})..."
      kill -TERM "${SERVER_PID}" 2>/dev/null || true
      wait "${SERVER_PID}" 2>/dev/null || true
    fi
  fi
  SERVER_PID=""
}

trap stop_server EXIT INT TERM

if curl --silent --fail --max-time 2 "${BASE_URL}/health" >/dev/null 2>&1; then
  echo "Port ${PORT} already has a healthy server; refusing to replace it." >&2
  exit 1
fi

vllm_args=(
  serve "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --attention-backend FORK_ATTN
  --dtype "${DTYPE}"
  --generation-config vllm
  --enable-prefix-caching
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
)
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  vllm_args+=(--enforce-eager)
fi
if [[ -n "${VLLM_SERVER_EXTRA_ARGS}" ]]; then
  read -r -a extra_vllm_args <<<"${VLLM_SERVER_EXTRA_ARGS}"
  vllm_args+=("${extra_vllm_args[@]}")
fi

nsys_args=(
  profile
  --force-overwrite=true
  --sample=none
  --cpuctxsw=none
  --trace=cuda,nvtx,osrt,cublas,cudnn
  --cuda-graph-trace=node
  --output "${NSYS_OUTPUT}"
)
if [[ -n "${NSYS_EXTRA_ARGS}" ]]; then
  read -r -a extra_nsys_args <<<"${NSYS_EXTRA_ARGS}"
  nsys_args+=("${extra_nsys_args[@]}")
fi

echo "Starting ${MODEL_PATH} with FORK_ATTN under Nsight Systems..."
"${NSYS_BIN}" "${nsys_args[@]}" "${VLLM_BIN}" "${vllm_args[@]}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl --silent --fail --max-time 2 "${BASE_URL}/health" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "vLLM exited during startup. Last log lines:" >&2
    tail -n 120 "${SERVER_LOG}" >&2
    exit 1
  fi
  if ((SECONDS >= deadline)); then
    echo "Timed out waiting for vLLM after ${STARTUP_TIMEOUT}s." >&2
    tail -n 120 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 2
done

echo "vLLM is ready; available models:"
curl --silent --fail "${BASE_URL}/v1/models"
echo

benchmark_args=(
  -m cli run-api
  --dataset "${DATASET}"
  --sample-index "${SAMPLE_INDEX}"
  --sample-count "${SAMPLE_COUNT}"
  --api-mode chat
  --base-url "${BASE_URL}/v1"
  --model "${SERVED_MODEL_NAME}"
  --prefix-tokens "${PREFIX_TOKENS}"
  --branches "${BRANCHES}"
  --case-count "${CASE_COUNT}"
  --branch-group-size "${BRANCH_GROUP_SIZE}"
  --suffix-distribution "${SUFFIX_DISTRIBUTION}"
  --suffix-mean "${SUFFIX_MEAN}"
  --output-tokens "${OUTPUT_TOKENS}"
  --common-analysis-tokens "${COMMON_ANALYSIS_TOKENS}"
  --concurrency "${CONCURRENCY}"
  --arrival-interval-ms "${ARRIVAL_INTERVAL_MS}"
  --seed "${SEED}"
  --output-dir "${BACKEND_OUTPUT_DIR}"
)
if [[ -n "${DATA_PATH}" ]]; then
  benchmark_args+=(--data-path "${DATA_PATH}")
fi
if [[ "${FULL_DATASET}" == "1" ]]; then
  benchmark_args+=(--full-dataset)
fi
if [[ -n "${BENCHMARK_EXTRA_ARGS}" ]]; then
  read -r -a extra_benchmark_args <<<"${BENCHMARK_EXTRA_ARGS}"
  benchmark_args+=("${extra_benchmark_args[@]}")
fi

echo "Running Agentrix benchmark for FORK_ATTN under Nsight Systems..."
OPENAI_API_KEY="vllm-local" "${BENCHMARK_PYTHON}" "${benchmark_args[@]}"

stop_server

echo "Nsight Systems report base: ${NSYS_OUTPUT}"
echo "Benchmark complete: ${BACKEND_LOG_DIR}"
