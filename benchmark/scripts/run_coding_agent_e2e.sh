#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
PYTHON="${BENCHMARK_PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-${REPO_ROOT}/vllm/.venv/bin/vllm}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
CASES_PATH="${CASES_PATH:?CASES_PATH is required}"
CASE_ID="${CASE_ID:?CASE_ID is required}"
SOURCE_REPO="${SOURCE_REPO:?SOURCE_REPO is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
MODEL_NAME="${SERVED_MODEL_NAME:-coding-agent-e2e}"
BACKEND="${ATTENTION_BACKEND:-FORK_ATTN}"
PREFIX_ROUTING="${PREFIX_ROUTING:-1}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
DP_REPLICAS="${DP_REPLICAS:-4}"
PORT="${PORT:-9000}"
SERVER_PID=""

export PYTHONPATH="${BENCHMARK_DIR}/src:${REPO_ROOT}/vllm${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_DIR}"

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap stop_server EXIT INT TERM

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_FORK_ATTN_ENABLE_FOREST=1 \
VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH=1 \
VLLM_FORK_ATTN_FANOUT_SCHEDULING_ENABLED="${PREFIX_ROUTING}" \
VLLM_FORK_ATTN_DP_PREFIX_ROUTING="${PREFIX_ROUTING}" \
VLLM_FORK_ATTN_DP_RELOAD_REBALANCE=0 \
VLLM_FORK_ATTN_DP_ARRIVAL_WAVE_MS=10 \
"${VLLM_BIN}" serve "${MODEL_PATH}" \
  --host 127.0.0.1 --port "${PORT}" --served-model-name "${MODEL_NAME}" \
  --attention-backend "${BACKEND}" --dtype float16 \
  --generation-config vllm --enable-prefix-caching --no-async-scheduling \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --data-parallel-size "${DP_REPLICAS}" --api-server-count 1 \
  --gpu-memory-utilization 0.70 --num-gpu-blocks-override 3852 \
  --max-model-len 40960 --max-num-batched-tokens 16384 \
  --max-num-seqs 64 >"${OUTPUT_DIR}/vllm_server.log" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + 600))
until curl --silent --fail --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -n 100 "${OUTPUT_DIR}/vllm_server.log" >&2
    exit 1
  fi
  ((SECONDS < deadline)) || exit 1
  sleep 2
done

"${PYTHON}" -m coding_agent_e2e_runner \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "${MODEL_NAME}" \
  --cases "${CASES_PATH}" --case-id "${CASE_ID}" \
  --task-root "${BENCHMARK_DIR}/coding_tasks" --repo "${SOURCE_REPO}" \
  --rounds "${ROUNDS:-3}" --trajectory-mode "${TRAJECTORY_MODE:-live}" \
  --branch-output-tokens "${BRANCH_OUTPUT_TOKENS:-128}" \
  --parent-output-tokens "${PARENT_OUTPUT_TOKENS:-512}" \
  --max-tool-steps "${MAX_TOOL_STEPS:-14}" \
  --output "${OUTPUT_DIR}/run.json"
