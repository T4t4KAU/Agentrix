#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/dp_reload_comparison}"
LMCACHE_PYTHON="${LMCACHE_PYTHON:-${REPO_ROOT}/vllm/.venv/bin/python}"
LMCACHE_SOURCE="${LMCACHE_SOURCE:-${REPO_ROOT}/LMCache}"
LMCACHE_HOST="${LMCACHE_HOST:-127.0.0.1}"
LMCACHE_PORT="${LMCACHE_PORT:-5555}"
LMCACHE_L1_GB="${LMCACHE_L1_GB:-16}"
LMCACHE_L1_INIT_GB="${LMCACHE_L1_INIT_GB:-4}"
LMCACHE_MAX_WORKERS="${LMCACHE_MAX_WORKERS:-8}"
LMCACHE_STARTUP_TIMEOUT="${LMCACHE_STARTUP_TIMEOUT:-60}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
KV_BLOCKS="${KV_BLOCKS:-608}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_OPTIMIZED="${RUN_OPTIMIZED:-1}"
LMCACHE_PID=""

connector_config="$(printf '%s' \
  '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both",' \
  '"kv_load_failure_policy":"recompute",' \
  '"kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector",' \
  '"kv_connector_extra_config":{"lmcache.mp.host":"tcp://' \
  "${LMCACHE_HOST}" \
  '","lmcache.mp.port":' \
  "${LMCACHE_PORT}" \
  '}}')"

stop_lmcache() {
  if [[ -n "${LMCACHE_PID}" ]] && kill -0 "${LMCACHE_PID}" 2>/dev/null; then
    kill -TERM "${LMCACHE_PID}" 2>/dev/null || true
    wait "${LMCACHE_PID}" 2>/dev/null || true
  fi
  LMCACHE_PID=""
}

start_lmcache() {
  local variant="$1"
  local log_dir="${BENCHMARK_DIR}/${OUTPUT_ROOT}/${variant}"
  local log_path="${log_dir}/lmcache_server.log"
  mkdir -p "${log_dir}"
  PYTHONPATH="${LMCACHE_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${LMCACHE_PYTHON}" -m lmcache.v1.multiprocess.server \
    --host "${LMCACHE_HOST}" \
    --port "${LMCACHE_PORT}" \
    --chunk-size 256 \
    --max-workers "${LMCACHE_MAX_WORKERS}" \
    --l1-size-gb "${LMCACHE_L1_GB}" \
    --l1-init-size-gb "${LMCACHE_L1_INIT_GB}" \
    --eviction-policy LRU \
    --disable-observability >"${log_path}" 2>&1 &
  LMCACHE_PID="$!"

  local deadline=$((SECONDS + LMCACHE_STARTUP_TIMEOUT))
  until grep -q "cache server is running" "${log_path}"; do
    if ! kill -0 "${LMCACHE_PID}" 2>/dev/null; then
      tail -n 80 "${log_path}" >&2
      return 1
    fi
    if ((SECONDS >= deadline)); then
      echo "Timed out waiting for LMCache server." >&2
      tail -n 80 "${log_path}" >&2
      return 1
    fi
    sleep 1
  done
}

run_variant() {
  local variant="$1"
  local enabled="$2"
  start_lmcache "${variant}"
  VLLM_FORK_ATTN_DP_RELOAD_REBALANCE="${enabled}" \
  VLLM_FORK_ATTN_DP_PREFIX_ROUTING=1 \
  VLLM_FORK_ATTN_DP_PREFIX_LOAD_SLACK="${DP_PREFIX_LOAD_SLACK:-28}" \
  VLLM_FORK_ATTN_DP_ARRIVAL_WAVE_MS="${DP_ARRIVAL_WAVE_MS:-50}" \
  VLLM_FORK_ATTN_DP_RELOAD_MIN_EXTERNAL_TOKENS="${DP_RELOAD_MIN_EXTERNAL_TOKENS:-256}" \
  VLLM_FORK_ATTN_DP_RELOAD_MAX_KV_USAGE="${DP_RELOAD_MAX_KV_USAGE:-1.0}" \
  MODEL_PATH="${MODEL_PATH}" \
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8b-dp-reload}" \
  BACKENDS=FORK_ATTN \
  DP_REPLICAS=2 \
  DP_DEPLOYMENT=internal \
  GPU_IDS="${GPU_IDS:-0,1}" \
  DATASET="${DATASET:-swebench}" \
  SAMPLE_INDEX="${SAMPLE_INDEX:-0}" \
  CASE_COUNT="${CASE_COUNT:-2}" \
  SAMPLE_COUNT="${SAMPLE_COUNT:-2}" \
  PREFIX_TOKENS="${PREFIX_TOKENS:-8192}" \
  BRANCHES="${BRANCHES:-9}" \
  BRANCH_GROUP_SIZE="${BRANCH_GROUP_SIZE:-1}" \
  CONCURRENCY="${CONCURRENCY:-18}" \
  SUFFIX_MEAN="${SUFFIX_MEAN:-32}" \
  OUTPUT_TOKENS="${OUTPUT_TOKENS:-768}" \
  MINORITY_HEADSTART_MS="${MINORITY_HEADSTART_MS:-3000}" \
  COMMON_ANALYSIS_TOKENS="${COMMON_ANALYSIS_TOKENS:-128}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-9600}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-24}" \
  STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-420}" \
  OUTPUT_DIR="${OUTPUT_ROOT}/${variant}" \
  DP_ROUTING=prefix_skewed \
  KV_TRANSFER_CONFIG="${connector_config}" \
  VLLM_SERVER_EXTRA_ARGS="--num-gpu-blocks-override ${KV_BLOCKS} --scheduling-policy priority --no-async-scheduling --no-scheduler-reserve-full-isl ${VLLM_SERVER_EXTRA_ARGS:-}" \
    "${BENCHMARK_DIR}/scripts/run_vllm_benchmark.sh"
  stop_lmcache
}

trap stop_lmcache EXIT INT TERM
if [[ "${RUN_BASELINE}" == "1" ]]; then
  run_variant baseline 0
fi
if [[ "${RUN_OPTIMIZED}" == "1" ]]; then
  run_variant optimized 1
fi
if [[ -f "${BENCHMARK_DIR}/${OUTPUT_ROOT}/baseline/fork_attn/benchmark_results.csv" && \
      -f "${BENCHMARK_DIR}/${OUTPUT_ROOT}/optimized/fork_attn/benchmark_results.csv" ]]; then
  "${BENCHMARK_PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}" \
    "${BENCHMARK_DIR}/src/dp_reload_report.py" \
    "${BENCHMARK_DIR}/${OUTPUT_ROOT}"
  echo "Comparison: ${BENCHMARK_DIR}/${OUTPUT_ROOT}/comparison.md"
fi
