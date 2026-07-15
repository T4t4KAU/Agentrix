#!/usr/bin/env bash
set -Eeuo pipefail

# Run the Agentrix OpenAI-compatible API benchmark against SGLang with LMCache MP mode.
BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
SGLANG_ROOT="${SGLANG_ROOT:-${REPO_ROOT}/sglang}"
SGLANG_VENV="${SGLANG_VENV:-${SGLANG_ROOT}/.venv}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${SGLANG_VENV}/bin/python}"
LMCACHE_BIN="${LMCACHE_BIN:-${SGLANG_VENV}/bin/lmcache}"
OUTPUT_DIR="${OUTPUT_DIR:-results/sglang_lmcache_smoke}"
OUTPUT_ROOT="${BENCHMARK_DIR}/${OUTPUT_DIR}"
LMCACHE_HOST="${LMCACHE_HOST:-127.0.0.1}"
LMCACHE_PORT="${LMCACHE_PORT:-5556}"
LMCACHE_L1_SIZE_GB="${LMCACHE_L1_SIZE_GB:-1}"
LMCACHE_EVICTION_POLICY="${LMCACHE_EVICTION_POLICY:-LRU}"
LMCACHE_STARTUP_TIMEOUT="${LMCACHE_STARTUP_TIMEOUT:-90}"
LMCACHE_CONFIG_FILE="${LMCACHE_CONFIG_FILE:-${OUTPUT_ROOT}/lmcache_mp.yaml}"
LMCACHE_LOG_FILE="${LMCACHE_LOG_FILE:-${OUTPUT_ROOT}/lmcache_server.log}"
LMCACHE_DISABLE_OBSERVABILITY="${LMCACHE_DISABLE_OBSERVABILITY:-1}"
LMCACHE_DISABLE_BANNER="${LMCACHE_DISABLE_BANNER:-1}"
LMCACHE_LD_PRELOAD="${LMCACHE_LD_PRELOAD:-}"
SGLANG_LMCACHE_ATTENTION_BACKEND="${SGLANG_LMCACHE_ATTENTION_BACKEND:-triton}"
LMCACHE_PID=""

mkdir -p "${OUTPUT_ROOT}"

if [[ ! -x "${LMCACHE_BIN}" ]]; then
  if command -v lmcache >/dev/null 2>&1; then
    LMCACHE_BIN="$(command -v lmcache)"
  else
    echo "LMCache executable does not exist: ${LMCACHE_BIN}" >&2
    echo "Install LMCache into the SGLang environment first." >&2
    exit 1
  fi
fi

if [[ -z "${LMCACHE_LD_PRELOAD}" && -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]]; then
  LMCACHE_LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
fi

if [[ -n "${LMCACHE_LD_PRELOAD}" ]]; then
  export LD_PRELOAD="${LMCACHE_LD_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
export LMCACHE_DISABLE_BANNER

cat >"${LMCACHE_CONFIG_FILE}" <<EOF
mp_host: ${LMCACHE_HOST}
mp_port: ${LMCACHE_PORT}
EOF

stop_lmcache() {
  if [[ -n "${LMCACHE_PID}" ]] && kill -0 "${LMCACHE_PID}" 2>/dev/null; then
    kill -TERM "${LMCACHE_PID}" 2>/dev/null || true
    wait "${LMCACHE_PID}" 2>/dev/null || true
  fi
  LMCACHE_PID=""
}

start_lmcache() {
  local server_args=(
    server
    --host "${LMCACHE_HOST}"
    --port "${LMCACHE_PORT}"
    --l1-size-gb "${LMCACHE_L1_SIZE_GB}"
    --eviction-policy "${LMCACHE_EVICTION_POLICY}"
  )
  if [[ "${LMCACHE_DISABLE_OBSERVABILITY}" == "1" ]]; then
    server_args+=(--disable-observability)
  fi
  if [[ -n "${LMCACHE_CHUNK_SIZE:-}" ]]; then
    server_args+=(--chunk-size "${LMCACHE_CHUNK_SIZE}")
  fi
  if [[ -n "${LMCACHE_MAX_WORKERS:-}" ]]; then
    server_args+=(--max-workers "${LMCACHE_MAX_WORKERS}")
  fi
  if [[ -n "${LMCACHE_L1_INIT_SIZE_GB:-}" ]]; then
    server_args+=(--l1-init-size-gb "${LMCACHE_L1_INIT_SIZE_GB}")
  fi

  "${LMCACHE_BIN}" "${server_args[@]}" >"${LMCACHE_LOG_FILE}" 2>&1 &
  LMCACHE_PID="$!"

  local deadline=$((SECONDS + LMCACHE_STARTUP_TIMEOUT))
  until grep -q "cache server is running" "${LMCACHE_LOG_FILE}"; do
    if ! kill -0 "${LMCACHE_PID}" 2>/dev/null; then
      tail -n 100 "${LMCACHE_LOG_FILE}" >&2
      return 1
    fi
    if ((SECONDS >= deadline)); then
      echo "Timed out waiting for LMCache server." >&2
      tail -n 100 "${LMCACHE_LOG_FILE}" >&2
      return 1
    fi
    sleep 1
  done
}

trap stop_lmcache EXIT INT TERM

start_lmcache

lmcache_sglang_args=(
  --attention-backend "${SGLANG_LMCACHE_ATTENTION_BACKEND}"
  --enable-lmcache
  --lmcache-config-file "${LMCACHE_CONFIG_FILE}"
  --disable-cuda-graph
)
if [[ -n "${SGLANG_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_sglang_args <<<"${SGLANG_EXTRA_ARGS}"
  lmcache_sglang_args+=("${extra_sglang_args[@]}")
fi

SGLANG_PYTHON="${SGLANG_PYTHON}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
SGLANG_EXTRA_ARGS="${lmcache_sglang_args[*]}" \
  "${BENCHMARK_DIR}/scripts/run_sglang_benchmark.sh"

if grep -Eq "Traceback|ModuleNotFoundError|REGISTER_KV_CACHE.*error|EngineCore encountered a fatal error" \
  "${LMCACHE_LOG_FILE}" "${OUTPUT_ROOT}/sglang/sglang_server.log"; then
  echo "SGLang LMCache benchmark finished with errors in logs." >&2
  tail -n 100 "${LMCACHE_LOG_FILE}" >&2
  tail -n 100 "${OUTPUT_ROOT}/sglang/sglang_server.log" >&2
  exit 1
fi

if ! grep -q "Registered KV cache" "${LMCACHE_LOG_FILE}"; then
  echo "SGLang LMCache benchmark did not register KV cache with LMCache." >&2
  tail -n 100 "${LMCACHE_LOG_FILE}" >&2
  exit 1
fi

echo "SGLang LMCache benchmark complete: ${OUTPUT_ROOT}/sglang"
echo "LMCache log: ${LMCACHE_LOG_FILE}"
