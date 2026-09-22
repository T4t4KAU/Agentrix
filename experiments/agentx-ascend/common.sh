#!/usr/bin/env bash
# Source from the launchers; keep the Ascend and CUDA environments separate.
AGENTRIX_ROOT=${AGENTRIX_ROOT:-/data/Agentrix}
MODEL_NAME=${MODEL_NAME:-Qwen3.5-9B}
MODEL_PATH=${MODEL_PATH:-/data/models/$MODEL_NAME}
case "$MODEL_NAME" in
  Qwen3.5-9B) CONTEXT_LENGTH=${CONTEXT_LENGTH:-262144}; CONCURRENCY=${CONCURRENCY:-16} ;;
  Qwen3-8B) CONTEXT_LENGTH=${CONTEXT_LENGTH:-131072}; CONCURRENCY=${CONCURRENCY:-4} ;;
  *) echo "Unsupported model: $MODEL_NAME" >&2; exit 2 ;;
esac
ASCEND_ENV=${ASCEND_ENV:-$AGENTRIX_ROOT/activate-ascend.sh}
VLLM_BIN=${VLLM_BIN:-$AGENTRIX_ROOT/.venv/bin/vllm}
HARNESS_ROOT=${HARNESS_ROOT:-$AGENTRIX_ROOT/agentx/harness}
AIPERF_BIN=${AIPERF_BIN:-$AGENTRIX_ROOT/agentx/.venv/bin/aiperf}
HARNESS_COMMIT=56a0cf70f4c0359454ee4bd15a17770b541a3e3e
BACKEND_PORT=${BACKEND_PORT:-8001}
BENCHMARK_URL=${BENCHMARK_URL:-http://127.0.0.1:8000}
BENCHMARK_DURATION=${BENCHMARK_DURATION:-900}
RANDOM_SEED=${RANDOM_SEED:-20260707}

run_command() {
  if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    exec "$@"
  fi
}
