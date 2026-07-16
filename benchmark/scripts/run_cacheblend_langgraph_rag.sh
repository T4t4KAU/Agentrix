#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
VLLM_BIN="${VLLM_BIN:-${REPO_ROOT}/vllm/.venv/bin/vllm}"
PYTHON="${BENCHMARK_PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/hwx/Documents/models/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-cacheblend-rag}"
TASK_FILE="${TASK_FILE:-${BENCHMARK_DIR}/configs/langgraph_cacheblend_rag_tasks.jsonl}"
RAG_ROOT="${RAG_ROOT:-${REPO_ROOT}/docs}"
LMCACHE_CONFIG="${LMCACHE_CONFIG:-${BENCHMARK_DIR}/configs/lmcache_cacheblend_rag.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BENCHMARK_DIR}/results/cacheblend_langgraph_rag}"
SCENARIOS="${SCENARIOS:-incident_diagnosis deployment_review cache_reuse_audit mixed_queue_control}"
CASES_PER_SCENARIO="${CASES_PER_SCENARIO:-3}"
CONCURRENCY="${CONCURRENCY:-16}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9000}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
DTYPE="${DTYPE:-bfloat16}"
REPEATS="${REPEATS:-3}"
ENABLE_CACHEBLEND="${ENABLE_CACHEBLEND:-0}"
SERVER_PID=""
SERVER_LOG=""

if [[ "${ENABLE_CACHEBLEND}" != "0" && "${ENABLE_CACHEBLEND}" != "1" ]]; then
  echo "ENABLE_CACHEBLEND must be 0 or 1; got ${ENABLE_CACHEBLEND}." >&2
  exit 2
fi
if [[ "${ENABLE_CACHEBLEND}" != "1" ]]; then
  echo "CacheBlend is disabled by default. Re-run with ENABLE_CACHEBLEND=1." >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}/vllm:${REPO_ROOT}/LMCache${PYTHONPATH:+:${PYTHONPATH}}"

validate_separator() {
  "${PYTHON}" - "${MODEL_PATH}" <<'PY'
import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
separator = "§CACHEBLEND§"
needle = tokenizer.encode(separator, add_special_tokens=False)
messages = [
    {
        "role": "user",
        "content": f"first document\n{separator}\nsecond document\n{separator}\nquery",
    }
]
encoded = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
)
tokens = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
matches = sum(
    tokens[index : index + len(needle)] == needle
    for index in range(len(tokens) - len(needle) + 1)
)
if not needle or matches != 2:
    raise SystemExit(
        f"CacheBlend separator preflight failed: needle={needle}, matches={matches}"
    )
print(f"CacheBlend separator preflight passed: tokens={needle}, matches={matches}")
PY
}

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
  SERVER_LOG=""
}

trap stop_server EXIT INT TERM

wait_for_server() {
  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  while ((SECONDS < deadline)); do
    if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "vLLM exited before becoming ready." >&2
      return 1
    fi
    if curl --silent --fail --max-time 2 \
      "http://${HOST}:${PORT}/v1/models" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for vLLM on ${HOST}:${PORT}." >&2
  return 1
}

start_server() {
  local variant="$1"
  local output_dir="$2"
  local -a connector_args=()
  mkdir -p "${output_dir}"
  if [[ "${variant}" == "cacheblend" ]]; then
    connector_args=(
      --kv-transfer-config
      '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"use_layerwise":true}}'
    )
  fi

  echo "Starting ${variant} server for ${output_dir}..."
  SERVER_LOG="${output_dir}/vllm_server.log"
  LMCACHE_CONFIG_FILE="$([[ "${variant}" == "cacheblend" ]] && printf '%s' "${LMCACHE_CONFIG}" || true)" \
    PYTHONHASHSEED=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
  "${VLLM_BIN}" serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype "${DTYPE}" \
    --attention-backend FLASH_ATTN \
    --enforce-eager \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --generation-config vllm \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    "${connector_args[@]}" \
    >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  wait_for_server
}

warm_backend() {
  "${PYTHON}" - "http://${HOST}:${PORT}/v1" "${SERVED_MODEL_NAME}" <<'PY'
import asyncio
import sys
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(api_key="local", base_url=sys.argv[1])
    separator = "\n§CACHEBLEND§\n"
    documents = [
        "unrelated alpha kernel warmup document " * 320,
        "unrelated beta transfer warmup document " * 320,
    ]
    async def request(order):
        context = separator + separator.join(documents[index] for index in order)
        await client.chat.completions.create(
            model=sys.argv[2],
            messages=[
                {"role": "system", "content": "Agentrix benchmark kernel warmup"},
                {"role": "user", "content": context + separator + "Reply OK."},
            ],
            temperature=0,
            max_tokens=1,
        )

    # The first three calls populate the cache and exercise a real reordered
    # blend twice.  Concurrent calls then warm the fused RoPE batch shapes used
    # by branch fan-out; without this, the first measured branch wave includes
    # a one-time CUDA extension/JIT cost.
    for order in ((0, 1), (1, 0), (0, 1)):
        await request(order)
    for width in (2, 4, 8):
        await asyncio.gather(
            *(request((index % 2, (index + 1) % 2)) for index in range(width))
        )

asyncio.run(main())
PY
}

capture_trace() {
  local scenario="$1"
  local output="$2"
  "${PYTHON}" -m langgraph_runner live \
    --base-url "http://${HOST}:${PORT}/v1" \
    --model "${SERVED_MODEL_NAME}" \
    --task-file "${TASK_FILE}" \
    --scenario "${scenario}" \
    --cases "${CASES_PER_SCENARIO}" \
    --rag-root "${RAG_ROOT}" \
    --rag-format cacheblend \
    --concurrency "${CONCURRENCY}" \
    --planner-tokens 96 \
    --tool-tokens 64 \
    --reflect-tokens 128 \
    --reduce-tokens 128 \
    --output "${output}"
}

replay_trace() {
  local trace="$1"
  local output="$2"
  "${PYTHON}" -m langgraph_runner replay \
    --base-url "http://${HOST}:${PORT}/v1" \
    --model "${SERVED_MODEL_NAME}" \
    --trace "${trace}" \
    --timing agent \
    --concurrency "${CONCURRENCY}" \
    --output "${output}"
}

mkdir -p "${OUTPUT_ROOT}"
validate_separator

for scenario in ${SCENARIOS}; do
  scenario_root="${OUTPUT_ROOT}/${scenario}"
  baseline_root="${scenario_root}/baseline"
  cacheblend_root="${scenario_root}/cacheblend"
  mkdir -p "${baseline_root}" "${cacheblend_root}"

  capture_root="${scenario_root}/capture"
  start_server baseline "${capture_root}"
  warm_backend
  capture_trace "${scenario}" "${scenario_root}/trace.json"
  stop_server

  for variant in baseline cacheblend; do
    variant_root="${scenario_root}/${variant}"
    for ((repeat = 1; repeat <= REPEATS; repeat++)); do
      repeat_root="${variant_root}/repeat${repeat}"
      start_server "${variant}" "${repeat_root}"
      warm_backend
      measured_log_start=$(( $(wc -l <"${SERVER_LOG}") + 1 ))
      replay_trace \
        "${scenario_root}/trace.json" \
        "${variant_root}/run${repeat}.json"
      sed -n "${measured_log_start},\$p" "${SERVER_LOG}" \
        >"${repeat_root}/measured_server.log"
      curl --silent --fail --max-time 10 "http://${HOST}:${PORT}/metrics" \
        >"${repeat_root}/metrics.prom" || true
      stop_server
    done
  done
done

"${PYTHON}" -m cacheblend_rag_report --root "${OUTPUT_ROOT}"
