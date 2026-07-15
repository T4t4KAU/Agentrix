#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
VLLM_BIN="${VLLM_BIN:-${REPO_ROOT}/vllm/.venv/bin/vllm}"
PYTHON="${BENCHMARK_PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/hwx/Documents/models/Qwen3-0.6B}"
MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-agentrix-e2e}"
TASK_FILE="${TASK_FILE:-${BENCHMARK_DIR}/configs/langgraph_fork_cacheblend_20_tasks.jsonl}"
LMCACHE_CONFIG="${LMCACHE_CONFIG:-${BENCHMARK_DIR}/configs/lmcache_cacheblend_rag.yaml}"
RAG_MANIFEST="${RAG_MANIFEST:-${BENCHMARK_DIR}/configs/langgraph_rag_corpus_files.txt}"
EXPECTED_RAG_CORPUS_VERSION="${EXPECTED_RAG_CORPUS_VERSION:-b01c0c10bb027921}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BENCHMARK_DIR}/results/langgraph_fork_cacheblend_20_e2e}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9000}"
CONCURRENCY="${CONCURRENCY:-16}"
CASE_CONCURRENCY="${CASE_CONCURRENCY:-1}"
CASES="${CASES:-20}"
VARIANTS="${VARIANTS:-cacheblend forkattention baseline}"
PROMPT_COMPACTION="${PROMPT_COMPACTION:-0}"
SERVER_PID=""
SERVER_LOG=""
MEMORY_SAMPLER_PID=""
MEMORY_SAMPLE_INTERVAL="${MEMORY_SAMPLE_INTERVAL:-0.5}"

export PYTHONPATH="${REPO_ROOT}/application/src:${REPO_ROOT}/vllm:${REPO_ROOT}/LMCache${PYTHONPATH:+:${PYTHONPATH}}"

stop_server() {
  if [[ -n "${MEMORY_SAMPLER_PID}" ]] && kill -0 "${MEMORY_SAMPLER_PID}" 2>/dev/null; then
    kill -TERM "${MEMORY_SAMPLER_PID}" 2>/dev/null || true
    wait "${MEMORY_SAMPLER_PID}" 2>/dev/null || true
  fi
  MEMORY_SAMPLER_PID=""
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT INT TERM

wait_server() {
  local deadline=$((SECONDS + 300))
  while ((SECONDS < deadline)); do
    if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "vLLM exited during startup; inspect ${SERVER_LOG}" >&2
      return 1
    fi
    curl --silent --fail --max-time 2 "http://${HOST}:${PORT}/v1/models" >/dev/null && return 0
    sleep 1
  done
  return 1
}

start_server() {
  local variant="$1"
  local output_dir="${OUTPUT_ROOT}/${variant}"
  local backend="FLASH_ATTN"
  local config_file=""
  local -a connector=()
  local -a eager=()
  local -a prefix_caching=(--enable-prefix-caching)
  if [[ "${variant}" == forkattention* ]]; then
    backend="FORK_ATTN"
  elif [[ "${variant}" == cacheblend* ]]; then
    config_file="${LMCACHE_CONFIG}"
    connector=(--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"use_layerwise":true}}')
    eager=(--enforce-eager)
    # CacheBlend owns reuse admission for this path. Combining it with APC can
    # produce overlapping partial hits and inconsistent layerwise Q/K lengths.
    prefix_caching=(--no-enable-prefix-caching)
  fi
  mkdir -p "${output_dir}"
  nvidia-smi --query-gpu=timestamp,memory.used,memory.free \
    --format=csv,noheader,nounits >"${output_dir}/gpu_before_server.csv"
  SERVER_LOG="${output_dir}/vllm_server.log"
  LMCACHE_CONFIG_FILE="${config_file}" PYTHONHASHSEED=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_FORK_ATTN_ENABLE_FOREST=1 \
    VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH=1 \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --served-model-name "${MODEL_NAME}" --host "${HOST}" --port "${PORT}" \
      --dtype bfloat16 --attention-backend "${backend}" \
      "${eager[@]}" \
      "${prefix_caching[@]}" --no-async-scheduling \
      --enable-auto-tool-choice --tool-call-parser hermes \
      --generation-config vllm \
      --default-chat-template-kwargs '{"enable_thinking":false}' \
      --gpu-memory-utilization 0.75 --max-model-len 32768 \
      --max-num-batched-tokens 16384 --max-num-seqs 32 \
      "${connector[@]}" >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  wait_server
}

start_memory_sampler() {
  local output_dir="$1"
  "${PYTHON}" - "${HOST}" "${PORT}" "${SERVER_PID}" "${MEMORY_SAMPLE_INTERVAL}" \
    "${output_dir}/memory_samples.csv" <<'PY' &
import csv
import re
import subprocess
import sys
import time
import urllib.request

host, port, root_pid, interval, output = sys.argv[1:]
metrics = (
    "vllm:kv_cache_usage_perc",
    "process_resident_memory_bytes",
    "lmcache:local_cache_usage",
    "lmcache:remote_cache_usage",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
)

def metric(text, name):
    values = re.findall(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+(\S+)$", text, re.M)
    return sum(float(value) for value in values) if values else ""

def process_tree_rss(pid):
    pending = [pid]
    seen = set()
    total_kib = 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            children = open(
                f"/proc/{current}/task/{current}/children", encoding="utf-8"
            ).read().split()
            pending.extend(children)
            status = open(f"/proc/{current}/status", encoding="utf-8").read()
            match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.M)
            total_kib += int(match.group(1)) if match else 0
        except OSError:
            pass
    return total_kib * 1024

with open(output, "w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow((
        "unix_s", "gpu_used_mib", "gpu_free_mib", "gpu_util_pct",
        "memory_controller_util_pct", "server_tree_rss_bytes", *metrics,
    ))
    while True:
        try:
            gpu = subprocess.check_output(
                (
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.free,utilization.gpu,utilization.memory",
                    "--format=csv,noheader,nounits",
                ),
                text=True,
                timeout=2,
            ).strip().splitlines()[0].split(", ")
            with urllib.request.urlopen(
                f"http://{host}:{port}/metrics", timeout=2
            ) as response:
                prometheus = response.read().decode("utf-8", "replace")
            writer.writerow((
                time.time(), *gpu, process_tree_rss(root_pid),
                *(metric(prometheus, name) for name in metrics),
            ))
            handle.flush()
        except Exception:
            pass
        time.sleep(float(interval))
PY
  MEMORY_SAMPLER_PID=$!
}

stop_memory_sampler() {
  if [[ -n "${MEMORY_SAMPLER_PID}" ]] && kill -0 "${MEMORY_SAMPLER_PID}" 2>/dev/null; then
    kill -TERM "${MEMORY_SAMPLER_PID}" 2>/dev/null || true
    wait "${MEMORY_SAMPLER_PID}" 2>/dev/null || true
  fi
  MEMORY_SAMPLER_PID=""
}

warm_server() {
  "${PYTHON}" - "http://${HOST}:${PORT}/v1" "${MODEL_NAME}" <<'PY'
import asyncio
import sys
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(api_key="local", base_url=sys.argv[1])
    sep = "\n§CACHEBLEND§\n"
    docs = ["unrelated warmup alpha " * 700, "unrelated warmup beta " * 700]
    async def request(index):
        order = (index % 2, (index + 1) % 2)
        context = sep + sep.join(docs[item] for item in order)
        await client.chat.completions.create(
            model=sys.argv[2],
            messages=[{"role":"system","content":"Agentrix warmup"},
                      {"role":"user","content":context + sep + "Reply OK."}],
            temperature=0, max_tokens=1,
        )
    await request(0)
    await request(1)
    await request(0)
    for width in (2, 4, 8, 16):
        await asyncio.gather(*(request(index) for index in range(width)))

asyncio.run(main())
PY
}

run_variant() {
  local variant="$1"
  local output_dir="${OUTPUT_ROOT}/${variant}"
  start_server "${variant}"
  warm_server
  curl --silent --fail --max-time 10 "http://${HOST}:${PORT}/metrics" \
    >"${output_dir}/metrics_before.prom" || true
  nvidia-smi --query-gpu=timestamp,memory.used,memory.free \
    --format=csv,noheader,nounits >"${output_dir}/gpu_after_warm.csv"
  local measured_start=$(( $(wc -l <"${SERVER_LOG}") + 1 ))
  start_memory_sampler "${output_dir}"
  local -a compaction=()
  if [[ "${PROMPT_COMPACTION}" == "1" || "${variant}" == *_compact ]]; then
    compaction=(--prompt-compaction)
  fi
  "${PYTHON}" -m langgraph_runner live \
    --base-url "http://${HOST}:${PORT}/v1" --model "${MODEL_NAME}" \
    --task-file "${TASK_FILE}" --cases "${CASES}" \
    --case-concurrency "${CASE_CONCURRENCY}" --rag-root "${REPO_ROOT}/docs" \
    --rag-manifest "${RAG_MANIFEST}" \
    --expected-rag-corpus-version "${EXPECTED_RAG_CORPUS_VERSION}" \
    --rag-format cacheblend --concurrency "${CONCURRENCY}" \
    "${compaction[@]}" \
    --planner-tokens 128 --tool-tokens 64 --reflect-tokens 256 --reduce-tokens 192 \
    --output "${output_dir}/run.json"
  stop_memory_sampler
  sed -n "${measured_start},\$p" "${SERVER_LOG}" >"${output_dir}/measured_server.log"
  curl --silent --fail --max-time 10 "http://${HOST}:${PORT}/metrics" \
    >"${output_dir}/metrics.prom" || true
  stop_server
}

mkdir -p "${OUTPUT_ROOT}"
for variant in ${VARIANTS}; do
  run_variant "${variant}"
done
"${PYTHON}" -m langgraph_e2e_report --root "${OUTPUT_ROOT}"
