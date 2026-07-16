#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
VLLM_BIN="${VLLM_BIN:-${REPO_ROOT}/vllm/.venv/bin/vllm}"
PYTHON="${BENCHMARK_PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-agentrix-e2e}"
HOTPOT_PATH="${HOTPOT_PATH:-}"
HOTPOT_CASE_FILE="${HOTPOT_CASE_FILE:-${BENCHMARK_DIR}/configs/hotpot_agentrix_long_prefix_100.jsonl}"
LMCACHE_CONFIG="${LMCACHE_CONFIG:-${BENCHMARK_DIR}/configs/lmcache_cacheblend_rag.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BENCHMARK_DIR}/results/hotpot_agentrix_100_e2e}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9000}"
CONCURRENCY="${CONCURRENCY:-20}"
CASE_CONCURRENCY="${CASE_CONCURRENCY:-2}"
CASES="${CASES:-100}"
BOOTSTRAP_CHUNKS="${BOOTSTRAP_CHUNKS:-40}"
BOOTSTRAP_MAX_CHARS="${BOOTSTRAP_MAX_CHARS:-100000}"
HOTPOT_BRANCHES="${HOTPOT_BRANCHES:-10}"
ENABLE_CACHEBLEND="${ENABLE_CACHEBLEND:-0}"
VARIANTS="${VARIANTS:-baseline forkattention}"
PROMPT_COMPACTION="${PROMPT_COMPACTION:-0}"
SERVER_PID=""
SERVER_LOG=""
MEMORY_SAMPLER_PID=""
MEMORY_SAMPLE_INTERVAL="${MEMORY_SAMPLE_INTERVAL:-0.5}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
GPU_KV_CACHE_GIB="${GPU_KV_CACHE_GIB:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
CPU_OFFLOAD_GIB="${CPU_OFFLOAD_GIB:-0.5}"
OFFLOAD_BLOCK_SIZE="${OFFLOAD_BLOCK_SIZE:-16}"
FS_READ_THREADS="${FS_READ_THREADS:-4}"
FS_WRITE_THREADS="${FS_WRITE_THREADS:-4}"
OFFLOAD_FANOUT_OPTIMIZATION="${OFFLOAD_FANOUT_OPTIMIZATION:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
TRACE_PATH="${TRACE_PATH:-}"
REPLAY_TIMING="${REPLAY_TIMING:-agent}"
REPLAY_GAP_MS="${REPLAY_GAP_MS:-0}"
FIXED_OUTPUT_TOKENS="${FIXED_OUTPUT_TOKENS:-0}"
FIXED_OUTPUT_LENGTH_SOURCE="${FIXED_OUTPUT_LENGTH_SOURCE:-max}"

if [[ "${ENABLE_CACHEBLEND}" != "0" && "${ENABLE_CACHEBLEND}" != "1" ]]; then
  echo "ENABLE_CACHEBLEND must be 0 or 1; got ${ENABLE_CACHEBLEND}." >&2
  exit 2
fi
CACHEBLEND_REQUESTED=0
for variant in ${VARIANTS}; do
  if [[ "${variant}" == cacheblend* ]]; then
    CACHEBLEND_REQUESTED=1
  fi
done
if [[ -z "${RAG_FORMAT:-}" ]]; then
  RAG_FORMAT="plain"
  if [[ "${CACHEBLEND_REQUESTED}" == "1" ]]; then
    RAG_FORMAT="cacheblend"
  fi
fi
if [[ "${CACHEBLEND_REQUESTED}" == "1" && "${ENABLE_CACHEBLEND}" != "1" ]]; then
  echo "CacheBlend is disabled by default. Re-run with ENABLE_CACHEBLEND=1." >&2
  exit 2
fi
if [[ "${CACHEBLEND_REQUESTED}" == "1" && "${RAG_FORMAT}" != "cacheblend" ]]; then
  echo "CacheBlend variants require RAG_FORMAT=cacheblend." >&2
  exit 2
fi
if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH must point to the local model directory." >&2
  exit 2
fi
if [[ -z "${HOTPOT_PATH}" ]]; then
  echo "HOTPOT_PATH must point to the HotpotQA distractor JSON file." >&2
  exit 2
fi
if [[ ! -f "${HOTPOT_PATH}" ]]; then
  echo "HotpotQA source not found: ${HOTPOT_PATH}" >&2
  exit 2
fi
if [[ ! -f "${HOTPOT_CASE_FILE}" ]]; then
  echo "HotpotQA case manifest not found: ${HOTPOT_CASE_FILE}" >&2
  exit 2
fi
if [[ -n "${TRACE_PATH}" && ! -f "${TRACE_PATH}" ]]; then
  echo "Replay trace not found: ${TRACE_PATH}" >&2
  exit 2
fi

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
  # SharedOffloadRegion intentionally uses persistent files. Remove only the
  # mmap paths announced by the server that just exited, otherwise sequential
  # variants can silently fill /dev/shm and invalidate the next run.
  if [[ -n "${SERVER_LOG}" && -f "${SERVER_LOG}" ]]; then
    while IFS= read -r mmap_path; do
      if [[ "${mmap_path}" == /dev/shm/vllm_offload_*.mmap ]]; then
        find "${mmap_path}" -maxdepth 0 -type f -delete 2>/dev/null || true
      fi
    done < <(rg -o '/dev/shm/vllm_offload_[0-9]+\.mmap' "${SERVER_LOG}" | sort -u)
  fi
  SERVER_LOG=""
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
  local -a gpu_kv_cache=()
  local -a prefix_caching=(--enable-prefix-caching)
  if [[ -n "$(ss -H -ltn "sport = :${PORT}" 2>/dev/null)" ]]; then
    echo "Port ${PORT} is already in use; refusing to connect to a stale server." >&2
    return 1
  fi
  if [[ "${variant}" == fork* ]]; then
    backend="FORK_ATTN"
  elif [[ "${variant}" == cacheblend* ]]; then
    config_file="${LMCACHE_CONFIG}"
    connector=(--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"use_layerwise":true}}')
    eager=(--enforce-eager)
    # CacheBlend owns reuse admission for this path. Combining it with APC can
    # produce overlapping partial hits and inconsistent layerwise Q/K lengths.
    prefix_caching=(--no-enable-prefix-caching)
  fi
  if [[ "${variant}" == *_cpu || "${variant}" == *_tiered ]]; then
    local tier_mode="cpu"
    local fs_root="${output_dir}/kv_fs"
    [[ "${variant}" == *_tiered ]] && tier_mode="tiered"
    if [[ "${tier_mode}" == "tiered" ]]; then
      mkdir -p "${fs_root}"
      find "${fs_root}" -mindepth 1 -delete
    fi
    local connector_json
    connector_json="$("${PYTHON}" - "${tier_mode}" "${CPU_OFFLOAD_GIB}" \
      "${OFFLOAD_BLOCK_SIZE}" "${fs_root}" "${FS_READ_THREADS}" \
      "${FS_WRITE_THREADS}" "${OFFLOAD_FANOUT_OPTIMIZATION}" "${variant}" <<'PY'
import json
import sys

(
    mode,
    cpu_gib,
    block_size,
    fs_root,
    read_threads,
    write_threads,
    fanout_optimization,
    variant,
) = sys.argv[1:]
extra = {
    "cpu_bytes_to_use": int(float(cpu_gib) * 1024**3),
    "block_size": int(block_size),
    "eviction_policy": "lru",
    "offload_prompt_only": True,
}
if fanout_optimization == "1":
    extra.update({
        "fanout_offload": True,
        "fanout_profile": True,
        "fanout_allow_hot_prefix_backup": True,
    })
elif variant.startswith("fork"):
    # FORK_ATTN enables fanout admission by default when the key is absent.
    # Standard-policy comparisons must opt out explicitly.
    extra["fanout_offload"] = False
if mode == "tiered":
    extra.update({
        "spec_name": "TieringOffloadingSpec",
        "secondary_tiers": [{
            "type": "fs",
            "root_dir": fs_root,
            "n_read_threads": int(read_threads),
            "n_write_threads": int(write_threads),
        }],
    })
print(json.dumps({
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": extra,
}))
PY
)"
    connector=(--kv-transfer-config "${connector_json}")
  fi
  if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    eager=(--enforce-eager)
  fi
  if [[ -n "${GPU_KV_CACHE_GIB}" ]]; then
    local gpu_kv_cache_bytes
    gpu_kv_cache_bytes="$(${PYTHON} -c \
      'import sys; print(int(float(sys.argv[1]) * 1024**3))' \
      "${GPU_KV_CACHE_GIB}")"
    gpu_kv_cache=(--kv-cache-memory-bytes "${gpu_kv_cache_bytes}")
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
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" --max-model-len "${MAX_MODEL_LEN}" \
      "${gpu_kv_cache[@]}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" --max-num-seqs 32 \
      "${connector[@]}" >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  wait_server
}

start_memory_sampler() {
  local output_dir="$1"
  "${PYTHON}" - "${HOST}" "${PORT}" "${SERVER_PID}" "${MEMORY_SAMPLE_INTERVAL}" \
    "${output_dir}/memory_samples.csv" "${output_dir}/kv_fs" <<'PY' &
import csv
import os
import re
import subprocess
import sys
import time
import urllib.request

try:
    import pynvml
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    pynvml = None
    nvml_handle = None

host, port, root_pid, interval, output, fs_root = sys.argv[1:]
metrics = (
    "vllm:kv_cache_usage_perc",
    "process_resident_memory_bytes",
    "lmcache:local_cache_usage",
    "lmcache:remote_cache_usage",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_offload_cpu_cache_usage_perc",
    "vllm:kv_offload_cpu_cache_occupancy_perc",
    "vllm:kv_offload_load_bytes",
    "vllm:kv_offload_load_time",
    "vllm:kv_offload_store_bytes",
    "vllm:kv_offload_store_time",
)

def metric(text, name):
    values = re.findall(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+(\S+)$", text, re.M)
    return sum(float(value) for value in values) if values else ""

def process_tree(pid):
    pending = [pid]
    seen = set()
    total_kib = 0
    io_totals = {
        name: 0 for name in ("rchar", "wchar", "read_bytes", "write_bytes")
    }
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
            io_text = open(f"/proc/{current}/io", encoding="utf-8").read()
            for name in io_totals:
                io_match = re.search(rf"^{name}:\s+(\d+)$", io_text, re.M)
                io_totals[name] += int(io_match.group(1)) if io_match else 0
        except OSError:
            pass
    return total_kib * 1024, io_totals

def disk_usage(root):
    files = 0
    total = 0
    if not os.path.isdir(root):
        return files, total
    for directory, _, names in os.walk(root):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(directory, name))
                files += 1
            except OSError:
                pass
    return files, total

def pcie_throughput():
    if pynvml is None or nvml_handle is None:
        return "", ""
    try:
        rx = pynvml.nvmlDeviceGetPcieThroughput(
            nvml_handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
        )
        tx = pynvml.nvmlDeviceGetPcieThroughput(
            nvml_handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
        )
        return rx, tx
    except Exception:
        return "", ""

with open(output, "w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow((
        "unix_s", "gpu_used_mib", "gpu_free_mib", "gpu_util_pct",
        "memory_controller_util_pct", "server_tree_rss_bytes", *metrics,
        "process_rchar_bytes", "process_wchar_bytes", "process_read_bytes",
        "process_write_bytes", "fs_cache_files", "fs_cache_bytes",
        "pcie_rx_kib_s", "pcie_tx_kib_s",
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
            rss, io = process_tree(root_pid)
            disk_files, disk_bytes = disk_usage(fs_root)
            pcie_rx, pcie_tx = pcie_throughput()
            writer.writerow((
                time.time(), *gpu, rss,
                *(metric(prometheus, name) for name in metrics),
                io["rchar"], io["wchar"], io["read_bytes"], io["write_bytes"],
                disk_files, disk_bytes,
                pcie_rx, pcie_tx,
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

wait_offload_quiescent() {
  local output_dir="$1"
  local phase="$2"
  "${PYTHON}" - "${HOST}" "${PORT}" "${output_dir}/kv_fs" \
    "${output_dir}/${phase}_drain.json" <<'PY'
import json
import os
import re
import sys
import time
import urllib.request

host, port, fs_root, output = sys.argv[1:]
interval_s = 0.5
stable_required = 4
deadline = time.monotonic() + 120.0
started = time.monotonic()
previous_fs = None
stable_samples = 0
samples = 0
usage = 0.0

def filesystem_state(root):
    files = 0
    total = 0
    latest_mtime_ns = 0
    if not os.path.isdir(root):
        return files, total, latest_mtime_ns
    for directory, _, names in os.walk(root):
        for name in names:
            try:
                stat = os.stat(os.path.join(directory, name))
                files += 1
                total += stat.st_size
                latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
            except OSError:
                pass
    return files, total, latest_mtime_ns

timed_out = False
while True:
    samples += 1
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/metrics", timeout=2
        ) as response:
            metrics = response.read().decode("utf-8", "replace")
        values = re.findall(
            r"^vllm:kv_offload_cpu_cache_usage_perc(?:\{[^}]*\})?\s+(\S+)$",
            metrics,
            re.M,
        )
        usage = sum(float(value) for value in values) if values else 0.0
    except Exception:
        usage = float("inf")
    fs_state = filesystem_state(fs_root)
    if usage <= 0.0 and fs_state == previous_fs:
        stable_samples += 1
    else:
        stable_samples = 0
    previous_fs = fs_state
    if stable_samples >= stable_required:
        break
    if time.monotonic() >= deadline:
        timed_out = True
        break
    time.sleep(interval_s)

payload = {
    "elapsed_s": time.monotonic() - started,
    "confirmation_window_s": stable_required * interval_s,
    "samples": samples,
    "timed_out": timed_out,
    "final_cpu_cache_usage_fraction": usage,
    "final_fs_files": previous_fs[0] if previous_fs else 0,
    "final_fs_bytes": previous_fs[1] if previous_fs else 0,
}
with open(output, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
if timed_out:
    raise SystemExit("Timed out waiting for KV offload to become quiescent")
PY
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
  wait_offload_quiescent "${output_dir}" "warmup"
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
  if [[ -n "${TRACE_PATH}" ]]; then
    local -a fixed_output=()
    if [[ "${FIXED_OUTPUT_TOKENS}" == "1" ]]; then
      fixed_output=(--fixed-output-tokens)
    fi
    "${PYTHON}" -m langgraph_runner replay \
      --base-url "http://${HOST}:${PORT}/v1" --model "${MODEL_NAME}" \
      --trace "${TRACE_PATH}" --timing "${REPLAY_TIMING}" \
      --case-concurrency "${CASE_CONCURRENCY}" \
      --sequential-gap-ms "${REPLAY_GAP_MS}" \
      --fixed-output-length-source "${FIXED_OUTPUT_LENGTH_SOURCE}" \
      --concurrency "${CONCURRENCY}" "${fixed_output[@]}" \
      --output "${output_dir}/run.json"
  else
    "${PYTHON}" -m langgraph_runner live \
      --base-url "http://${HOST}:${PORT}/v1" --model "${MODEL_NAME}" \
      --hotpot-path "${HOTPOT_PATH}" --hotpot-case-file "${HOTPOT_CASE_FILE}" \
      --cases "${CASES}" \
      --case-concurrency "${CASE_CONCURRENCY}" \
      --bootstrap-chunks "${BOOTSTRAP_CHUNKS}" \
      --bootstrap-max-chars "${BOOTSTRAP_MAX_CHARS}" \
      --hotpot-branch-min "${HOTPOT_BRANCHES}" \
      --hotpot-branch-max "${HOTPOT_BRANCHES}" \
      --hotpot-tool-delay-profile synchronized \
      --rag-format "${RAG_FORMAT}" --concurrency "${CONCURRENCY}" \
      "${compaction[@]}" \
      --planner-tokens 128 --tool-tokens 64 --reflect-tokens 256 --reduce-tokens 192 \
      --output "${output_dir}/run.json"
  fi
  wait_offload_quiescent "${output_dir}" "measured"
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
if [[ -f "${OUTPUT_ROOT}/baseline/run.json" ]]; then
  "${PYTHON}" -m langgraph_e2e_report --root "${OUTPUT_ROOT}"
fi
if [[ " ${VARIANTS} " == *"_cpu "* || " ${VARIANTS} " == *"_tiered "* ]]; then
  "${PYTHON}" -m hotpot_offload_report --root "${OUTPUT_ROOT}"
fi
