#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
VLLM_BIN="${VLLM_BIN:-${REPO_ROOT}/vllm/.venv/bin/vllm}"
BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/test__02/hwx/Qwen3.6-27B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen36-weblinx}"
MANIFEST="${MANIFEST:-${BENCHMARK_DIR}/results/weblinx_subset/manifest.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BENCHMARK_DIR}/results/weblinx_8dp_$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PORT="${PORT:-8010}"
TEXT_PREFIX_TOKENS="${TEXT_PREFIX_TOKENS:-28000}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"
COMMON_ANALYSIS_TOKENS="${COMMON_ANALYSIS_TOKENS:-64}"
ROLLOUTS_PER_CANDIDATE="${ROLLOUTS_PER_CANDIDATE:-4}"
SUFFIX_MEAN="${SUFFIX_MEAN:-256}"
CONCURRENCY="${CONCURRENCY:-256}"
SEED="${SEED:-2026}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.65}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
PROFILE_FORK="${PROFILE_FORK:-1}"
VARIANTS="${VARIANTS:-flash_ordinary fork_ordinary fork_prefix_aware}"

SERVER_PID=""

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

trap stop_server EXIT INT TERM

if [[ ! -f "${MANIFEST}" ]]; then
  echo "WebLINX manifest does not exist: ${MANIFEST}" >&2
  echo "Build it with: python -m weblinx_data --output-dir results/weblinx_subset" >&2
  exit 1
fi
if [[ ! -x "${VLLM_BIN}" || ! -x "${BENCHMARK_PYTHON}" ]]; then
  echo "Missing vLLM or benchmark Python executable." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
cd "${BENCHMARK_DIR}"

run_variant() {
  local variant="$1"
  local backend="FORK_ATTN"
  local prefix_routing=0
  local fanout_scheduling=0
  if [[ "${variant}" == "flash_ordinary" ]]; then
    backend="FLASH_ATTN"
  elif [[ "${variant}" == "fork_prefix_aware" ]]; then
    prefix_routing=1
    fanout_scheduling=1
  elif [[ "${variant}" != "fork_ordinary" ]]; then
    echo "Unknown WebLINX variant: ${variant}" >&2
    exit 1
  fi

  local output_dir="${OUTPUT_ROOT}/${variant}"
  local server_log="${output_dir}/vllm_server.log"
  mkdir -p "${output_dir}"
  local args=(
    serve "${MODEL_PATH}"
    --host 127.0.0.1
    --port "${PORT}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --attention-backend "${backend}"
    --dtype bfloat16
    --generation-config vllm
    --enable-prefix-caching
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --tensor-parallel-size 1
    --data-parallel-size 8
    --api-server-count 1
  )
  if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    args+=(--enforce-eager)
  fi
  if [[ -n "${NUM_GPU_BLOCKS_OVERRIDE}" ]]; then
    args+=(--num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}")
  fi

  echo "Starting ${variant}: backend=${backend}, prefix_routing=${prefix_routing}"
  PROFILE_FORK="${PROFILE_FORK}" \
  VLLM_FORK_ATTN_ENABLE_FOREST=1 \
  VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH=1 \
  VLLM_FORK_ATTN_FANOUT_SCHEDULING_ENABLED="${fanout_scheduling}" \
  VLLM_FORK_ATTN_FANOUT_ADMISSION_WINDOW=0 \
  VLLM_FORK_ATTN_DP_PREFIX_ROUTING="${prefix_routing}" \
  VLLM_FORK_ATTN_DP_PREFIX_LOAD_SLACK=32 \
  VLLM_FORK_ATTN_DP_ARRIVAL_WAVE_MS=10 \
  VLLM_FORK_ATTN_DP_GRAPH_SLACK_BUCKETS=0 \
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    "${VLLM_BIN}" "${args[@]}" >"${server_log}" 2>&1 &
  SERVER_PID="$!"

  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  until curl --silent --fail --max-time 2 \
    "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -n 100 "${server_log}" >&2
      exit 1
    fi
    if ((SECONDS >= deadline)); then
      tail -n 100 "${server_log}" >&2
      exit 1
    fi
    sleep 2
  done

  OPENAI_API_KEY=vllm-local OPENAI_TIMEOUT_SECONDS=1200 \
    "${BENCHMARK_PYTHON}" -m weblinx_runner \
      --manifest "${MANIFEST}" \
      --model "${SERVED_MODEL_NAME}" \
      --base-url "http://127.0.0.1:${PORT}/v1" \
      --output-dir "${output_dir}" \
      --dp-size 8 \
      --text-prefix-tokens "${TEXT_PREFIX_TOKENS}" \
      --output-tokens "${OUTPUT_TOKENS}" \
      --common-analysis-tokens "${COMMON_ANALYSIS_TOKENS}" \
      --rollouts-per-candidate "${ROLLOUTS_PER_CANDIDATE}" \
      --suffix-mean "${SUFFIX_MEAN}" \
      --concurrency "${CONCURRENCY}" \
      --seed "${SEED}" \
      --branch-order shuffle \
      --image-mode same

  curl --silent --fail --max-time 10 \
    "http://127.0.0.1:${PORT}/metrics" >"${output_dir}/prometheus_metrics.prom" \
    || true
  stop_server
}

for variant in ${VARIANTS}; do
  run_variant "${variant}"
done

"${BENCHMARK_PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for variant in ("flash_ordinary", "fork_ordinary", "fork_prefix_aware"):
    path = root / variant / "benchmark_results.csv"
    if not path.exists():
        continue
    with path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    raw_path = root / variant / "raw_api_results.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))[0]
    rows.append((variant, row, raw))

report = root / "comparison.md"
with report.open("w", encoding="utf-8") as handle:
    handle.write("# WebLINX 8-DP Comparison\n\n")
    handle.write(
        "| Variant | Prefix tokens | Bootstrap ms | Branch wall ms | Total wall ms "
        "| Branch output tok/s | Total output tok/s | Mean latency ms |\n"
    )
    handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for variant, row, raw in rows:
        common_ms = float(raw["common"]["latency_ms"])
        branch_ms = float(raw["branch_phase_latency_ms"])
        output_tokens = int(raw["branch_total_output_tokens"])
        handle.write(
            f"| {variant} | {float(row['prefix_tokens']):.0f} | "
            f"{common_ms:.2f} | {branch_ms:.2f} | "
            f"{common_ms + branch_ms:.2f} | "
            f"{float(row['branch_output_tokens_per_s']):.2f} | "
            f"{1000 * output_tokens / (common_ms + branch_ms):.2f} | "
            f"{float(row['branch_mean_latency_ms']):.2f} |\n"
        )
PY

echo "WebLINX 8-DP results: ${OUTPUT_ROOT}"
