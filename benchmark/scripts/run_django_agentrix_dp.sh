#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
PYTHON="${BENCHMARK_PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-${REPO_ROOT}/vllm/.venv/bin/vllm}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
DEFAULT_CASES_PATHS="${BENCHMARK_DIR}/data/django_agentrix/cases_30k_b16.jsonl:${BENCHMARK_DIR}/data/sqlite_agentrix/cases_30k_b16.jsonl:${BENCHMARK_DIR}/data/ffmpeg_agentrix/cases_30k_b16.jsonl"
CASES_PATHS="${CASES_PATHS:-${CASES_PATH:-${DEFAULT_CASES_PATHS}}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BENCHMARK_DIR}/results/coding_agentrix_dp8}"
MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-32b-coding-agentrix}"
VARIANTS="${VARIANTS:-flash_uncompressed_dp fork_prefix_aware_compact_dp}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
DP_REPLICAS="${DP_REPLICAS:-8}"
PORT="${PORT:-9000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-3852}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
BRANCH_OUTPUT_TOKENS="${BRANCH_OUTPUT_TOKENS:-64}"
ROUNDS="${ROUNDS:-3}"
TRAJECTORY_MODE="${TRAJECTORY_MODE:-replay}"
CASE_LIMIT="${CASE_LIMIT:-0}"
SERVER_PID=""
SAMPLER_PID=""

IFS=: read -r -a CASE_FILES <<<"${CASES_PATHS}"
for case_file in "${CASE_FILES[@]}"; do
  if [[ ! -f "${case_file}" ]]; then
    echo "Coding-agent case file does not exist: ${case_file}" >&2
    exit 2
  fi
done
IFS=, read -r -a GPU_ID_LIST <<<"${GPU_IDS}"
if [[ "${#GPU_ID_LIST[@]}" -ne "${DP_REPLICAS}" ]]; then
  echo "GPU_IDS must contain exactly DP_REPLICAS=${DP_REPLICAS} entries" >&2
  exit 2
fi

export PYTHONPATH="${BENCHMARK_DIR}/src:${REPO_ROOT}/vllm${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH="${REPO_ROOT}/application/src:${PYTHONPATH}"

stop_server() {
  if [[ -n "${SAMPLER_PID}" ]] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then
    kill -TERM "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
  fi
  SAMPLER_PID=""
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT INT TERM

wait_server() {
  local log="$1"
  local deadline=$((SECONDS + 600))
  until curl --silent --fail --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "vLLM exited during startup; inspect ${log}" >&2
      tail -n 100 "${log}" >&2
      return 1
    fi
    if ((SECONDS >= deadline)); then
      echo "Timed out waiting for vLLM; inspect ${log}" >&2
      tail -n 100 "${log}" >&2
      return 1
    fi
    sleep 2
  done
}

run_variant() {
  local variant="$1"
  local backend="FLASH_ATTN"
  local prefix_routing=0
  local prompt_compaction=0
  local dp_policy="ordinary"
  if [[ "${variant}" == "fork_prefix_aware_compact_dp" ]]; then
    backend="FORK_ATTN"
    prefix_routing=1
    prompt_compaction=1
    dp_policy="prefix_aware"
  elif [[ "${variant}" != "flash_uncompressed_dp" ]]; then
    echo "Unknown variant: ${variant}" >&2
    return 2
  fi
  local output_dir="${OUTPUT_ROOT}/${variant}"
  mkdir -p "${output_dir}"
  local log="${output_dir}/vllm_server.log"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    PYTHONHASHSEED=0 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_FORK_ATTN_ENABLE_FOREST=1 \
    VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH=1 \
    VLLM_FORK_ATTN_FANOUT_SCHEDULING_ENABLED="${prefix_routing}" \
    VLLM_FORK_ATTN_FANOUT_ADMISSION_WINDOW=0 \
    VLLM_FORK_ATTN_DP_PREFIX_ROUTING="${prefix_routing}" \
    VLLM_FORK_ATTN_DP_RELOAD_REBALANCE=0 \
    VLLM_FORK_ATTN_DP_ARRIVAL_WAVE_MS=10 \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --host 127.0.0.1 --port "${PORT}" \
      --served-model-name "${MODEL_NAME}" \
      --attention-backend "${backend}" --dtype float16 \
      --generation-config vllm --enable-prefix-caching --no-async-scheduling \
      --default-chat-template-kwargs '{"enable_thinking":false}' \
      --data-parallel-size "${DP_REPLICAS}" --api-server-count 1 \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --num-gpu-blocks-override "${NUM_GPU_BLOCKS_OVERRIDE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${MAX_NUM_SEQS}" >"${log}" 2>&1 &
  SERVER_PID=$!
  wait_server "${log}"
  curl --silent --fail "http://127.0.0.1:${PORT}/metrics" \
    >"${output_dir}/metrics_before.prom" || true
  "${PYTHON}" -m resource_sampler --server-pid "${SERVER_PID}" \
    --metrics-url "http://127.0.0.1:${PORT}/metrics" \
    --output "${output_dir}/resource_samples.jsonl" &
  SAMPLER_PID=$!
  local -a compaction_args=()
  if [[ "${prompt_compaction}" == "1" ]]; then
    compaction_args+=(--prompt-compaction)
  fi
  AGENTRIX_TOOL_KV_TRIM_ENABLED=0 \
  AGENTRIX_TOOL_KV_TRIM_USE_PREDICTED_TTL=0 \
  "${PYTHON}" -m django_agentrix_runner \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --model "${MODEL_NAME}" --cases "${CASE_FILES[@]}" \
    --experiment-variant "${variant}" \
    --attention-backend "${backend}" --dp-policy "${dp_policy}" \
    --case-limit "${CASE_LIMIT}" \
    --branch-output-tokens "${BRANCH_OUTPUT_TOKENS}" \
    --rounds "${ROUNDS}" --trajectory-mode "${TRAJECTORY_MODE}" \
    "${compaction_args[@]}" \
    --output "${output_dir}/run.json"
  kill -TERM "${SAMPLER_PID}" 2>/dev/null || true
  wait "${SAMPLER_PID}" 2>/dev/null || true
  SAMPLER_PID=""
  curl --silent --fail "http://127.0.0.1:${PORT}/metrics" \
    >"${output_dir}/metrics_after.prom" || true
  stop_server
}

mkdir -p "${OUTPUT_ROOT}"
for variant in ${VARIANTS}; do
  run_variant "${variant}"
done

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for path in sorted(root.glob("*/run.json")):
    variant = path.parent.name
    if path.exists():
        data = json.loads(path.read_text())
        summary[variant] = {
            key: data[key]
            for key in (
                "case_count", "branch_count", "branch_request_count",
                "repositories", "repository_metrics",
                "experiment_variant", "attention_backend", "dp_policy",
                "round_count", "trajectory_mode", "wall_time_ms",
                "prompt_compaction", "compaction",
                "branch_wall_time_ms", "branch_input_tokens",
                "branch_output_tokens", "branch_output_tokens_per_s",
                "branch_ttft_ms", "branch_tpot_ms", "round_metrics",
            )
        }
(root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
