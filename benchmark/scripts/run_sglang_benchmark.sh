#!/usr/bin/env bash
set -Eeuo pipefail

# Run the Agentrix OpenAI-compatible API benchmark against SGLang.
BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
SGLANG_ROOT="${SGLANG_ROOT:-${REPO_ROOT}/sglang}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${SGLANG_ROOT}/.venv/bin/python}"
BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b-sglang}"
DTYPE="${DTYPE:-float16}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9000}"
TP_SIZE="${TP_SIZE:-1}"
DP_REPLICAS="${DP_REPLICAS:-1}"
DP_ROUTING="${DP_ROUTING:-single}"
GPU_IDS="${GPU_IDS:-}"
DATASET="${DATASET:-swebench}"
DATA_PATH="${DATA_PATH:-}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
CASE_COUNT="${CASE_COUNT:-1}"
SAMPLE_COUNT="${SAMPLE_COUNT:-${CASE_COUNT}}"
FULL_DATASET="${FULL_DATASET:-0}"
PREFIX_TOKENS="${PREFIX_TOKENS:-2048}"
BRANCHES="${BRANCHES:-2}"
BRANCH_GROUP_SIZE="${BRANCH_GROUP_SIZE:-1}"
BRANCH_ORDER="${BRANCH_ORDER:-round_robin}"
CONCURRENCY="${CONCURRENCY:-$((BRANCHES * CASE_COUNT))}"
SUFFIX_DISTRIBUTION="${SUFFIX_DISTRIBUTION:-lognormal}"
SUFFIX_MEAN="${SUFFIX_MEAN:-128}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
COMMON_ANALYSIS_TOKENS="${COMMON_ANALYSIS_TOKENS:-128}"
ARRIVAL_INTERVAL_MS="${ARRIVAL_INTERVAL_MS:-0}"
MINORITY_HEADSTART_MS="${MINORITY_HEADSTART_MS:-0}"
SEED="${SEED:-2026}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.70}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
SHUTDOWN_TIMEOUT="${SHUTDOWN_TIMEOUT:-30}"
WARM_SHARED_PREFIX="${WARM_SHARED_PREFIX:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/sglang_${DATASET}_c${CASE_COUNT}_p${PREFIX_TOKENS}_b${BRANCHES}_g${BRANCH_GROUP_SIZE}_o${OUTPUT_TOKENS}}"
KEEP_SERVER="${KEEP_SERVER:-0}"
ENABLE_TELEMETRY="${ENABLE_TELEMETRY:-1}"
ENABLE_METRICS="${ENABLE_METRICS:-1}"
TELEMETRY_INTERVAL_SECONDS="${TELEMETRY_INTERVAL_SECONDS:-0.5}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"
BENCHMARK_EXTRA_ARGS="${BENCHMARK_EXTRA_ARGS---no-stream}"
KV_BYTES_PER_TOKEN="${KV_BYTES_PER_TOKEN:-}"
DRY_RUN="${DRY_RUN:-0}"

LOG_DIR="${BENCHMARK_DIR}/${OUTPUT_DIR}/sglang"
SERVER_PIDS=()
SERVER_BASE_URLS=()
TELEMETRY_PID=""

cd "${BENCHMARK_DIR}"
mkdir -p "${LOG_DIR}"

if [[ ! -d "${SGLANG_ROOT}" ]]; then
  echo "SGLang submodule does not exist: ${SGLANG_ROOT}" >&2
  echo "Add the sglang submodule first, then rerun this script." >&2
  exit 1
fi

if [[ ! -x "${SGLANG_PYTHON}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    SGLANG_PYTHON="$(command -v python3)"
  else
    echo "SGLang Python does not exist: ${SGLANG_PYTHON}" >&2
    exit 1
  fi
fi

if [[ ! -x "${BENCHMARK_PYTHON}" ]]; then
  echo "Benchmark Python does not exist: ${BENCHMARK_PYTHON}" >&2
  echo "Install the benchmark environment first; see the repository README." >&2
  exit 1
fi

if ((DP_REPLICAS <= 0)); then
  echo "DP_REPLICAS must be positive." >&2
  exit 1
fi
if ((TP_SIZE <= 0)); then
  echo "TP_SIZE must be positive." >&2
  exit 1
fi
if [[ "${WARM_SHARED_PREFIX}" != "0" && "${WARM_SHARED_PREFIX}" != "1" ]]; then
  echo "WARM_SHARED_PREFIX must be 0 or 1." >&2
  exit 1
fi
if [[ "${ENABLE_TELEMETRY}" != "0" && "${ENABLE_TELEMETRY}" != "1" ]]; then
  echo "ENABLE_TELEMETRY must be 0 or 1." >&2
  exit 1
fi
if [[ "${ENABLE_METRICS}" != "0" && "${ENABLE_METRICS}" != "1" ]]; then
  echo "ENABLE_METRICS must be 0 or 1." >&2
  exit 1
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "DRY_RUN must be 0 or 1." >&2
  exit 1
fi

if [[ -z "${KV_BYTES_PER_TOKEN}" && -f "${MODEL_PATH}/config.json" ]]; then
  KV_BYTES_PER_TOKEN="$("${BENCHMARK_PYTHON}" - "${MODEL_PATH}/config.json" "${DTYPE}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
dtype_bytes = 4 if sys.argv[2] == "float32" else 2
layers = int(config["num_hidden_layers"])
kv_heads = int(config.get("num_key_value_heads", config["num_attention_heads"]))
head_dim = int(
    config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
)
print(2 * layers * kv_heads * head_dim * dtype_bytes)
PY
)"
fi
KV_BYTES_PER_TOKEN="${KV_BYTES_PER_TOKEN:-0}"

required_gpu_count=$((DP_REPLICAS * TP_SIZE))
gpu_ids_normalized="${GPU_IDS//,/ }"
gpu_id_list=()
if [[ -n "${gpu_ids_normalized}" ]]; then
  read -r -a gpu_id_list <<<"${gpu_ids_normalized}"
else
  for ((rank = 0; rank < required_gpu_count; rank++)); do
    gpu_id_list+=("${rank}")
  done
fi
if ((${#gpu_id_list[@]} < required_gpu_count)); then
  echo "GPU_IDS must provide at least DP_REPLICAS * TP_SIZE entries." >&2
  exit 1
fi

stop_telemetry() {
  if [[ -n "${TELEMETRY_PID}" ]] && kill -0 "${TELEMETRY_PID}" 2>/dev/null; then
    kill -TERM "${TELEMETRY_PID}" 2>/dev/null || true
    wait "${TELEMETRY_PID}" 2>/dev/null || true
  fi
  TELEMETRY_PID=""
}

stop_server() {
  stop_telemetry
  for server_pid in "${SERVER_PIDS[@]}"; do
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
      if [[ "${KEEP_SERVER}" == "1" ]]; then
        echo "SGLang server remains running (PID ${server_pid})."
      else
        echo "Stopping SGLang server (PID ${server_pid})..."
        kill -TERM "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
      fi
    fi
  done
  SERVER_PIDS=()
  SERVER_BASE_URLS=()
}

server_is_ready() {
  local base_url="$1"
  curl --silent --fail --max-time 2 "${base_url}/health" >/dev/null 2>&1 \
    || curl --silent --fail --max-time 2 "${base_url}/v1/models" >/dev/null 2>&1
}

write_prometheus_metrics() {
  for ((rank = 0; rank < DP_REPLICAS; rank++)); do
    local metrics_path="${LOG_DIR}/prometheus_metrics_rank${rank}.prom"
    if ((DP_REPLICAS == 1)); then
      metrics_path="${LOG_DIR}/prometheus_metrics.prom"
    fi
    curl --silent --fail --max-time 10 \
      "${SERVER_BASE_URLS[$rank]}/metrics" >"${metrics_path}" || true
  done
}

write_server_profile() {
  local profile_path="${LOG_DIR}/server_profile.json"
  "${BENCHMARK_PYTHON}" - "${profile_path}" "${MODEL_PATH}" "${SERVED_MODEL_NAME}" \
    "${DP_REPLICAS}" "${TP_SIZE}" "${CONTEXT_LENGTH}" "${MEM_FRACTION_STATIC}" \
    "${MAX_RUNNING_REQUESTS}" "${KV_BYTES_PER_TOKEN}" <<'PY'
import json
import sys

(
    profile_path,
    model_path,
    served_model_name,
    dp_replicas,
    tp_size,
    context_length,
    mem_fraction_static,
    max_running_requests,
    kv_bytes_per_token,
) = sys.argv[1:]
payload = {
    "engine": "sglang",
    "model_path": model_path,
    "served_model_name": served_model_name,
    "dp_replicas": int(dp_replicas),
    "tensor_parallel_size": int(tp_size),
    "context_length": int(context_length),
    "mem_fraction_static": float(mem_fraction_static),
    "max_running_requests": int(max_running_requests),
    "kv_bytes_per_token": int(kv_bytes_per_token),
}
with open(profile_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY
}

trap stop_server EXIT INT TERM

effective_dp_routing="${DP_ROUTING}"
if ((DP_REPLICAS == 1)); then
  effective_dp_routing="single"
fi

for ((rank = 0; rank < DP_REPLICAS; rank++)); do
  rank_port=$((PORT + rank))
  rank_base_url="http://${HOST}:${rank_port}"
  if server_is_ready "${rank_base_url}"; then
    echo "Port ${rank_port} already has a healthy server." >&2
    exit 1
  fi
  SERVER_BASE_URLS+=("${rank_base_url}")
done

for ((rank = 0; rank < DP_REPLICAS; rank++)); do
  rank_port=$((PORT + rank))
  gpu_offset=$((rank * TP_SIZE))
  gpu_id="$(IFS=,; echo "${gpu_id_list[*]:${gpu_offset}:${TP_SIZE}}")"
  server_log="${LOG_DIR}/sglang_server_rank${rank}.log"
  if ((DP_REPLICAS == 1)); then
    server_log="${LOG_DIR}/sglang_server.log"
  fi
  sglang_args=(
    -m sglang.launch_server
    --model-path "${MODEL_PATH}"
    --host "${HOST}"
    --port "${rank_port}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --dtype "${DTYPE}"
    --tp-size "${TP_SIZE}"
    --context-length "${CONTEXT_LENGTH}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
  )
  if [[ "${ENABLE_METRICS}" == "1" ]]; then
    sglang_args+=(--enable-metrics)
  fi
  if [[ -n "${SGLANG_EXTRA_ARGS}" ]]; then
    read -r -a extra_sglang_args <<<"${SGLANG_EXTRA_ARGS}"
    sglang_args+=("${extra_sglang_args[@]}")
  fi

  echo "Starting ${MODEL_PATH} with SGLang server ${rank}"
  echo "  GPUs ${gpu_id}, endpoint ${SERVER_BASE_URLS[$rank]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q %q' \
      "${gpu_id}" "${SGLANG_ROOT}/python" "${SGLANG_PYTHON}"
    printf ' %q' "${sglang_args[@]}"
    printf '\n'
  else
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
      PYTHONPATH="${SGLANG_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" \
      "${SGLANG_PYTHON}" "${sglang_args[@]}" >"${server_log}" 2>&1 &
    SERVER_PIDS+=("$!")
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

for ((rank = 0; rank < DP_REPLICAS; rank++)); do
  server_log="${LOG_DIR}/sglang_server_rank${rank}.log"
  if ((DP_REPLICAS == 1)); then
    server_log="${LOG_DIR}/sglang_server.log"
  fi
  deadline=$((SECONDS + STARTUP_TIMEOUT))
  until server_is_ready "${SERVER_BASE_URLS[$rank]}"; do
    if ! kill -0 "${SERVER_PIDS[$rank]}" 2>/dev/null; then
      echo "SGLang rank ${rank} exited during startup. Last log lines:" >&2
      tail -n 80 "${server_log}" >&2
      exit 1
    fi
    if ((SECONDS >= deadline)); then
      echo "Timed out waiting for SGLang rank ${rank}." >&2
      tail -n 80 "${server_log}" >&2
      exit 1
    fi
    sleep 2
  done
  echo "SGLang rank ${rank} is ready:"
  curl --silent --fail "${SERVER_BASE_URLS[$rank]}/v1/models" || true
  echo
done

base_urls_arg=""
for rank_base_url in "${SERVER_BASE_URLS[@]}"; do
  if [[ -n "${base_urls_arg}" ]]; then
    base_urls_arg+=","
  fi
  base_urls_arg+="${rank_base_url}/v1"
done

benchmark_output_dir="${OUTPUT_DIR}/sglang"
benchmark_args=(
  -m cli run-api
  --dataset "${DATASET}"
  --sample-index "${SAMPLE_INDEX}"
  --sample-count "${SAMPLE_COUNT}"
  --api-mode chat
  --base-url "${SERVER_BASE_URLS[0]}/v1"
  --base-urls "${base_urls_arg}"
  --dp-routing "${effective_dp_routing}"
  --model "${SERVED_MODEL_NAME}"
  --kv-bytes-per-token "${KV_BYTES_PER_TOKEN}"
  --prefix-tokens "${PREFIX_TOKENS}"
  --branches "${BRANCHES}"
  --case-count "${CASE_COUNT}"
  --branch-group-size "${BRANCH_GROUP_SIZE}"
  --branch-order "${BRANCH_ORDER}"
  --suffix-distribution "${SUFFIX_DISTRIBUTION}"
  --suffix-mean "${SUFFIX_MEAN}"
  --output-tokens "${OUTPUT_TOKENS}"
  --common-analysis-tokens "${COMMON_ANALYSIS_TOKENS}"
  --concurrency "${CONCURRENCY}"
  --arrival-interval-ms "${ARRIVAL_INTERVAL_MS}"
  --minority-headstart-ms "${MINORITY_HEADSTART_MS}"
  --seed "${SEED}"
  --output-dir "${benchmark_output_dir}"
)
if [[ "${WARM_SHARED_PREFIX}" == "1" ]]; then
  benchmark_args+=(--warm-shared-prefix)
fi
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

if [[ "${ENABLE_TELEMETRY}" == "1" ]]; then
  internal_gpu_ids="$(IFS=,; echo "${gpu_id_list[*]:0:${required_gpu_count}}")"
  telemetry_args=(
    "${BENCHMARK_DIR}/src/telemetry.py"
    --output "${LOG_DIR}/telemetry.json"
    --gpu-ids "${internal_gpu_ids}"
    --interval-seconds "${TELEMETRY_INTERVAL_SECONDS}"
  )
  if [[ "${ENABLE_METRICS}" == "1" ]]; then
    for rank_base_url in "${SERVER_BASE_URLS[@]}"; do
      telemetry_args+=(--metrics-url "${rank_base_url}/metrics")
    done
  fi
  "${BENCHMARK_PYTHON}" "${telemetry_args[@]}" &
  TELEMETRY_PID="$!"
fi

echo "Running Agentrix benchmark for SGLang..."
OPENAI_API_KEY="sglang-local" "${BENCHMARK_PYTHON}" "${benchmark_args[@]}"

stop_telemetry
write_prometheus_metrics
write_server_profile
stop_server
echo "Benchmark complete for SGLang: ${LOG_DIR}"
