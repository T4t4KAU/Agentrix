#!/usr/bin/env bash
set -Eeuo pipefail

# Environment variables can override every expensive benchmark parameter.
BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
VLLM_BIN="${VLLM_BIN:-${REPO_ROOT}/vllm/.venv/bin/vllm}"
BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}"
export PATH="$(dirname "${VLLM_BIN}"):${PATH}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b-local}"
BACKENDS="${BACKENDS:-FORK_ATTN}"
DTYPE="${DTYPE:-float16}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9000}"
DP_REPLICAS="${DP_REPLICAS:-1}"
DP_DEPLOYMENT="${DP_DEPLOYMENT:-external}"
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
SEED="${SEED:-2026}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
OUTPUT_DIR="${OUTPUT_DIR:-results/fork_attention_${DATASET}_c${CASE_COUNT}_p${PREFIX_TOKENS}_b${BRANCHES}_g${BRANCH_GROUP_SIZE}_o${OUTPUT_TOKENS}}"
KEEP_SERVER="${KEEP_SERVER:-0}"
VLLM_SERVER_EXTRA_ARGS="${VLLM_SERVER_EXTRA_ARGS:-}"
BENCHMARK_EXTRA_ARGS="${BENCHMARK_EXTRA_ARGS:-}"
KV_TRANSFER_CONFIG="${KV_TRANSFER_CONFIG:-}"
KV_BYTES_PER_TOKEN="${KV_BYTES_PER_TOKEN:-}"

LOG_DIR="${BENCHMARK_DIR}/${OUTPUT_DIR}"
SERVER_PIDS=()
SERVER_BASE_URLS=()
BACKENDS="${BACKENDS//,/ }"
read -r -a BACKEND_LIST <<<"${BACKENDS}"

cd "${BENCHMARK_DIR}"
mkdir -p "${LOG_DIR}"

if [[ -z "${KV_BYTES_PER_TOKEN}" && -f "${MODEL_PATH}/config.json" ]]; then
  KV_BYTES_PER_TOKEN="$(${BENCHMARK_PYTHON} - "${MODEL_PATH}/config.json" "${DTYPE}" <<'PY'
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

if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "vLLM executable does not exist: ${VLLM_BIN}" >&2
  echo "Build the vllm submodule first; see the repository README." >&2
  exit 1
fi

if [[ ! -x "${BENCHMARK_PYTHON}" ]]; then
  echo "Benchmark Python does not exist: ${BENCHMARK_PYTHON}" >&2
  echo "Install the benchmark environment first; see the repository README." >&2
  exit 1
fi

if ((${#BACKEND_LIST[@]} == 0)); then
  echo "BACKENDS must contain at least one attention backend." >&2
  echo "Example: BACKENDS=\"FORK_ATTN\" $0" >&2
  exit 1
fi

if ((DP_REPLICAS <= 0)); then
  echo "DP_REPLICAS must be positive." >&2
  exit 1
fi

if [[ "${DP_DEPLOYMENT}" != "external" && "${DP_DEPLOYMENT}" != "internal" ]]; then
  echo "DP_DEPLOYMENT must be external or internal." >&2
  exit 1
fi

SERVER_COUNT="${DP_REPLICAS}"
if [[ "${DP_DEPLOYMENT}" == "internal" ]]; then
  SERVER_COUNT=1
fi

if [[ "${KEEP_SERVER}" == "1" ]] && ((${#BACKEND_LIST[@]} > 1)); then
  echo "KEEP_SERVER=1 is only supported for single-backend debugging." >&2
  exit 1
fi

stop_server() {
  for server_pid in "${SERVER_PIDS[@]}"; do
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
      if [[ "${KEEP_SERVER}" == "1" ]]; then
        echo "vLLM server remains running (PID ${server_pid})."
      else
        echo "Stopping vLLM server (PID ${server_pid})..."
        kill -TERM "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
      fi
    fi
  done
  SERVER_PIDS=()
  SERVER_BASE_URLS=()
}

write_prometheus_metrics() {
  local backend_log_dir="$1"
  for ((rank = 0; rank < SERVER_COUNT; rank++)); do
    local metrics_path="${backend_log_dir}/prometheus_metrics_rank${rank}.prom"
    if ((SERVER_COUNT == 1)); then
      metrics_path="${backend_log_dir}/prometheus_metrics.prom"
    fi
    curl --silent --fail --max-time 10 \
      "${SERVER_BASE_URLS[$rank]}/metrics" >"${metrics_path}" || true
  done
}

write_server_profile() {
  local backend_log_dir="$1"
  local profile_path="${backend_log_dir}/server_profile.json"
  local log_paths=()
  if ((SERVER_COUNT == 1)); then
    log_paths+=("${backend_log_dir}/vllm_server.log")
  else
    for ((rank = 0; rank < SERVER_COUNT; rank++)); do
      log_paths+=("${backend_log_dir}/vllm_server_rank${rank}.log")
    done
  fi

  "${BENCHMARK_PYTHON}" - \
    "${profile_path}" \
    "${CASE_COUNT}" \
    "${SAMPLE_COUNT}" \
    "${BRANCHES}" \
    "${CONCURRENCY}" \
    "${PREFIX_TOKENS}" \
    "${BRANCH_ORDER}" \
    "${SUFFIX_DISTRIBUTION}" \
    "${log_paths[@]}" <<'PY'
from __future__ import annotations

import ast
import csv
import json
import re
import sys
from pathlib import Path


PROFILE_KEYS = [
    "candidates",
    "hot_shared_candidates",
    "selected_chunks",
    "selected_blocks",
    "protected_hot_shared_chunks",
    "protected_hot_shared_blocks",
    "selected_hot_shared_chunks",
    "selected_hot_shared_blocks",
    "lifecycle_hot_candidates",
    "lifecycle_cooling_candidates",
    "lifecycle_cold_candidates",
    "gpu_hotset_reserved_blocks",
]


def int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value.replace(",", ""))


def max_field(lines: list[str], key: str) -> int:
    pattern = re.compile(rf"{re.escape(key)}=(\d+)")
    values = [int(match.group(1)) for line in lines for match in pattern.finditer(line)]
    return max(values, default=0)


def prometheus_counter(path: Path, metric: str) -> float:
    if not path.exists():
        return 0.0
    pattern = re.compile(
        rf"^{re.escape(metric)}(?:_total)?(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$",
        re.MULTILINE,
    )
    text = path.read_text(encoding="utf-8", errors="ignore")
    return sum(float(value) for value in pattern.findall(text))


def parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    kv_match = re.search(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", text)
    concurrency_match = re.search(
        r"Maximum concurrency for\s*([0-9,]+)\s*tokens per request:\s*([0-9.]+)x",
        text,
    )
    profile = {f"max_{key}": max_field(lines, key) for key in PROFILE_KEYS}
    usages = [
        float(value)
        for value in re.findall(r"kv_cache_usage=([0-9.]+)", text)
    ]
    pressure_steps = {
        level: text.count(f"pressure={level}")
        for level in ("normal", "high", "critical")
    }
    metrics_name = (
        "prometheus_metrics.prom"
        if path.name == "vllm_server.log"
        else path.name.replace("vllm_server_rank", "prometheus_metrics_rank").replace(
            ".log", ".prom"
        )
    )
    metrics_path = path.with_name(metrics_name)
    dispatch_matches = re.findall(
        r"ForkAttention CUDA graph dispatch: total=\d+ counters=(\{[^\n]+\})",
        text,
    )
    graph_dispatch = (
        ast.literal_eval(dispatch_matches[-1]) if dispatch_matches else {}
    )
    metadata_averages: dict[str, float] = {}
    for match in re.finditer(
        r"ForkAttention profile:.*?path=([^ ]+).*?avg_metadata_ms=([0-9.]+)",
        text,
    ):
        metadata_averages[match.group(1)] = float(match.group(2))
    routing_matches = re.findall(
        r"ForkAttention DP prefix routing stats: requests=(\d+), "
        r"affinity_routes=(\d+), graph_bound_routes=(\d+), "
        r"arrival_waves=(\d+), "
        r"rank_routes=(\[[^\n]+?\]), "
        r"avg_route_us=([0-9.]+)",
        text,
    )
    routing_stats: dict[str, object] = {}
    if routing_matches:
        (
            requests,
            affinity_routes,
            graph_bound_routes,
            arrival_waves,
            rank_routes,
            avg_route_us,
        ) = routing_matches[-1]
        routing_stats = {
            "requests": int(requests),
            "affinity_routes": int(affinity_routes),
            "graph_bound_routes": int(graph_bound_routes),
            "arrival_waves": int(arrival_waves),
            "rank_routes": ast.literal_eval(rank_routes),
            "avg_route_us": float(avg_route_us),
        }
        telemetry_matches = re.findall(
            r"ForkAttention DP prefix routing stats:.*?telemetry=(\[[^\n]+\])",
            text,
        )
        if telemetry_matches:
            routing_stats["telemetry"] = ast.literal_eval(telemetry_matches[-1])
    return {
        "log": str(path),
        "gpu_kv_cache_tokens": int_or_none(kv_match.group(1) if kv_match else None),
        "max_model_len_for_concurrency": int_or_none(
            concurrency_match.group(1) if concurrency_match else None
        ),
        "max_concurrency": (
            float(concurrency_match.group(2)) if concurrency_match else None
        ),
        "fanout_preemption_selected_victims": text.count(
            "Fanout preemption selected victim"
        ),
        "fanout_admission_promoted_cohorts": text.count(
            "Fanout admission promoted cohort"
        ),
        "fanout_gpu_hotset_reservations": text.count(
            "Fanout GPU hotset reserved"
        ),
        "preempt_log_lines": len(re.findall("preempt", text, re.IGNORECASE)),
        "max_observed_kv_cache_usage": max(usages, default=0.0),
        "pressure_steps": pressure_steps,
        "kv_offload_load_bytes": prometheus_counter(
            metrics_path, "vllm:kv_offload_load_bytes"
        ),
        "kv_offload_store_bytes": prometheus_counter(
            metrics_path, "vllm:kv_offload_store_bytes"
        ),
        "fork_cudagraph_dispatch": graph_dispatch,
        "fork_metadata_avg_ms": metadata_averages,
        "fork_dp_prefix_routing": routing_stats,
        **profile,
    }


def main() -> int:
    profile_path = Path(sys.argv[1])
    result_path = profile_path.parent / "benchmark_results.csv"
    actual_sample_count = int(sys.argv[3])
    batch_count = None
    if result_path.exists():
        with result_path.open(encoding="utf-8", newline="") as handle:
            result_rows = list(csv.DictReader(handle))
        batch_count = len(result_rows)
        branches = int(sys.argv[4])
        actual_sample_count = sum(
            int(row["branch_count"]) // branches for row in result_rows
        )
    workload = {
        "case_count": int(sys.argv[2]),
        "sample_count": actual_sample_count,
        "batch_count": batch_count,
        "branches": int(sys.argv[4]),
        "concurrency": int(sys.argv[5]),
        "prefix_tokens": int(sys.argv[6]),
        "branch_order": sys.argv[7],
        "suffix_distribution": sys.argv[8],
    }
    logs = [Path(arg) for arg in sys.argv[9:]]
    ranks = [parse_log(path) for path in logs if path.exists()]
    aggregate: dict[str, object] = {"workload": workload, "ranks": ranks}
    for key in (
        "gpu_kv_cache_tokens",
        "fanout_preemption_selected_victims",
        "fanout_admission_promoted_cohorts",
        "fanout_gpu_hotset_reservations",
        "preempt_log_lines",
        *(f"max_{name}" for name in PROFILE_KEYS),
    ):
        values = [rank.get(key) for rank in ranks if isinstance(rank.get(key), int)]
        aggregate[key] = (
            sum(values)
            if key.endswith(("victims", "cohorts", "reservations", "lines"))
            else max(values, default=0)
        )
    aggregate["max_observed_kv_cache_usage"] = max(
        (
            float(rank.get("max_observed_kv_cache_usage", 0.0))
            for rank in ranks
        ),
        default=0.0,
    )
    aggregate["pressure_steps"] = {
        level: sum(
            int(rank.get("pressure_steps", {}).get(level, 0))
            for rank in ranks
            if isinstance(rank.get("pressure_steps"), dict)
        )
        for level in ("normal", "high", "critical")
    }
    for key in ("kv_offload_load_bytes", "kv_offload_store_bytes"):
        aggregate[key] = sum(float(rank.get(key, 0.0)) for rank in ranks)
    profile_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
}

trap stop_server EXIT INT TERM

run_backend() {
  local attention_backend="$1"
  local backend_name="${attention_backend,,}"
  local backend_output_dir="${OUTPUT_DIR}/${backend_name}"
  local backend_log_dir="${BENCHMARK_DIR}/${backend_output_dir}"
  local effective_dp_routing="${DP_ROUTING}"

  mkdir -p "${backend_log_dir}"
  if ((DP_REPLICAS == 1)) || [[ "${DP_DEPLOYMENT}" == "internal" ]]; then
    effective_dp_routing="single"
  fi

  SERVER_PIDS=()
  SERVER_BASE_URLS=()
  local gpu_ids_normalized="${GPU_IDS//,/ }"
  local gpu_id_list=()
  if [[ -n "${gpu_ids_normalized}" ]]; then
    read -r -a gpu_id_list <<<"${gpu_ids_normalized}"
  else
    for ((rank = 0; rank < DP_REPLICAS; rank++)); do
      gpu_id_list+=("${rank}")
    done
  fi
  if ((${#gpu_id_list[@]} < DP_REPLICAS)); then
    echo "GPU_IDS must provide at least DP_REPLICAS entries." >&2
    exit 1
  fi
  local internal_gpu_ids
  internal_gpu_ids="$(IFS=,; echo "${gpu_id_list[*]:0:${DP_REPLICAS}}")"

  for ((rank = 0; rank < SERVER_COUNT; rank++)); do
    local rank_port=$((PORT + rank))
    local rank_base_url="http://${HOST}:${rank_port}"
    if curl --silent --fail --max-time 2 \
      "${rank_base_url}/health" >/dev/null 2>&1; then
      echo "Port ${rank_port} already has a healthy server." >&2
      exit 1
    fi
    SERVER_BASE_URLS+=("${rank_base_url}")
  done

  for ((rank = 0; rank < SERVER_COUNT; rank++)); do
    local rank_port=$((PORT + rank))
    local rank_base_url="${SERVER_BASE_URLS[$rank]}"
    local gpu_id="${gpu_id_list[$rank]}"
    if [[ "${DP_DEPLOYMENT}" == "internal" ]]; then
      gpu_id="${internal_gpu_ids}"
    fi
    local server_log="${backend_log_dir}/vllm_server_rank${rank}.log"
    if ((SERVER_COUNT == 1)); then
      server_log="${backend_log_dir}/vllm_server.log"
    fi
    local vllm_args=(
      serve "${MODEL_PATH}"
      --host "${HOST}"
      --port "${rank_port}"
      --served-model-name "${SERVED_MODEL_NAME}"
      --attention-backend "${attention_backend}"
      --dtype "${DTYPE}"
      --generation-config vllm
      --enable-prefix-caching
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
      --max-model-len "${MAX_MODEL_LEN}"
      --max-num-seqs "${MAX_NUM_SEQS}"
    )
    if [[ "${ENFORCE_EAGER}" == "1" ]]; then
      vllm_args+=(--enforce-eager)
    fi
    if [[ "${DP_DEPLOYMENT}" == "internal" ]]; then
      vllm_args+=(
        --data-parallel-size "${DP_REPLICAS}"
        --api-server-count 1
      )
    fi
    if [[ -n "${KV_TRANSFER_CONFIG}" ]]; then
      vllm_args+=(--kv-transfer-config "${KV_TRANSFER_CONFIG}")
    fi
    if [[ -n "${VLLM_SERVER_EXTRA_ARGS}" ]]; then
      read -r -a extra_vllm_args <<<"${VLLM_SERVER_EXTRA_ARGS}"
      vllm_args+=("${extra_vllm_args[@]}")
    fi

    echo "Starting ${MODEL_PATH} with ${attention_backend} server ${rank}"
    echo "  GPUs ${gpu_id}, endpoint ${rank_base_url}, DP ${DP_DEPLOYMENT}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
      "${VLLM_BIN}" "${vllm_args[@]}" >"${server_log}" 2>&1 &
    SERVER_PIDS+=("$!")
  done

  for ((rank = 0; rank < SERVER_COUNT; rank++)); do
    local rank_base_url="${SERVER_BASE_URLS[$rank]}"
    local server_log="${backend_log_dir}/vllm_server_rank${rank}.log"
    if ((SERVER_COUNT == 1)); then
      server_log="${backend_log_dir}/vllm_server.log"
    fi
    local deadline=$((SECONDS + STARTUP_TIMEOUT))
    until curl --silent --fail --max-time 2 \
      "${rank_base_url}/health" >/dev/null 2>&1; do
      if ! kill -0 "${SERVER_PIDS[$rank]}" 2>/dev/null; then
        echo "vLLM rank ${rank} exited during startup. Last log lines:" >&2
        tail -n 80 "${server_log}" >&2
        exit 1
      fi
      if ((SECONDS >= deadline)); then
        echo "Timed out waiting for vLLM rank ${rank}." >&2
        tail -n 80 "${server_log}" >&2
        exit 1
      fi
      sleep 2
    done
    echo "vLLM rank ${rank} is ready:"
    curl --silent --fail "${rank_base_url}/v1/models"
    echo
  done

  local base_urls_arg=""
  for rank_base_url in "${SERVER_BASE_URLS[@]}"; do
    if [[ -n "${base_urls_arg}" ]]; then
      base_urls_arg+=","
    fi
    base_urls_arg+="${rank_base_url}/v1"
  done

  local benchmark_args=(
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
    --seed "${SEED}"
    --output-dir "${backend_output_dir}"
  )
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

  echo "Running Agentrix benchmark for ${attention_backend}..."
  OPENAI_API_KEY="vllm-local" "${BENCHMARK_PYTHON}" "${benchmark_args[@]}"

  write_prometheus_metrics "${backend_log_dir}"
  write_server_profile "${backend_log_dir}"
  stop_server
  echo "Benchmark complete for ${attention_backend}: ${backend_log_dir}"
}

write_comparison() {
  local comparison_dir="${BENCHMARK_DIR}/${OUTPUT_DIR}"
  local comparison_csv="${comparison_dir}/backend_comparison.csv"
  local comparison_md="${comparison_dir}/backend_comparison.md"
  local args=("${comparison_csv}" "${comparison_md}")

  for attention_backend in "${BACKEND_LIST[@]}"; do
    local backend_name="${attention_backend,,}"
    args+=("${attention_backend}" "${comparison_dir}/${backend_name}/benchmark_results.csv")
  done

  "${BENCHMARK_PYTHON}" - "${args[@]}" <<'PY'
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path


LOWER_IS_BETTER = {
    "case_wall_time_ms",
    "branch_phase_wall_ms",
    "branch_mean_latency_ms",
    "branch_median_latency_ms",
    "branch_max_latency_ms",
    "common_latency_ms",
}
HIGHER_IS_BETTER = {
    "end_to_end_output_tokens_per_s",
    "branch_output_tokens_per_s",
}
METRICS = [
    "case_wall_time_ms",
    "end_to_end_output_tokens_per_s",
    "branch_phase_wall_ms",
    "branch_output_tokens_per_s",
    "branch_mean_latency_ms",
    "branch_median_latency_ms",
    "branch_max_latency_ms",
    "common_latency_ms",
]


def read_result(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"expected at least one result row in {path}")
    if len(rows) == 1:
        return rows[0]
    return aggregate_results(rows)


def aggregate_results(rows: list[dict[str, str]]) -> dict[str, str]:
    total_output_tokens = sum(float(row["branch_total_output_tokens"]) for row in rows)
    total_case_ms = sum(float(row["case_wall_time_ms"]) for row in rows)
    total_branch_ms = sum(float(row["branch_phase_wall_ms"]) for row in rows)

    def weighted_mean(metric: str) -> float:
        total_weight = 0.0
        weighted_sum = 0.0
        for row in rows:
            weight = float(row.get("branch_count") or 1)
            weighted_sum += float(row[metric]) * weight
            total_weight += weight
        return weighted_sum / total_weight

    aggregated = rows[0].copy()
    aggregated["case_id"] = f"aggregate_{len(rows)}"
    aggregated["case_wall_time_ms"] = f"{total_case_ms}"
    aggregated["branch_phase_wall_ms"] = f"{total_branch_ms}"
    aggregated["branch_total_output_tokens"] = f"{total_output_tokens}"
    aggregated["end_to_end_output_tokens_per_s"] = (
        f"{total_output_tokens / (total_case_ms / 1000)}"
    )
    aggregated["branch_output_tokens_per_s"] = (
        f"{total_output_tokens / (total_branch_ms / 1000)}"
    )
    for metric in (
        "branch_mean_latency_ms",
        "branch_median_latency_ms",
        "branch_min_latency_ms",
        "branch_max_latency_ms",
        "common_latency_ms",
    ):
        values = [float(row[metric]) for row in rows if row.get(metric)]
        if not values:
            continue
        if metric == "branch_max_latency_ms":
            value = max(values)
        elif metric == "branch_min_latency_ms":
            value = min(values)
        elif metric in {"branch_mean_latency_ms", "branch_median_latency_ms"}:
            value = weighted_mean(metric)
        elif metric == "common_latency_ms":
            value = sum(values)
        else:
            value = statistics.fmean(values)
        aggregated[metric] = f"{value}"
    return aggregated


def to_float(row: dict[str, str], metric: str) -> float | None:
    value = row.get(metric)
    if value in (None, ""):
        return None
    return float(value)


def format_value(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}%"


def main() -> int:
    if len(sys.argv) < 6 or len(sys.argv[3:]) % 2:
        raise SystemExit(
            "usage: python - <comparison.csv> <comparison.md> "
            "<backend> <result.csv> [<backend> <result.csv> ...]"
        )

    comparison_csv = Path(sys.argv[1])
    comparison_md = Path(sys.argv[2])
    pairs = list(zip(sys.argv[3::2], sys.argv[4::2], strict=True))
    results = [(name, read_result(Path(path))) for name, path in pairs]
    baseline_name, baseline = results[0]

    rows: list[dict[str, str]] = []
    for candidate_name, candidate in results[1:]:
        for metric in METRICS:
            baseline_value = to_float(baseline, metric)
            candidate_value = to_float(candidate, metric)
            if baseline_value in (None, 0.0) or candidate_value is None:
                delta_pct = None
                speedup = None
            else:
                delta_pct = (candidate_value - baseline_value) / baseline_value * 100.0
                if metric in LOWER_IS_BETTER:
                    speedup = baseline_value / candidate_value
                elif metric in HIGHER_IS_BETTER:
                    speedup = candidate_value / baseline_value
                else:
                    speedup = None
            rows.append(
                {
                    "baseline_backend": baseline_name,
                    "candidate_backend": candidate_name,
                    "metric": metric,
                    "baseline": format_value(baseline_value),
                    "candidate": format_value(candidate_value),
                    "delta_pct": format_percent(delta_pct),
                    "candidate_speedup": format_value(speedup),
                }
            )

    comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    with comparison_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# vLLM Backend End-to-End Comparison",
        "",
        f"Baseline: `{baseline_name}`",
        "",
        "| Candidate | Metric | Baseline | Candidate | Delta | Speedup |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['candidate_backend']}` | `{row['metric']}` "
            f"| {row['baseline']} | {row['candidate']} "
            f"| {row['delta_pct']} | {row['candidate_speedup']}x |"
        )
    lines.extend(
        [
            "",
            "Latency speedup is `baseline / candidate`; throughput speedup is "
            "`candidate / baseline`. End-to-end metrics include server-side "
            "generation as observed through the OpenAI-compatible API client.",
            "",
        ]
    )
    comparison_md.write_text("\n".join(lines), encoding="utf-8")
    print(comparison_md.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
}

for attention_backend in "${BACKEND_LIST[@]}"; do
  run_backend "${attention_backend}"
done

if ((${#BACKEND_LIST[@]} > 1)); then
  write_comparison

  echo "Comparison complete: ${BENCHMARK_DIR}/${OUTPUT_DIR}"
else
  echo "Benchmark complete: ${BENCHMARK_DIR}/${OUTPUT_DIR}"
fi
