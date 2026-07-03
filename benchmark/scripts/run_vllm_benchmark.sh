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
BACKENDS="${BACKENDS:-FLASH_ATTN FORK_ATTN}"
DTYPE="${DTYPE:-float16}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9000}"
DATASET="${DATASET:-swebench}"
DATA_PATH="${DATA_PATH:-}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
PREFIX_TOKENS="${PREFIX_TOKENS:-2048}"
BRANCHES="${BRANCHES:-2}"
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
OUTPUT_DIR="${OUTPUT_DIR:-results/fork_vs_flash_${DATASET}_p${PREFIX_TOKENS}_b${BRANCHES}_o${OUTPUT_TOKENS}}"
KEEP_SERVER="${KEEP_SERVER:-0}"
VLLM_SERVER_EXTRA_ARGS="${VLLM_SERVER_EXTRA_ARGS:-}"
BENCHMARK_EXTRA_ARGS="${BENCHMARK_EXTRA_ARGS:-}"

LOG_DIR="${BENCHMARK_DIR}/${OUTPUT_DIR}"
BASE_URL="http://${HOST}:${PORT}"
SERVER_PID=""
BACKENDS="${BACKENDS//,/ }"
read -r -a BACKEND_LIST <<<"${BACKENDS}"

cd "${BENCHMARK_DIR}"
mkdir -p "${LOG_DIR}"

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

if ((${#BACKEND_LIST[@]} < 2)); then
  echo "BACKENDS must contain at least two attention backends to compare." >&2
  echo "Example: BACKENDS=\"FLASH_ATTN FORK_ATTN\" $0" >&2
  exit 1
fi

if [[ "${KEEP_SERVER}" == "1" ]] && ((${#BACKEND_LIST[@]} > 1)); then
  echo "KEEP_SERVER=1 is only supported for single-backend debugging." >&2
  exit 1
fi

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    if [[ "${KEEP_SERVER}" == "1" ]]; then
      echo "vLLM server remains running (PID ${SERVER_PID})."
    else
      echo "Stopping vLLM server (PID ${SERVER_PID})..."
      kill -TERM "${SERVER_PID}" 2>/dev/null || true
      wait "${SERVER_PID}" 2>/dev/null || true
    fi
  fi
  SERVER_PID=""
}

trap stop_server EXIT INT TERM

run_backend() {
  local attention_backend="$1"
  local backend_name="${attention_backend,,}"
  local backend_output_dir="${OUTPUT_DIR}/${backend_name}"
  local backend_log_dir="${BENCHMARK_DIR}/${backend_output_dir}"
  local server_log="${backend_log_dir}/vllm_server.log"

  mkdir -p "${backend_log_dir}"

  if curl --silent --fail --max-time 2 "${BASE_URL}/health" >/dev/null 2>&1; then
    echo "Port ${PORT} already has a healthy server; refusing to replace it." >&2
    exit 1
  fi

  local vllm_args=(
    serve "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT}"
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
  if [[ -n "${VLLM_SERVER_EXTRA_ARGS}" ]]; then
    read -r -a extra_vllm_args <<<"${VLLM_SERVER_EXTRA_ARGS}"
    vllm_args+=("${extra_vllm_args[@]}")
  fi

  echo "Starting ${MODEL_PATH} with ${attention_backend} on ${BASE_URL}..."
  "${VLLM_BIN}" "${vllm_args[@]}" >"${server_log}" 2>&1 &
  SERVER_PID=$!

  local deadline=$((SECONDS + STARTUP_TIMEOUT))
  until curl --silent --fail --max-time 2 "${BASE_URL}/health" >/dev/null 2>&1; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "vLLM exited during startup. Last log lines:" >&2
      tail -n 80 "${server_log}" >&2
      exit 1
    fi
    if ((SECONDS >= deadline)); then
      echo "Timed out waiting for vLLM after ${STARTUP_TIMEOUT}s." >&2
      tail -n 80 "${server_log}" >&2
      exit 1
    fi
    sleep 2
  done

  echo "vLLM is ready; available models:"
  curl --silent --fail "${BASE_URL}/v1/models"
  echo

  local benchmark_args=(
    -m cli run-api
    --dataset "${DATASET}"
    --sample-index "${SAMPLE_INDEX}"
    --api-mode chat
    --base-url "${BASE_URL}/v1"
    --model "${SERVED_MODEL_NAME}"
    --prefix-tokens "${PREFIX_TOKENS}"
    --branches "${BRANCHES}"
    --suffix-distribution "${SUFFIX_DISTRIBUTION}"
    --suffix-mean "${SUFFIX_MEAN}"
    --output-tokens "${OUTPUT_TOKENS}"
    --common-analysis-tokens "${COMMON_ANALYSIS_TOKENS}"
    --concurrency "${BRANCHES}"
    --arrival-interval-ms "${ARRIVAL_INTERVAL_MS}"
    --seed "${SEED}"
    --output-dir "${backend_output_dir}"
  )
  if [[ -n "${DATA_PATH}" ]]; then
    benchmark_args+=(--data-path "${DATA_PATH}")
  fi
  if [[ -n "${BENCHMARK_EXTRA_ARGS}" ]]; then
    read -r -a extra_benchmark_args <<<"${BENCHMARK_EXTRA_ARGS}"
    benchmark_args+=("${extra_benchmark_args[@]}")
  fi

  echo "Running Agentrix benchmark for ${attention_backend}..."
  OPENAI_API_KEY="vllm-local" "${BENCHMARK_PYTHON}" "${benchmark_args[@]}"

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
    if len(rows) != 1:
        raise SystemExit(f"expected one result row in {path}, found {len(rows)}")
    return rows[0]


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

write_comparison

echo "Comparison complete: ${BENCHMARK_DIR}/${OUTPUT_DIR}"
