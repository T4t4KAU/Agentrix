#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="${BENCHMARK_DIR}/scripts/run_vllm_benchmark.sh"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8b-local}"
DATASET="${DATASET:-swebench}"
DATA_PATH="${DATA_PATH:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/dp_full_dataset_$(date +%Y%m%d_%H%M%S)}"
DP_REPLICAS="${DP_REPLICAS:-2}"
GPU_IDS="${GPU_IDS:-0,1}"
CASE_COUNT="${CASE_COUNT:-8}"
BRANCHES="${BRANCHES:-8}"
PREFIX_TOKENS="${PREFIX_TOKENS:-8192}"
CONCURRENCY="${CONCURRENCY:-64}"
SUFFIX_MEAN="${SUFFIX_MEAN:-128}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
RUN_PRESSURE="${RUN_PRESSURE:-1}"
PROFILE_FORK="${PROFILE_FORK:-1}"
CAPTURE_BUCKETS="${CAPTURE_BUCKETS:-common:2,4,8;forest:64,128,256,512}"

run_variant() {
  local name="$1"
  local backend="$2"
  local prefix_routing="$3"
  local kv_transfer_config="${4:-}"
  local gpu_memory_utilization="${5:-${GPU_MEMORY_UTILIZATION}}"
  local output_dir="${OUTPUT_ROOT}/${name}"
  local result_path="${BENCHMARK_DIR}/${output_dir}/${backend,,}/benchmark_results.csv"

  if [[ -s "${result_path}" ]]; then
    echo "Skipping completed variant ${name}: ${result_path}"
    return
  fi

  echo "Running full-dataset variant ${name} (${backend})"
  DATASET="${DATASET}" \
  DATA_PATH="${DATA_PATH}" \
  FULL_DATASET=1 \
  MODEL_PATH="${MODEL_PATH}" \
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
  BACKENDS="${backend}" \
  DP_DEPLOYMENT=internal \
  DP_REPLICAS="${DP_REPLICAS}" \
  DP_ROUTING=single \
  GPU_IDS="${GPU_IDS}" \
  CASE_COUNT="${CASE_COUNT}" \
  SAMPLE_COUNT="${CASE_COUNT}" \
  PREFIX_TOKENS="${PREFIX_TOKENS}" \
  BRANCHES="${BRANCHES}" \
  BRANCH_GROUP_SIZE="${BRANCHES}" \
  BRANCH_ORDER=round_robin \
  CONCURRENCY="${CONCURRENCY}" \
  SUFFIX_DISTRIBUTION=equal \
  SUFFIX_MEAN="${SUFFIX_MEAN}" \
  OUTPUT_TOKENS="${OUTPUT_TOKENS}" \
  COMMON_ANALYSIS_TOKENS=32 \
  MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  GPU_MEMORY_UTILIZATION="${gpu_memory_utilization}" \
  ENFORCE_EAGER=0 \
  PROFILE_FORK="${PROFILE_FORK}" \
  VLLM_FORK_ATTN_ENABLE_FOREST=1 \
  VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH=1 \
  VLLM_FORK_ATTN_FOREST_CTA_BUCKETS=64,128,256,384,512 \
  VLLM_FORK_ATTN_CUDAGRAPH_CAPTURE_BUCKETS="${CAPTURE_BUCKETS}" \
  VLLM_FORK_ATTN_DP_PREFIX_ROUTING="${prefix_routing}" \
  VLLM_FORK_ATTN_DP_GRAPH_SLACK_BUCKETS=0 \
  VLLM_FORK_ATTN_DP_ARRIVAL_WAVE_MS=10 \
  KV_TRANSFER_CONFIG="${kv_transfer_config}" \
  OUTPUT_DIR="${output_dir}" \
  "${RUN_SCRIPT}"
}

run_variant flash_dp FLASH_ATTN 0
run_variant fork_dp FORK_ATTN 0
run_variant fork_optimized_dp FORK_ATTN 1

if [[ "${RUN_PRESSURE}" == "1" ]]; then
  OFFLOAD_CONFIG="${OFFLOAD_CONFIG:-{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":68719476736,\"fanout_offload\":true,\"fanout_profile\":true,\"fanout_budget_blocks\":256,\"fanout_allow_hot_prefix_backup\":true}}}"
  run_variant fork_optimized_dp_offload FORK_ATTN 1 "${OFFLOAD_CONFIG}" 0.60
fi

"${BENCHMARK_DIR}/.venv/bin/python" - \
  "${BENCHMARK_DIR}/${OUTPUT_ROOT}" <<'PY'
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
variants = {
    "flash_dp": "flash_attn",
    "fork_dp": "fork_attn",
    "fork_optimized_dp": "fork_attn",
    "fork_optimized_dp_offload": "fork_attn",
}
metrics = {
    "case_wall_time_ms": "E2E wall ms",
    "branch_phase_wall_ms": "Branch wall ms",
    "end_to_end_output_tokens_per_s": "E2E tok/s",
    "branch_output_tokens_per_s": "Branch tok/s",
}
rows = []
for variant, backend in variants.items():
    result_path = root / variant / backend / "benchmark_results.csv"
    if not result_path.exists():
        continue
    with result_path.open(encoding="utf-8", newline="") as handle:
        results = list(csv.DictReader(handle))
    row = {"variant": variant, "batches": len(results), "samples": 0}
    for key in metrics:
        values = [float(result[key]) for result in results]
        row[key] = statistics.fmean(values)
    profile_path = root / variant / backend / "server_profile.json"
    if profile_path.exists():
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        row["samples"] = profile.get("workload", {}).get("sample_count", 0)
        route = profile.get("ranks", [{}])[0].get("fork_dp_prefix_routing", {})
        row["routing"] = route
    rows.append(row)

report = root / "full_dataset_comparison.md"
with report.open("w", encoding="utf-8") as handle:
    handle.write("# Full Dataset DP Comparison\n\n")
    handle.write(
        "| Variant | Samples | Batches | " + " | ".join(metrics.values()) + " |\n"
    )
    handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for row in rows:
        values = " | ".join(f"{row[key]:.2f}" for key in metrics)
        handle.write(
            f"| {row['variant']} | {row['samples']} | {row['batches']} | "
            f"{values} |\n"
        )
    by_variant = {row["variant"]: row for row in rows}
    optimized = by_variant.get("fork_optimized_dp")
    if optimized:
        handle.write("\n## Optimized DP Delta\n\n")
        handle.write("| Baseline | E2E latency | Branch latency | E2E throughput | Branch throughput |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for baseline_name in ("flash_dp", "fork_dp"):
            baseline = by_variant.get(baseline_name)
            if not baseline:
                continue
            e2e_latency = (
                1 - optimized["case_wall_time_ms"] / baseline["case_wall_time_ms"]
            ) * 100
            branch_latency = (
                1
                - optimized["branch_phase_wall_ms"]
                / baseline["branch_phase_wall_ms"]
            ) * 100
            e2e_throughput = (
                optimized["end_to_end_output_tokens_per_s"]
                / baseline["end_to_end_output_tokens_per_s"]
                - 1
            ) * 100
            branch_throughput = (
                optimized["branch_output_tokens_per_s"]
                / baseline["branch_output_tokens_per_s"]
                - 1
            ) * 100
            handle.write(
                f"| {baseline_name} | {e2e_latency:+.2f}% | "
                f"{branch_latency:+.2f}% | {e2e_throughput:+.2f}% | "
                f"{branch_throughput:+.2f}% |\n"
            )
    handle.write("\n## Routing\n\n")
    for row in rows:
        if row.get("routing"):
            handle.write(f"- `{row['variant']}`: `{row['routing']}`\n")
print(f"Wrote {report}")
PY

echo "Full dataset DP benchmark complete: ${BENCHMARK_DIR}/${OUTPUT_ROOT}"
