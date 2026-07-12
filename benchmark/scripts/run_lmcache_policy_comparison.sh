#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}"
BASE_OUTPUT_DIR="${OUTPUT_DIR:-results/lmcache_policy_comparison}"
BASELINE_POLICY="LRU"
OPTIMIZED_POLICY="${OPTIMIZED_POLICY:-FORK_AWARE}"

run_policy() {
  local policy="$1"
  local name="$2"
  LMCACHE_CACHE_POLICY="${policy}" \
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${name}" \
    "${BENCHMARK_DIR}/scripts/run_lmcache_tiered_smoke.sh"
}

# LRU is LMCache's default policy and is always measured first.
run_policy "${BASELINE_POLICY}" baseline
run_policy "${OPTIMIZED_POLICY}" optimized

"${BENCHMARK_PYTHON}" \
  "${BENCHMARK_DIR}/src/lmcache_policy_report.py" \
  "${BENCHMARK_DIR}/${BASE_OUTPUT_DIR}" \
  "${BASELINE_POLICY}" "${OPTIMIZED_POLICY}"
