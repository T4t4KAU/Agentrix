#!/usr/bin/env bash
set -Eeuo pipefail

# Capture matched Flash/Cascade/Fork counters while preserving every raw CSV
# and the UI-openable .ncu-rep report emitted by the underlying runner.

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

BACKENDS="${BACKENDS:-FLASH_ATTN,CASCADE_ATTN,FORK_ATTN}" \
  "${BENCHMARK_DIR}/scripts/run_fork_attention_ncu.sh"
