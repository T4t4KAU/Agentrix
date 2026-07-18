#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
PREFIXES="${PREFIXES:-4096,8192,16384,32768,65536}"
QUERY_COUNTS="${QUERY_COUNTS:-2,4,8,16,32}"
MATRIX_DIR="${MATRIX_DIR:-${BENCHMARK_DIR}/results/ncu_forkattention_matrix}"

IFS=',' read -r -a prefixes <<<"${PREFIXES}"
IFS=',' read -r -a query_counts <<<"${QUERY_COUNTS}"
for prefix in "${prefixes[@]}"; do
  for queries in "${query_counts[@]}"; do
    cell="${MATRIX_DIR}/p${prefix}_b${queries}"
    if [[ -s "${cell}/flash_attn_summary.json" \
      && -s "${cell}/fork_attn_summary.json" ]]; then
      echo "Skipping complete matrix cell p${prefix}_b${queries}"
      continue
    fi
    echo "Running matrix cell p${prefix}_b${queries}"
    PREFIX_TOKENS="${prefix}" \
    BRANCHES="${queries}" \
    PREFIX_CHUNK_TOKENS=0 \
    OUTPUT_DIR="${cell}" \
      "${BENCHMARK_DIR}/scripts/run_fork_attention_ncu.sh"
  done
done

"${REPO_ROOT}/vllm/.venv/bin/python" \
  "${BENCHMARK_DIR}/scripts/summarize_fork_attention_ncu_matrix.py" \
  "${MATRIX_DIR}" \
  --output "${MATRIX_DIR}/matrix.csv"
