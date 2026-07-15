#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Core 2x2: attention backend x prompt compaction. The final pair measures
# whether eliminating duplicate RAG chunks complements or competes with
# CacheBlend. Every variant gets the same warm-up and a fresh server.
export VARIANTS="${VARIANTS:-baseline baseline_compact forkattention forkattention_compact cacheblend cacheblend_compact}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/../results/langgraph_prompt_compaction_ablation}"

exec "${SCRIPT_DIR}/run_langgraph_fork_cacheblend_20_e2e.sh"
