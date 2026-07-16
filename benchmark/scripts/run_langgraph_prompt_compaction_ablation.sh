#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ENABLE_CACHEBLEND="${ENABLE_CACHEBLEND:-0}"

# The default is the core 2x2 attention-backend x prompt-compaction matrix.
# CacheBlend is opt-in because it regressed this host workload; when enabled,
# its final pair measures whether compaction complements or competes with it.
if [[ -z "${VARIANTS+x}" ]]; then
  if [[ "${ENABLE_CACHEBLEND}" == "1" ]]; then
    export VARIANTS="baseline baseline_compact forkattention forkattention_compact cacheblend cacheblend_compact"
  else
    export VARIANTS="baseline baseline_compact forkattention forkattention_compact"
  fi
fi
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/../results/langgraph_prompt_compaction_ablation}"

exec "${SCRIPT_DIR}/run_hotpot_agentrix_e2e.sh"
