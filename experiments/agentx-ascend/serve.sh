#!/usr/bin/env bash
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ ${DRY_RUN:-0} != 1 ]]; then
  source "$ASCEND_ENV"
  python - <<'PY'
from importlib.metadata import version
if version('vllm').split('+')[0] != '0.22.1':
    raise SystemExit('Use the separate Ascend vLLM 0.22.1 environment, not the CUDA submodule.')
PY
fi
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}
if [[ ! ${MAMBA_PREFER_REUSE_BOUNDARIES:-0} =~ ^[01]$ ]]; then
  echo 'MAMBA_PREFER_REUSE_BOUNDARIES must be 0 or 1.' >&2
  exit 2
fi
args=(serve "$MODEL_PATH" --served-model-name "$MODEL_NAME"
  --host 127.0.0.1 --port "$BACKEND_PORT"
  --tensor-parallel-size 1 --data-parallel-size 2 --dtype bfloat16
  --max-model-len "$CONTEXT_LENGTH" --max-num-seqs 8
  --gpu-memory-utilization 0.9 --enforce-eager
  --enable-prefix-caching --enable-prompt-tokens-details
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
  --long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD:-0}")
if [[ -n ${MAMBA_RETENTION_INTERVAL+x} || ${MAMBA_RETENTION_DIAGNOSTICS:-0} == 1 || ${MAMBA_PREFER_REUSE_BOUNDARIES:-0} == 1 ]]; then
  if [[ -n ${MAMBA_RETENTION_INTERVAL+x} && ! $MAMBA_RETENTION_INTERVAL =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo 'MAMBA_RETENTION_INTERVAL must be a nonnegative integer in tokens.' >&2
    exit 2
  fi
  retention_diagnostics=false
  [[ ${MAMBA_RETENTION_DIAGNOSTICS:-0} != 1 ]] || retention_diagnostics=true
  retention_prefer_boundaries=false
  [[ ${MAMBA_PREFER_REUSE_BOUNDARIES:-0} != 1 ]] || retention_prefer_boundaries=true
  if [[ $retention_prefer_boundaries == true && -z ${MAMBA_RETENTION_INTERVAL+x} ]]; then
    echo 'MAMBA_PREFER_REUSE_BOUNDARIES requires MAMBA_RETENTION_INTERVAL.' >&2
    exit 2
  fi
  args+=(--additional-config "{\"mamba_cache_retention\":{\"interval\":${MAMBA_RETENTION_INTERVAL:-null},\"diagnostics\":$retention_diagnostics,\"prefer_reuse_boundaries\":$retention_prefer_boundaries}}")
fi
if [[ $MODEL_NAME == Qwen3.5-9B ]]; then
  args+=(--language-model-only)
else
  args+=(--hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}')
fi
run_command "$VLLM_BIN" "${args[@]}"
