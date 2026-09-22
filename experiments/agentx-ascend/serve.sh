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
args=(serve "$MODEL_PATH" --served-model-name "$MODEL_NAME"
  --host 127.0.0.1 --port "$BACKEND_PORT"
  --tensor-parallel-size 1 --data-parallel-size 2 --dtype bfloat16
  --max-model-len "$CONTEXT_LENGTH" --max-num-seqs 8
  --gpu-memory-utilization 0.9 --enforce-eager
  --enable-prefix-caching --enable-prompt-tokens-details
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
  --long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD:-0}")
if [[ $MODEL_NAME == Qwen3.5-9B ]]; then
  args+=(--language-model-only)
else
  args+=(--hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}')
fi
run_command "$VLLM_BIN" "${args[@]}"
