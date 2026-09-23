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
  --gpu-memory-utilization 0.9
  --enable-prefix-caching --enable-prompt-tokens-details
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
  --long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD:-0}")
for flag in ENABLE_NPUGRAPH_EX ENABLE_CPU_BINDING BATCH_DIAGNOSTICS; do
  if [[ -n ${!flag+x} && ! ${!flag} =~ ^[01]$ ]]; then
    echo "$flag must be 0 or 1." >&2
    exit 2
  fi
done
if [[ ${BATCH_DIAGNOSTICS:-0} == 1 ]]; then
  args+=(--cudagraph-metrics --enable-logging-iteration-details)
fi
cpu_binding=true
[[ ${ENABLE_CPU_BINDING:-1} != 0 ]] || cpu_binding=false
npugraph_ex=false
[[ ${ENABLE_NPUGRAPH_EX:-0} != 1 ]] || npugraph_ex=true
if [[ $npugraph_ex == true && ${EXECUTION_MODE:-eager} != decode-graph ]]; then
  echo 'ENABLE_NPUGRAPH_EX requires EXECUTION_MODE=decode-graph.' >&2
  exit 2
fi
additional_config="\"enable_cpu_binding\":$cpu_binding"
case ${EXECUTION_MODE:-eager} in
  eager) args+=(--enforce-eager) ;;
  decode-graph)
    # Ascend requires VLLM_COMPILE for graph metadata.
    args+=(--compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}')
    additional_config+=",\"ascend_compilation_config\":{\"enable_npugraph_ex\":$npugraph_ex}"
    ;;
  *) echo 'EXECUTION_MODE must be eager or decode-graph.' >&2; exit 2 ;;
esac
if [[ -n ${KV_CACHE_MEMORY_BYTES+x} ]]; then
  if [[ ! $KV_CACHE_MEMORY_BYTES =~ ^[1-9][0-9]*$ ]]; then
    echo 'KV_CACHE_MEMORY_BYTES must be a positive integer.' >&2
    exit 2
  fi
  args+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
fi
if [[ -n ${PROFILE_DIR:-} ]]; then
  args+=(--profiler-config.profiler torch
    --profiler-config.torch_profiler_dir "$PROFILE_DIR"
    --profiler-config.torch_profiler_with_stack false
    --profiler-config.ignore_frontend true)
fi
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
  [[ -z $additional_config ]] || additional_config+=','
  additional_config+="\"mamba_cache_retention\":{\"interval\":${MAMBA_RETENTION_INTERVAL:-null},\"diagnostics\":$retention_diagnostics,\"prefer_reuse_boundaries\":$retention_prefer_boundaries}"
fi
[[ -z $additional_config ]] || args+=(--additional-config "{$additional_config}")
if [[ $MODEL_NAME == Qwen3.5-9B ]]; then
  args+=(--language-model-only)
else
  args+=(--hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}')
fi
run_command "$VLLM_BIN" "${args[@]}"
