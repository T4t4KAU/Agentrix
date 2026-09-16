#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/sda1/hwx/Agentrix
RUN=$ROOT/experiments/agentx-qwen3-8b-flash-attn
export PATH="$ROOT/vllm/.venv/bin:/mnt/sda1/hwx/.local/bin:/usr/local/cuda-12.8/bin:$PATH"
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$ROOT/vllm
export TMPDIR=/mnt/sda1/hwx/tmp
export XDG_CACHE_HOME=/mnt/sda1/hwx/.cache
export HF_HOME=/mnt/sda1/hwx/.cache/huggingface
export TRITON_CACHE_DIR=/mnt/sda1/hwx/.cache/triton
export HF_HUB_OFFLINE=1
cd "$ROOT/vllm"
echo $$ > "$RUN/server.pid"
exec .venv/bin/vllm serve /mnt/sda1/hwx/models/Qwen3-8B --served-model-name Qwen3-8B --host 127.0.0.1 --port 18000 --dtype bfloat16 --gpu-memory-utilization 0.9 --enable-prefix-caching --max-num-seqs 64 --max-num-batched-tokens 8192 --attention-config '{"backend":"FLASH_ATTN"}' --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' --max-model-len 131072 --enable-prompt-tokens-details
