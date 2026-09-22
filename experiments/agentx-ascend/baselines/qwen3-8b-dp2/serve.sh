#!/usr/bin/env bash
set -eo pipefail
source /data/Agentrix/activate-ascend.sh
export HF_HUB_OFFLINE=1
export ASCEND_RT_VISIBLE_DEVICES=0,1
exec vllm serve /data/models/Qwen3-8B --served-model-name Qwen3-8B --host 127.0.0.1 --port 8001 --tensor-parallel-size 1 --data-parallel-size 2 --dtype bfloat16 --max-model-len 131072 --max-num-seqs 8 --gpu-memory-utilization 0.9 --enforce-eager --hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' --enable-prompt-tokens-details
