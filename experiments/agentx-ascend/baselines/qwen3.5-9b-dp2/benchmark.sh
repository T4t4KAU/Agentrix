#!/usr/bin/env bash
set -eo pipefail
source /data/Agentrix/agentx/.venv/bin/activate
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=/data/huggingface_home
export HF_ENDPOINT=https://huggingface.co
export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1
export HTTPS_PROXY=http://127.0.0.1:17897
export HTTP_PROXY=http://127.0.0.1:17897
export NO_PROXY=localhost,127.0.0.1
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_HTTP_KEEPALIVE_TIMEOUT=4
export AIPERF_DATASET_MMAP_CACHE_DIR=/data/Agentrix/agentx/dataset-cache
cd /data/Agentrix/agentx/harness
for i in {1..120}; do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then break; fi
  sleep 5
done
curl -fsS http://127.0.0.1:8000/health
exec aiperf profile \
  --scenario inferencex-agentx-mvp \
  --url http://127.0.0.1:8000 \
  --model Qwen3.5-9B \
  --tokenizer /data/models/Qwen3.5-9B \
  --max-context-length 262144 \
  --endpoint-type chat \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --concurrency 16 \
  --use-server-token-count \
  --streaming \
  --extra-inputs ignore_eos:true \
  --cache-bust first_turn_prefix \
  --system-idle-gap-cap-seconds 10 \
  --trajectory-start-min-ratio 0.0 \
  --trajectory-start-max-ratio 1.0 \
  --benchmark-duration 900 \
  --random-seed 20260707 \
  --artifact-dir /data/Agentrix/agentx/runs/qwen3.5-9b-dp2/artifacts-c16 \
  --ui simple
