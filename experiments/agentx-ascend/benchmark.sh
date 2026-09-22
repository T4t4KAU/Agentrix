#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
: "${ARTIFACT_DIR:?Set ARTIFACT_DIR to a new directory for this run}"
if [[ ! $BENCHMARK_DURATION =~ ^[0-9]+$ ]] || ((BENCHMARK_DURATION < 900)); then
  echo 'The official AgentX scenario requires at least 900 seconds.' >&2
  exit 2
fi
if [[ ${DRY_RUN:-0} != 1 ]]; then
  [[ $(git -C "$HARNESS_ROOT" rev-parse HEAD) == "$HARNESS_COMMIT" ]] || {
    echo "Expected official harness commit $HARNESS_COMMIT" >&2; exit 2;
  }
  [[ ! -e $ARTIFACT_DIR ]] || { echo "Artifact directory already exists: $ARTIFACT_DIR" >&2; exit 2; }
  curl --max-time 10 -fsS "$BENCHMARK_URL/health" >/dev/null
fi
export HF_HOME=${HF_HOME:-/data/huggingface_home}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1
export AIPERF_HTTP_KEEPALIVE_TIMEOUT=4
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800 AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_DATASET_MMAP_CACHE_DIR=${AIPERF_DATASET_MMAP_CACHE_DIR:-$AGENTRIX_ROOT/agentx/dataset-cache}
export NO_PROXY="localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"
unset PYTHONPATH
run_command "$AIPERF_BIN" profile --scenario inferencex-agentx-mvp \
  --url "$BENCHMARK_URL" --model "$MODEL_NAME" --tokenizer "$MODEL_PATH" \
  --max-context-length "$CONTEXT_LENGTH" --endpoint-type chat \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --concurrency "$CONCURRENCY" --use-server-token-count --streaming \
  --extra-inputs ignore_eos:true --cache-bust first_turn_prefix \
  --system-idle-gap-cap-seconds 10 \
  --trajectory-start-min-ratio 0.0 --trajectory-start-max-ratio 1.0 \
  --benchmark-duration "$BENCHMARK_DURATION" --random-seed "$RANDOM_SEED" \
  --artifact-dir "$ARTIFACT_DIR" --ui simple
