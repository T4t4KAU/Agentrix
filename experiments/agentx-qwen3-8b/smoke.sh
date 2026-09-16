#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda1/hwx/Agentrix/experiments/agentx-qwen3-8b
source ./bench-env.sh
echo $$ > smoke.pid
trap 'rc=$?; echo "$rc" > smoke.exit' EXIT
.venv/bin/python - <<'PY'
import time, urllib.request
for i in range(180):
 try:
  urllib.request.urlopen('http://127.0.0.1:18000/health',timeout=2)
  break
 except Exception: time.sleep(2)
else: raise RuntimeError('vLLM not ready')
PY
exec 9> smoke.lock
flock -n 9
.venv/bin/aiperf profile --scenario inferencex-agentx-mvp --unsafe-override --url http://127.0.0.1:18000 --model Qwen3-8B --tokenizer Agentrix/Qwen3-8B-local --endpoint-type chat --custom-dataset-type weka_trace --input-file "$RUN/smoke-traces" --max-context-length 131072 --concurrency 1 --streaming --use-server-token-count --extra-inputs ignore_eos:true --cache-bust first_turn_prefix --system-idle-gap-cap-seconds 10 --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 --warmup-requests-per-lane 10 --benchmark-duration 60 --random-seed 20260707 --ui simple --artifact-dir "$RUN/smoke"
