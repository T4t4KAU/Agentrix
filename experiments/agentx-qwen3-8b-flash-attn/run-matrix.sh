#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda1/hwx/Agentrix/experiments/agentx-qwen3-8b-flash-attn
source ./bench-env.sh
exec 9> matrix.lock
flock -n 9 || exit 1
echo $$ > matrix.pid
trap 'rc=$?; echo "$rc" > matrix.exit; if [ "$rc" -eq 0 ]; then echo success > status; else echo failed > status; fi' EXIT
[[ -f dataset.ready ]]
.venv/bin/python corpus_inventory.py
nvidia-smi -q > gpu-environment.txt
/mnt/sda1/hwx/.local/bin/uv pip freeze --python .venv/bin/python > installed-requirements.txt
for concurrency in 1 4 8; do
 echo "starting_server_c${concurrency}" > status
 .venv/bin/python restart_server.py "c${concurrency}"
 echo "running_c${concurrency}" > status
 .venv/bin/aiperf profile --scenario inferencex-agentx-mvp --url http://127.0.0.1:18000 --model Qwen3-8B --tokenizer Agentrix/Qwen3-8B-local --endpoint-type chat --public-dataset semianalysis_cc_traces_weka_062126 --max-context-length 131072 --concurrency "$concurrency" --streaming --use-server-token-count --extra-inputs ignore_eos:true --cache-bust first_turn_prefix --system-idle-gap-cap-seconds 10 --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 --warmup-requests-per-lane 10 --benchmark-duration 3600 --random-seed 20260707 --ui simple --artifact-dir "$RUN/c${concurrency}" > "c${concurrency}.log" 2>&1
 echo 0 > "c${concurrency}.exit"
 cp server.log "server-c${concurrency}.log"
done
