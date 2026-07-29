#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
PYTHON="${BENCHMARK_PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-${REPO_ROOT}/vllm/.venv/bin/python}"
SOURCE_CLI="${VLLM_SOURCE_CLI:-${BENCHMARK_DIR}/scripts/vllm_source_cli.py}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
MODEL_NAME="${SERVED_MODEL_NAME:-agentrix-coding-quality}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
DP_REPLICAS="${DP_REPLICAS:-4}"
PORT="${PORT:-9000}"
BASELINE_NUM_GPU_BLOCKS="${BASELINE_NUM_GPU_BLOCKS:-3852}"
AGENTRIX_NUM_GPU_BLOCKS="${AGENTRIX_NUM_GPU_BLOCKS:-2600}"
SERVER_PID=""
SAMPLER_PID=""

export PYTHONPATH="${BENCHMARK_DIR}/src:${REPO_ROOT}/application/src:${REPO_ROOT}/vllm${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_ROOT}"

if [[ "${SKIP_ORACLE_PREFLIGHT:-0}" != 1 ]]; then
  "${PYTHON}" "${BENCHMARK_DIR}/scripts/validate_coding_oracle_tasks.py" \
    --task-root "${BENCHMARK_DIR}/coding_tasks" \
    --source-root "${SOURCE_ROOT}" \
    --output "${OUTPUT_ROOT}/oracle_preflight.json"
fi

if [[ "${ALLOW_BUSY_GPUS:-0}" != 1 ]]; then
  busy="$(
    nvidia-smi -i "${GPU_IDS}" --query-gpu=memory.used \
      --format=csv,noheader,nounits |
      awk '$1 > 1024 {count++} END {print count+0}'
  )"
  if [[ "${busy}" != 0 ]]; then
    echo "selected GPUs are already in use; refusing a confounded A/B run" >&2
    exit 2
  fi
fi

stop_processes() {
  if [[ -n "${SAMPLER_PID}" ]] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then
    kill -TERM "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
  fi
  SAMPLER_PID=""
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_processes EXIT INT TERM

wait_for_server() {
  local log_path="$1"
  local deadline=$((SECONDS + 900))
  until curl --silent --fail --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -n 160 "${log_path}" >&2
      return 1
    fi
    ((SECONDS < deadline)) || return 1
    sleep 2
  done
}

run_arm() {
  local arm="$1"
  local backend="$2"
  local routing="$3"
  local compaction="$4"
  local num_gpu_blocks="$5"
  local arm_dir="${OUTPUT_ROOT}/${arm}"
  mkdir -p "${arm_dir}"
  "${PYTHON}" - "${arm_dir}/arm_metadata.json" "${arm}" "${backend}" \
    "${routing}" "${compaction}" "${num_gpu_blocks}" <<'PY'
import json
import sys
from pathlib import Path

path, arm, backend, routing, compaction, blocks = sys.argv[1:]
Path(path).write_text(json.dumps({
    "arm": arm,
    "attention_backend": backend,
    "prefix_routing": bool(int(routing)),
    "prompt_compaction": bool(int(compaction)),
    "num_gpu_blocks": int(blocks),
}, indent=2) + "\n")
PY

  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}" \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_FORK_ATTN_ENABLE_FOREST=1 \
  VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH=1 \
  VLLM_FORK_ATTN_FANOUT_SCHEDULING_ENABLED="${routing}" \
  VLLM_FORK_ATTN_DP_PREFIX_ROUTING="${routing}" \
  VLLM_FORK_ATTN_DP_RELOAD_REBALANCE=0 \
  VLLM_FORK_ATTN_DP_ARRIVAL_WAVE_MS=10 \
  "${VLLM_PYTHON}" "${SOURCE_CLI}" serve "${MODEL_PATH}" \
    --host 127.0.0.1 --port "${PORT}" --served-model-name "${MODEL_NAME}" \
    --attention-backend "${backend}" --dtype bfloat16 \
    --generation-config vllm --enable-prefix-caching --no-async-scheduling \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --data-parallel-size "${DP_REPLICAS}" --api-server-count 1 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.70}" \
    --num-gpu-blocks-override "${num_gpu_blocks}" \
    --max-model-len "${MAX_MODEL_LEN:-40960}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-16384}" \
    --max-num-seqs "${MAX_NUM_SEQS:-64}" \
    >"${arm_dir}/vllm_server.log" 2>&1 &
  SERVER_PID=$!
  wait_for_server "${arm_dir}/vllm_server.log"

  nvidia-smi --query-compute-apps=timestamp,gpu_uuid,pid,used_memory \
    --format=csv,noheader,nounits -lms 200 >"${arm_dir}/gpu_process_memory.csv" &
  SAMPLER_PID=$!

  while IFS='|' read -r task_id repository; do
    case "${repository}" in
      django/django)
        cases="${BENCHMARK_DIR}/data/django_agentrix/cases_30k_b16.jsonl"
        source_repo="${SOURCE_ROOT}/django"
        ;;
      sqlite/sqlite)
        cases="${BENCHMARK_DIR}/data/sqlite_agentrix/cases_30k_b16.jsonl"
        source_repo="${SOURCE_ROOT}/sqlite"
        ;;
      FFmpeg/FFmpeg)
        cases="${BENCHMARK_DIR}/data/ffmpeg_agentrix/cases_30k_b16.jsonl"
        source_repo="${SOURCE_ROOT}/ffmpeg"
        ;;
      *)
        echo "unsupported repository: ${repository}" >&2
        return 1
        ;;
    esac
    task_dir="${arm_dir}/${task_id}"
    mkdir -p "${task_dir}"
    args=(
      -m coding_agent_e2e_runner
      --base-url "http://127.0.0.1:${PORT}/v1"
      --model "${MODEL_NAME}"
      --cases "${cases}"
      --task-id "${task_id}"
      --task-root "${BENCHMARK_DIR}/coding_tasks"
      --repo "${source_repo}"
      --rounds "${ROUNDS:-3}"
      --trajectory-mode live
      --branch-output-tokens "${BRANCH_OUTPUT_TOKENS:-128}"
      --parent-output-tokens "${PARENT_OUTPUT_TOKENS:-512}"
      --max-tool-steps "${MAX_TOOL_STEPS:-14}"
      --output "${task_dir}/run.json"
    )
    if [[ "${compaction}" == 1 ]]; then
      args+=(--prompt-compaction)
    fi
    "${PYTHON}" "${args[@]}" | tee "${task_dir}/runner.log"
  done < <(
    "${PYTHON}" - "${BENCHMARK_DIR}/coding_tasks" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index = json.loads((root / "index.json").read_text())
for entry in index["tasks"]:
    if not entry.get("score_in_formal_accuracy", False):
        continue
    task = json.loads((root / entry["manifest"]).read_text())
    print(f"{task['task_id']}|{task['repository']}")
PY
  )
  stop_processes
}

if [[ "${RUN_BASELINE:-1}" == 1 ]]; then
  run_arm flash_ordinary_dp FLASH_ATTN 0 0 "${BASELINE_NUM_GPU_BLOCKS}"
fi
if [[ "${RUN_OPTIMIZED:-1}" == 1 ]]; then
  run_arm agentrix FORK_ATTN 1 1 "${AGENTRIX_NUM_GPU_BLOCKS}"
fi

"${PYTHON}" -m coding_quality_report \
  --baseline "${OUTPUT_ROOT}/flash_ordinary_dp" \
  --optimized "${OUTPUT_ROOT}/agentrix" \
  --margin "${NONINFERIORITY_MARGIN:-0.05}" \
  --minimum-resolved-rate "${MINIMUM_RESOLVED_RATE:-0.50}" \
  --output-json "${OUTPUT_ROOT}/quality_ab.json" \
  --output-markdown "${OUTPUT_ROOT}/quality_ab.md"
