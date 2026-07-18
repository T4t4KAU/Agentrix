#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
PYTHON="${BENCHMARK_PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-${REPO_ROOT}/vllm/.venv/bin/vllm}"
MODEL_PATH="${MODEL_PATH:-}"
CASES_PATH="${CASES_PATH:-${BENCHMARK_DIR}/data/django_agentrix/cases_30k_b16_commit24.jsonl}"
LEFT_GPUS="${LEFT_GPUS:-0,1,2,3}"
RIGHT_GPUS="${RIGHT_GPUS:-4,5,6,7}"
LEFT_PORT="${LEFT_PORT:-9000}"
RIGHT_PORT="${RIGHT_PORT:-9001}"
LEFT_MODEL_NAME="${LEFT_MODEL_NAME:-agentrix-coding-demo}"
RIGHT_MODEL_NAME="${RIGHT_MODEL_NAME:-vllm-coding-demo}"
OUTPUT_DIR="${OUTPUT_DIR:-${BENCHMARK_DIR}/results/coding_agent_demo}"
CASE_OFFSET="${CASE_OFFSET:-0}"
CASE_COUNT="${CASE_COUNT:-12}"
ROUNDS="${ROUNDS:-1}"
MAX_TOKENS="${MAX_TOKENS:-128}"
MOCK="${MOCK:-0}"
SERVERS_ONLY="${SERVERS_ONLY:-0}"
CLIENT_ONLY="${CLIENT_ONLY:-0}"
CLEANUP_ONLY="${CLEANUP_ONLY:-0}"
TELEMETRY_PORT="${TELEMETRY_PORT:-9010}"
STARTUP_LOG_PATTERN='Starting vLLM|Initializing a V1 LLM engine|Loading safetensors|Model loading took|torch\.compile took|Available KV cache|GPU KV cache size|Maximum concurrency|Graph capturing finished|init engine|Application startup complete|Uvicorn running'
LEFT_PID=""
RIGHT_PID=""
TELEMETRY_PID=""

stop_recorded_demo_pid() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] || return 0
  local pid signal_target pgid current_pgid target_label
  pid="$(<"${pid_file}")"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    local command_line
    command_line="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
    if [[ "${pid_file}" == *supervisor.pid ]]; then
      [[ "${command_line}" == *run_coding_agent_demo.sh* ]] || {
        echo "Ignoring stale supervisor PID ${pid}; command no longer matches."
        rm -f -- "${pid_file}"
        return 0
      }
    elif [[ "${pid_file}" == *telemetry.pid ]]; then
      [[ "${command_line}" == *coding_agent_gpu_telemetry* ]] || {
        echo "Ignoring stale telemetry PID ${pid}; command no longer matches."
        rm -f -- "${pid_file}"
        return 0
      }
    elif [[ "${command_line}" != *vllm*serve* ]]; then
      echo "Ignoring stale server PID ${pid}; command no longer matches."
      rm -f -- "${pid_file}"
      return 0
    fi

    # vLLM launches EngineCore workers below the API process. Killing only the
    # recorded PID can orphan those workers and leave their CUDA contexts alive.
    # Every demo launch has its own session/process group, so terminate the whole
    # group after validating the recorded leader above.
    signal_target="${pid}"
    target_label="process ${pid}"
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
    current_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]')"
    if [[ "${pgid}" =~ ^[0-9]+$ && "${pgid}" != "${current_pgid}" ]]; then
      signal_target="-${pgid}"
      target_label="process group ${pgid}"
    fi

    echo "Stopping residual demo ${target_label} recorded in ${pid_file}."
    kill -TERM -- "${signal_target}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 -- "${signal_target}" 2>/dev/null || break
      sleep 0.25
    done
    if kill -0 -- "${signal_target}" 2>/dev/null; then
      echo "Residual demo ${target_label} did not stop after 5 seconds; killing it."
      kill -KILL -- "${signal_target}" 2>/dev/null || true
      for _ in $(seq 1 20); do
        kill -0 -- "${signal_target}" 2>/dev/null || break
        sleep 0.10
      done
    fi
  fi
  rm -f -- "${pid_file}"
}

cleanup_residual_demo_services() {
  mkdir -p "${OUTPUT_DIR}"
  stop_recorded_demo_pid "${OUTPUT_DIR}/supervisor.pid"
  stop_recorded_demo_pid "${OUTPUT_DIR}/left_server.pid"
  stop_recorded_demo_pid "${OUTPUT_DIR}/right_server.pid"
  stop_recorded_demo_pid "${OUTPUT_DIR}/telemetry.pid"

  local port pids pid
  for port in "${LEFT_PORT}" "${RIGHT_PORT}" "${TELEMETRY_PORT}"; do
    pids="$(fuser -n tcp "${port}" 2>/dev/null || true)"
    [[ -n "${pids//[[:space:]]/}" ]] || continue
    echo "Stopping residual demo listener on TCP ${port}: ${pids}."
    for pid in ${pids}; do
      kill -TERM "${pid}" 2>/dev/null || true
    done
  done

  for _ in $(seq 1 40); do
    for port in "${LEFT_PORT}" "${RIGHT_PORT}" "${TELEMETRY_PORT}"; do
      if fuser -n tcp "${port}" >/dev/null 2>&1; then
        sleep 0.25
        continue 2
      fi
    done
    break
  done
  for port in "${LEFT_PORT}" "${RIGHT_PORT}" "${TELEMETRY_PORT}"; do
    if fuser -n tcp "${port}" >/dev/null 2>&1; then
      echo "Residual demo ports did not become free after 10 seconds" >&2
      return 1
    fi
  done

  local gpu_line index used busy elapsed=0
  while ((elapsed < 120)); do
    busy=0
    while IFS=, read -r index used; do
      index="${index//[[:space:]]/}"
      used="${used//[[:space:]]/}"
      if [[ ",${LEFT_GPUS},${RIGHT_GPUS}," == *",${index},"* ]] && ((used > 1024)); then
        busy=1
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    ((busy == 0)) && return 0
    if ((elapsed % 5 == 0)); then
      echo "Waiting for residual CUDA contexts to release GPU memory (${elapsed}s)…"
    fi
    sleep 1
    ((elapsed += 1))
  done
  echo "Target GPUs still use more than 1 GiB after 120 seconds:" >&2
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >&2
  return 1
}

export PYTHONPATH="${BENCHMARK_DIR}/src:${REPO_ROOT}/application/src:${REPO_ROOT}/vllm${PYTHONPATH:+:${PYTHONPATH}}"

cleanup() {
  local pid
  for pid in "${LEFT_PID}" "${RIGHT_PID}"; do
    [[ -n "${pid}" ]] || continue
    # start_server uses setsid, making this PID the process-group leader for
    # the API server, EngineCore workers, and multiprocessing resource tracker.
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 -- "-${pid}" 2>/dev/null || break
      sleep 0.25
    done
    if kill -0 -- "-${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
  done
  if [[ -n "${TELEMETRY_PID}" ]] && kill -0 "${TELEMETRY_PID}" 2>/dev/null; then
    kill -TERM "${TELEMETRY_PID}" 2>/dev/null || true
    wait "${TELEMETRY_PID}" 2>/dev/null || true
  fi
  rm -f -- \
    "${OUTPUT_DIR}/supervisor.pid" \
    "${OUTPUT_DIR}/left_server.pid" \
    "${OUTPUT_DIR}/right_server.pid" \
    "${OUTPUT_DIR}/telemetry.pid"
}
trap cleanup EXIT INT TERM

wait_server() {
  local port="$1"
  local pid="$2"
  local log="$3"
  local show_progress="${4:-0}"
  local deadline=$((SECONDS + 600))
  local progress_line=""
  local previous_progress_line=""
  local selected_worker=""
  until curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "vLLM exited during startup; inspect ${log}" >&2
      tail -n 80 "${log}" >&2
      return 1
    fi
    if ((SECONDS >= deadline)); then
      echo "Timed out waiting for vLLM; inspect ${log}" >&2
      return 1
    fi
    if [[ "${show_progress}" == "1" ]]; then
      if [[ -z "${selected_worker}" ]]; then
        selected_worker="$(
          { tr '\r' '\n' <"${log}" \
            | grep -aoE '^\(Worker pid=[0-9]+\)' \
            | head -n 1; } || true
        )"
      fi
      progress_line="$(
        { tr '\r' '\n' <"${log}" \
          | grep -aE "${STARTUP_LOG_PATTERN}" \
          | awk -v worker="${selected_worker}" '
              /^\(Worker pid=/ && worker != "" && index($0, worker) != 1 { next }
              { print }
            ' \
          | tail -n 1; } || true
      )"
      if [[ -n "${progress_line}" && "${progress_line}" != "${previous_progress_line}" ]]; then
        echo "${progress_line}"
        previous_progress_line="${progress_line}"
      fi
    fi
    sleep 2
  done
}

start_server() {
  local side="$1"
  local gpu_ids="$2"
  local port="$3"
  local model_name="$4"
  local backend="$5"
  local prefix_routing="$6"
  local log="${OUTPUT_DIR}/${side}_server.log"
  setsid env \
    CUDA_VISIBLE_DEVICES="${gpu_ids}" \
    PYTHONHASHSEED=0 \
    VLLM_SERVER_DEV_MODE=1 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_FORK_ATTN_ENABLE_FOREST=1 \
    VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH=1 \
    VLLM_FORK_ATTN_FANOUT_SCHEDULING_ENABLED="${prefix_routing}" \
    VLLM_FORK_ATTN_FANOUT_ADMISSION_WINDOW=0 \
    VLLM_FORK_ATTN_DP_PREFIX_ROUTING="${prefix_routing}" \
    VLLM_FORK_ATTN_DP_RELOAD_REBALANCE=0 \
    VLLM_FORK_ATTN_DP_ARRIVAL_WAVE_MS=10 \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --host 127.0.0.1 --port "${port}" \
      --served-model-name "${model_name}" \
      --attention-backend "${backend}" --dtype float16 \
      --generation-config vllm --enable-prefix-caching --no-async-scheduling \
      --default-chat-template-kwargs '{"enable_thinking":false}' \
      --data-parallel-size 4 --api-server-count 1 \
      --gpu-memory-utilization 0.70 --num-gpu-blocks-override 3852 \
      --max-model-len 40960 --max-num-batched-tokens 16384 \
      --max-num-seqs 64 >"${log}" 2>&1 &
  STARTED_PID=$!
}

mkdir -p "${OUTPUT_DIR}"
if [[ "${MOCK}" == "1" ]]; then
  exec "${PYTHON}" -m coding_agent_demo_tui \
    --cases "${CASES_PATH}" --case-offset "${CASE_OFFSET}" \
    --case-count "${CASE_COUNT}" --mock
fi

if [[ "${CLIENT_ONLY}" == "1" ]]; then
  for port in "${LEFT_PORT}" "${RIGHT_PORT}"; do
    if ! curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null; then
      echo "Prestarted demo server on port ${port} is not ready" >&2
      exit 1
    fi
  done
  exec "${PYTHON}" -m coding_agent_demo_tui \
    --cases "${CASES_PATH}" --case-offset "${CASE_OFFSET}" \
    --case-count "${CASE_COUNT}" --rounds "${ROUNDS}" \
    --max-tokens "${MAX_TOKENS}" \
    --left-base-url "http://127.0.0.1:${LEFT_PORT}/v1" \
    --right-base-url "http://127.0.0.1:${RIGHT_PORT}/v1" \
    --left-model "${LEFT_MODEL_NAME}" --right-model "${RIGHT_MODEL_NAME}" \
    --left-gpus "${LEFT_GPUS}" --right-gpus "${RIGHT_GPUS}"
fi

if [[ "${CLEANUP_ONLY}" == "1" ]]; then
  cleanup_residual_demo_services
  echo "Residual demo services and ports have been cleared."
  exit 0
fi

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required unless MOCK=1" >&2
  exit 2
fi

cleanup_residual_demo_services
echo "$$" >"${OUTPUT_DIR}/supervisor.pid"

start_server left "${LEFT_GPUS}" "${LEFT_PORT}" "${LEFT_MODEL_NAME}" FORK_ATTN 1
LEFT_PID="${STARTED_PID}"
echo "${LEFT_PID}" >"${OUTPUT_DIR}/left_server.pid"
start_server right "${RIGHT_GPUS}" "${RIGHT_PORT}" "${RIGHT_MODEL_NAME}" FLASH_ATTN 0
RIGHT_PID="${STARTED_PID}"
echo "${RIGHT_PID}" >"${OUTPUT_DIR}/right_server.pid"
echo "Streaming Agentrix vLLM startup output; baseline is starting in parallel."
wait_server "${LEFT_PORT}" "${LEFT_PID}" "${OUTPUT_DIR}/left_server.log" 1 &
left_wait=$!
wait_server "${RIGHT_PORT}" "${RIGHT_PID}" "${OUTPUT_DIR}/right_server.log" 0 &
right_wait=$!
wait "${left_wait}"
wait "${right_wait}"
echo "Both vLLM servers are ready; starting the live dashboard."

if [[ "${SERVERS_ONLY}" == "1" ]]; then
  "${PYTHON}" -m coding_agent_gpu_telemetry \
    --host 127.0.0.1 --port "${TELEMETRY_PORT}" \
    >"${OUTPUT_DIR}/gpu_telemetry.log" 2>&1 &
  TELEMETRY_PID=$!
  echo "${TELEMETRY_PID}" >"${OUTPUT_DIR}/telemetry.pid"
  echo "Agentrix and baseline demo servers are ready on ports ${LEFT_PORT} and ${RIGHT_PORT}."
  wait -n "${LEFT_PID}" "${RIGHT_PID}"
  exit 1
fi

AGENTRIX_TOOL_KV_TRIM_ENABLED=0 \
AGENTRIX_TOOL_KV_TRIM_USE_PREDICTED_TTL=0 \
"${PYTHON}" -m coding_agent_demo_tui \
  --cases "${CASES_PATH}" --case-offset "${CASE_OFFSET}" \
  --case-count "${CASE_COUNT}" --rounds "${ROUNDS}" \
  --max-tokens "${MAX_TOKENS}" \
  --left-base-url "http://127.0.0.1:${LEFT_PORT}/v1" \
  --right-base-url "http://127.0.0.1:${RIGHT_PORT}/v1" \
  --left-model "${LEFT_MODEL_NAME}" --right-model "${RIGHT_MODEL_NAME}" \
  --left-gpus "${LEFT_GPUS}" --right-gpus "${RIGHT_GPUS}"
