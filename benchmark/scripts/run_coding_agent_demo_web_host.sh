#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
PYTHON="${BENCHMARK_PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}"
SSH_TARGET="${SSH_TARGET:-}"
SSH_PORT="${SSH_PORT:-}"
REMOTE_ROOT="${REMOTE_ROOT:-/test__02/hwx/Agentrix}"
REMOTE_MODEL_PATH="${REMOTE_MODEL_PATH:-/test__02/hwx/Qwen3-32B}"
REMOTE_LEFT_PORT="${REMOTE_LEFT_PORT:-9000}"
REMOTE_RIGHT_PORT="${REMOTE_RIGHT_PORT:-9001}"
REMOTE_TELEMETRY_PORT="${REMOTE_TELEMETRY_PORT:-9010}"
LOCAL_LEFT_PORT="${LOCAL_LEFT_PORT:-19000}"
LOCAL_RIGHT_PORT="${LOCAL_RIGHT_PORT:-19001}"
LOCAL_TELEMETRY_PORT="${LOCAL_TELEMETRY_PORT:-19010}"
WEB_PORT="${WEB_PORT:-8088}"
CASES_PATH="${CASES_PATH:-${BENCHMARK_DIR}/data/django_agentrix/cases_30k_b16_commit24.jsonl}"
CASE_OFFSET="${CASE_OFFSET:-0}"
CASE_COUNT="${CASE_COUNT:-12}"
ROUNDS="${ROUNDS:-1}"
MAX_TOKENS="${MAX_TOKENS:-128}"
LEFT_MODEL_NAME="${LEFT_MODEL_NAME:-agentrix-coding-demo}"
RIGHT_MODEL_NAME="${RIGHT_MODEL_NAME:-vllm-coding-demo}"

TUNNEL_PID=""
REMOTE_SERVICE_SSH_PID=""
WEB_PID=""

if [[ -z "${SSH_TARGET}" ]]; then
  echo "SSH_TARGET is required (use a hostname or an SSH config alias)." >&2
  echo "Example: SSH_TARGET=agentrix-demo $0" >&2
  exit 2
fi
SSH_PORT_ARGS=()
if [[ -n "${SSH_PORT}" ]]; then
  SSH_PORT_ARGS=(-p "${SSH_PORT}")
fi

cleanup() {
  if [[ -n "${WEB_PID}" ]] && kill -0 "${WEB_PID}" 2>/dev/null; then
    kill -TERM "${WEB_PID}" 2>/dev/null || true
    wait "${WEB_PID}" 2>/dev/null || true
  fi
  if [[ -n "${REMOTE_SERVICE_SSH_PID}" ]] && kill -0 "${REMOTE_SERVICE_SSH_PID}" 2>/dev/null; then
    kill -TERM "${REMOTE_SERVICE_SSH_PID}" 2>/dev/null || true
    wait "${REMOTE_SERVICE_SSH_PID}" 2>/dev/null || true
  fi
  if [[ -n "${TUNNEL_PID}" ]] && kill -0 "${TUNNEL_PID}" 2>/dev/null; then
    kill -TERM "${TUNNEL_PID}" 2>/dev/null || true
    wait "${TUNNEL_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cleanup_local_demo_listeners() {
  local port pids pid
  for port in "${WEB_PORT}" "${LOCAL_LEFT_PORT}" "${LOCAL_RIGHT_PORT}" "${LOCAL_TELEMETRY_PORT}"; do
    pids="$(fuser -n tcp "${port}" 2>/dev/null || true)"
    [[ -n "${pids//[[:space:]]/}" ]] || continue
    echo "Stopping residual local demo listener on TCP ${port}: ${pids}."
    for pid in ${pids}; do
      kill -TERM "${pid}" 2>/dev/null || true
    done
  done
  for _ in $(seq 1 40); do
    for port in "${WEB_PORT}" "${LOCAL_LEFT_PORT}" "${LOCAL_RIGHT_PORT}" "${LOCAL_TELEMETRY_PORT}"; do
      if fuser -n tcp "${port}" >/dev/null 2>&1; then
        sleep 0.25
        continue 2
      fi
    done
    return 0
  done
  echo "Residual local demo ports did not become free after 10 seconds" >&2
  return 1
}

wait_endpoint() {
  local port="$1"
  local deadline=$((SECONDS + 900))
  until curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null; do
    if [[ -z "${TUNNEL_PID}" ]] || ! kill -0 "${TUNNEL_PID}" 2>/dev/null; then
      echo "SSH tunnel exited during startup" >&2
      return 1
    fi
    if [[ -n "${REMOTE_SERVICE_SSH_PID}" ]] && ! kill -0 "${REMOTE_SERVICE_SSH_PID}" 2>/dev/null; then
      echo "Remote demo service command exited during startup" >&2
      return 1
    fi
    if ((SECONDS >= deadline)); then
      echo "Timed out waiting for forwarded endpoint on localhost:${port}" >&2
      return 1
    fi
    sleep 2
  done
}

printf -v remote_command \
  'cd %q && exec env MODEL_PATH=%q SERVERS_ONLY=1 LEFT_PORT=%q RIGHT_PORT=%q TELEMETRY_PORT=%q LEFT_MODEL_NAME=%q RIGHT_MODEL_NAME=%q benchmark/scripts/run_coding_agent_demo.sh' \
  "${REMOTE_ROOT}" "${REMOTE_MODEL_PATH}" "${REMOTE_LEFT_PORT}" \
  "${REMOTE_RIGHT_PORT}" "${REMOTE_TELEMETRY_PORT}" "${LEFT_MODEL_NAME}" "${RIGHT_MODEL_NAME}"

echo "Clearing residual local and remote demo services…"
cleanup_local_demo_listeners
ssh -T "${SSH_PORT_ARGS[@]}" "${SSH_TARGET}" \
  "cd $(printf '%q' "${REMOTE_ROOT}") && CLEANUP_ONLY=1 benchmark/scripts/run_coding_agent_demo.sh"

echo "Opening the SSH forwarding tunnel…"
ssh -N -T "${SSH_PORT_ARGS[@]}" \
  -o LogLevel=QUIET -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=20 -o ServerAliveCountMax=3 \
  -L "127.0.0.1:${LOCAL_LEFT_PORT}:127.0.0.1:${REMOTE_LEFT_PORT}" \
  -L "127.0.0.1:${LOCAL_RIGHT_PORT}:127.0.0.1:${REMOTE_RIGHT_PORT}" \
  -L "127.0.0.1:${LOCAL_TELEMETRY_PORT}:127.0.0.1:${REMOTE_TELEMETRY_PORT}" \
  "${SSH_TARGET}" &
TUNNEL_PID=$!
sleep 0.5
if ! kill -0 "${TUNNEL_PID}" 2>/dev/null; then
  wait "${TUNNEL_PID}"
fi

echo "Starting both remote DP=4 services…"
ssh -T "${SSH_PORT_ARGS[@]}" \
  -o ServerAliveInterval=20 -o ServerAliveCountMax=3 \
  "${SSH_TARGET}" "${remote_command}" &
REMOTE_SERVICE_SSH_PID=$!

wait_endpoint "${LOCAL_LEFT_PORT}" &
left_wait=$!
wait_endpoint "${LOCAL_RIGHT_PORT}" &
right_wait=$!
wait "${left_wait}"
wait "${right_wait}"
echo "Both remote vLLM services are ready; starting the local dashboard…"

export PYTHONPATH="${BENCHMARK_DIR}/src:${REPO_ROOT}/application/src${PYTHONPATH:+:${PYTHONPATH}}"
AGENTRIX_TOOL_KV_TRIM_ENABLED=0 \
AGENTRIX_TOOL_KV_TRIM_USE_PREDICTED_TTL=0 \
"${PYTHON}" -m coding_agent_demo_web \
  --cases "${CASES_PATH}" --case-offset "${CASE_OFFSET}" \
  --case-count "${CASE_COUNT}" --rounds "${ROUNDS}" \
  --max-tokens "${MAX_TOKENS}" \
  --left-base-url "http://127.0.0.1:${LOCAL_LEFT_PORT}/v1" \
  --right-base-url "http://127.0.0.1:${LOCAL_RIGHT_PORT}/v1" \
  --left-model "${LEFT_MODEL_NAME}" --right-model "${RIGHT_MODEL_NAME}" \
  --left-gpus 0,1,2,3 --right-gpus 4,5,6,7 \
  --left-gpu-metrics-url "http://127.0.0.1:${LOCAL_TELEMETRY_PORT}/gpu" \
  --right-gpu-metrics-url "http://127.0.0.1:${LOCAL_TELEMETRY_PORT}/gpu" \
  --web-host 127.0.0.1 --web-port "${WEB_PORT}" --no-open-browser &
WEB_PID=$!

dashboard_url="http://127.0.0.1:${WEB_PORT}"
dashboard_ready=0
for _ in $(seq 1 60); do
  if curl --silent --fail --max-time 1 "${dashboard_url}/" >/dev/null; then
    dashboard_ready=1
    break
  fi
  if ! kill -0 "${WEB_PID}" 2>/dev/null; then
    wait "${WEB_PID}"
  fi
  sleep 0.25
done
if [[ "${dashboard_ready}" != "1" ]]; then
  echo "Local dashboard did not become ready within 15 seconds" >&2
  exit 1
fi

echo "Browser dashboard is ready: ${dashboard_url}"
xdg-open "${dashboard_url}" >/dev/null 2>&1 &
wait "${WEB_PID}"
