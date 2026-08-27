#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGENTRIX_ROOT="${AGENTRIX_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$(dirname "${AGENTRIX_ROOT}")}"
BENCH_ROOT="${BENCH_ROOT:-${AGENTRIX_ROOT}/benchmark}"
VLLM_ROOT="${VLLM_ROOT:-${AGENTRIX_ROOT}/vllm}"
PYTHON="${PYTHON:-${VLLM_ROOT}/.venv/bin/python}"
RUN_DIR="${RUN_DIR:-$(readlink -f "${WORK_ROOT}/epd_experiments/runs/current")}"
TRACE_DIR="${TRACE_DIR:-${WORK_ROOT}/epd_experiments/traces_formal}"
RESULTS_DIR="${RESULTS_DIR:-${RUN_DIR}/temporal_matrix}"
PROXY_ENDPOINT="${PROXY_ENDPOINT:-http://127.0.0.1:10001/v1/chat/completions}"

if pgrep -af "modelscope download" >/dev/null; then
  echo "Refusing to benchmark while a ModelScope download is active." >&2
  pgrep -af "modelscope download" >&2
  exit 2
fi

curl -fsS "${PROXY_ENDPOINT%/v1/chat/completions}/health" >/dev/null
mkdir -p "${RESULTS_DIR}"

ec_path=$(sed -n 's/^EC_SHARED_STORAGE_PATH=//p' "${RUN_DIR}/run.env")
if [[ "${ec_path}" != /dev/shm/agentrix_epd_ec_* || ! -d "${ec_path}" ]]; then
  echo "Unexpected encoder-cache path: ${ec_path}" >&2
  exit 1
fi

clear_ec_cache() {
  find "${ec_path}" -mindepth 1 -delete
}

cleanup_sampler() {
  if [[ -n "${sampler_pid:-}" ]] && kill -0 "${sampler_pid}" 2>/dev/null; then
    kill "${sampler_pid}" 2>/dev/null || true
    wait "${sampler_pid}" 2>/dev/null || true
  fi
}
trap cleanup_sampler EXIT INT TERM

cd "${BENCH_ROOT}"

for repeat in 0 1 2; do
  mapfile -t probabilities < <(
    "${PYTHON}" -c '
import random
values = ["0.05", "0.10", "0.25", "0.50", "1.00"]
random.Random(20260902 + int(__import__("sys").argv[1])).shuffle(values)
print("\n".join(values))
' "${repeat}"
  )
  for probability in "${probabilities[@]}"; do
    name="sparse_p${probability}_r${repeat}"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) starting ${name}"
    "${PYTHON}" epd_heterogeneity/sample_telemetry.py \
      --output "${RESULTS_DIR}/${name}_telemetry.jsonl" \
      --interval 0.1 \
      --metrics-url http://127.0.0.1:19534/metrics \
      --metrics-url http://127.0.0.1:19535/metrics \
      >"${RESULTS_DIR}/${name}_telemetry.log" 2>&1 &
    sampler_pid=$!
    "${PYTHON}" epd_heterogeneity/run_session_trace.py \
      --trace "${TRACE_DIR}/sparse_p${probability}.jsonl" \
      --output "${RESULTS_DIR}/${name}_requests.jsonl" \
      --endpoint "${PROXY_ENDPOINT}"
    sleep 0.3
    cleanup_sampler
    sampler_pid=""
    clear_ec_cache
    sleep 2
  done
done

"${PYTHON}" epd_heterogeneity/summarize_temporal_matrix.py \
  --results-dir "${RESULTS_DIR}" \
  --proxy-log "${RUN_DIR}/proxy.log"

echo "Temporal matrix complete: ${RESULTS_DIR}"
