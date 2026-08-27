#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGENTRIX_ROOT="${AGENTRIX_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$(dirname "${AGENTRIX_ROOT}")}"
BENCH_ROOT="${BENCH_ROOT:-${AGENTRIX_ROOT}/benchmark}"
VLLM_ROOT="${VLLM_ROOT:-${AGENTRIX_ROOT}/vllm}"
PYTHON="${PYTHON:-${VLLM_ROOT}/.venv/bin/python}"
RUN_DIR="${RUN_DIR:-$(readlink -f "${WORK_ROOT}/epd_experiments/runs/current")}"
RESULTS_DIR="${RESULTS_DIR:-${RUN_DIR}/text_matrix}"
PD_ENDPOINT="${PD_ENDPOINT:-http://127.0.0.1:19535/v1/completions}"

if pgrep -af "modelscope download" >/dev/null; then
  echo "Refusing to benchmark while a ModelScope download is active." >&2
  pgrep -af "modelscope download" >&2
  exit 2
fi

curl -fsS "${PD_ENDPOINT%/v1/completions}/health" >/dev/null
mkdir -p "${RESULTS_DIR}"

cleanup_sampler() {
  if [[ -n "${sampler_pid:-}" ]] && kill -0 "${sampler_pid}" 2>/dev/null; then
    kill "${sampler_pid}" 2>/dev/null || true
    wait "${sampler_pid}" 2>/dev/null || true
  fi
}
trap cleanup_sampler EXIT INT TERM

cd "${BENCH_ROOT}"

for input_tokens in 512 4096 16000; do
  "${PYTHON}" epd_heterogeneity/run_text_profile.py \
    --input-tokens "${input_tokens}" \
    --output-tokens 1 \
    --concurrency 4 \
    --output "${RESULTS_DIR}/prefill_i${input_tokens}_warmup.jsonl" \
    --endpoint "${PD_ENDPOINT}"
done

for input_tokens in 128 4096; do
  for output_tokens in 256 1024; do
    "${PYTHON}" epd_heterogeneity/run_text_profile.py \
      --input-tokens "${input_tokens}" \
      --output-tokens "${output_tokens}" \
      --concurrency 4 \
      --output "${RESULTS_DIR}/decode_i${input_tokens}_o${output_tokens}_warmup.jsonl" \
      --endpoint "${PD_ENDPOINT}"
  done
done

for repeat in 0 1 2; do
  mapfile -t configs < <(
    "${PYTHON}" -c '
import random
configs = []
for input_tokens in (512, 4096, 16000):
    for concurrency in (1, 8, 16):
        configs.append(("prefill", input_tokens, 1, concurrency))
for input_tokens in (128, 4096):
    for output_tokens in (256, 1024):
        for concurrency in (1, 8, 16):
            configs.append(("decode", input_tokens, output_tokens, concurrency))
random.Random(20260830 + int(__import__("sys").argv[1])).shuffle(configs)
for config in configs:
    print(":".join(map(str, config)))
' "${repeat}"
  )
  for config in "${configs[@]}"; do
    IFS=: read -r workload input_tokens output_tokens concurrency <<<"${config}"
    if [[ "${workload}" == prefill ]]; then
      if [[ "${concurrency}" -eq 1 ]]; then
        request_count=16
      elif [[ "${concurrency}" -eq 8 ]]; then
        request_count=32
      else
        request_count=64
      fi
    elif [[ "${concurrency}" -eq 1 ]]; then
      request_count=4
    elif [[ "${concurrency}" -eq 8 ]]; then
      request_count=16
    else
      request_count=32
    fi
    name="${workload}_i${input_tokens}_o${output_tokens}_c${concurrency}_r${repeat}"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) starting ${name}"
    "${PYTHON}" epd_heterogeneity/sample_telemetry.py \
      --output "${RESULTS_DIR}/${name}_telemetry.jsonl" \
      --interval 0.1 \
      --metrics-url http://127.0.0.1:19535/metrics \
      >"${RESULTS_DIR}/${name}_telemetry.log" 2>&1 &
    sampler_pid=$!
    "${PYTHON}" epd_heterogeneity/run_text_profile.py \
      --input-tokens "${input_tokens}" \
      --output-tokens "${output_tokens}" \
      --concurrency "${concurrency}" \
      --output "${RESULTS_DIR}/${name}_requests.jsonl" \
      --endpoint "${PD_ENDPOINT}"
    sleep 0.3
    cleanup_sampler
    sampler_pid=""
    sleep 2
  done
done

"${PYTHON}" epd_heterogeneity/summarize_text_matrix.py \
  --results-dir "${RESULTS_DIR}"

echo "Text matrix complete: ${RESULTS_DIR}"
