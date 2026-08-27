#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AGENTRIX_ROOT="${AGENTRIX_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$(dirname "${AGENTRIX_ROOT}")}"
BENCH_ROOT="${BENCH_ROOT:-${AGENTRIX_ROOT}/benchmark}"
VLLM_ROOT="${VLLM_ROOT:-${AGENTRIX_ROOT}/vllm}"
PYTHON="${PYTHON:-${VLLM_ROOT}/.venv/bin/python}"
RUN_DIR="${RUN_DIR:-$(readlink -f "${WORK_ROOT}/epd_experiments/runs/current")}"
MANIFEST="${MANIFEST:-${WORK_ROOT}/datasets/epd_coco_control/images.jsonl}"
PROXY_ENDPOINT="${PROXY_ENDPOINT:-http://127.0.0.1:10001/v1/chat/completions}"
RESULTS_DIR="${RESULTS_DIR:-${RUN_DIR}/encoder_matrix}"
DIRECT_ENCODER="${DIRECT_ENCODER:-0}"

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

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  uname -a
  lscpu
  free -h
  nvidia-smi --query-gpu=index,name,uuid,memory.total,persistence_mode,power.limit,clocks.current.sm,clocks.current.memory --format=csv
  nvidia-smi topo -m
  git -C "${VLLM_ROOT}" status --short --branch
  git -C "${VLLM_ROOT}" rev-parse HEAD
  git -C "${VLLM_ROOT}" describe --tags --exact-match
  "${PYTHON}" -V
} >"${RESULTS_DIR}/environment.txt" 2>&1

cd "${BENCH_ROOT}"

for bucket in low medium 1080p_class 4k_class; do
  "${PYTHON}" epd_heterogeneity/run_encoder_profile.py \
    --image-manifest "${MANIFEST}" \
    --bucket "${bucket}" \
    --requests 8 \
    --concurrency 4 \
    --image-offset 0 \
    --output "${RESULTS_DIR}/${bucket}_warmup.jsonl" \
    --endpoint "${PROXY_ENDPOINT}"
  clear_ec_cache
done

for repeat in 0 1 2; do
  mapfile -t configs < <(
    "${PYTHON}" -c '
import random
configs = [(b, c) for b in ("low", "medium", "1080p_class", "4k_class") for c in (1, 4, 16)]
random.Random(20260827 + int(__import__("sys").argv[1])).shuffle(configs)
for bucket, concurrency in configs:
    print(f"{bucket}:{concurrency}")
' "${repeat}"
  )
  for config in "${configs[@]}"; do
    bucket=${config%:*}
    concurrency=${config#*:}
    if [[ "${concurrency}" -eq 16 ]]; then
      request_count=64
      image_offset=$((208 + repeat * 64))
    elif [[ "${concurrency}" -eq 4 ]]; then
      request_count=32
      image_offset=$((112 + repeat * 32))
    else
      request_count=32
      image_offset=$((16 + repeat * 32))
    fi
    name="${bucket}_c${concurrency}_r${repeat}"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) starting ${name}"
    "${PYTHON}" epd_heterogeneity/sample_telemetry.py \
      --output "${RESULTS_DIR}/${name}_telemetry.jsonl" \
      --interval 0.1 \
      --metrics-url http://127.0.0.1:19534/metrics \
      --metrics-url http://127.0.0.1:19535/metrics \
      >"${RESULTS_DIR}/${name}_telemetry.log" 2>&1 &
    sampler_pid=$!
    "${PYTHON}" epd_heterogeneity/run_encoder_profile.py \
      --image-manifest "${MANIFEST}" \
      --bucket "${bucket}" \
      --requests "${request_count}" \
      --concurrency "${concurrency}" \
      --image-offset "${image_offset}" \
      --output "${RESULTS_DIR}/${name}_requests.jsonl" \
      --endpoint "${PROXY_ENDPOINT}"
    sleep 0.3
    cleanup_sampler
    sampler_pid=""
    clear_ec_cache
    sleep 2
  done
done

summary_args=(
  --results-dir "${RESULTS_DIR}"
  --proxy-log "${RUN_DIR}/proxy.log"
)
if [[ "${DIRECT_ENCODER}" == 1 ]]; then
  summary_args+=(--direct-encoder)
fi
"${PYTHON}" epd_heterogeneity/summarize_encoder_matrix.py "${summary_args[@]}"

echo "Encoder matrix complete: ${RESULTS_DIR}"
