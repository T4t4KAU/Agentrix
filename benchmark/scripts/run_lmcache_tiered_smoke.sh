#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-results/lmcache_tiered_smoke}"
OUTPUT_ROOT="${BENCHMARK_DIR}/${OUTPUT_DIR}"
disk_path_was_set=0
[[ -n "${LMCACHE_DISK_PATH:-}" ]] && disk_path_was_set=1
LMCACHE_DISK_PATH="${LMCACHE_DISK_PATH:-${OUTPUT_ROOT}/lmcache_disk}"
LMCACHE_CONFIG_FILE="${OUTPUT_ROOT}/lmcache.yaml"

mkdir -p "${OUTPUT_ROOT}" "${LMCACHE_DISK_PATH}"
if ((disk_path_was_set == 0)); then
  find "${LMCACHE_DISK_PATH}" -mindepth 1 -delete
fi

cat >"${LMCACHE_CONFIG_FILE}" <<EOF
chunk_size: ${LMCACHE_CHUNK_SIZE:-256}
local_cpu: true
max_local_cpu_size: ${LMCACHE_CPU_SIZE_GB:-0.10}
local_disk: ${LMCACHE_DISK_PATH}
max_local_disk_size: ${LMCACHE_DISK_SIZE_GB:-1.0}
cache_policy: LRU
extra_config:
  disk_cache_mode: eviction
  disk_io_threads: 2
EOF

export LMCACHE_CONFIG_FILE
export PYTHONPATH="${REPO_ROOT}/vllm:${REPO_ROOT}/LMCache${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -z "${KV_TRANSFER_CONFIG:-}" ]]; then
  KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
fi
export KV_TRANSFER_CONFIG

BACKENDS="${BACKENDS:-FORK_ATTN}" \
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}" \
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-1.7b-lmcache-smoke}" \
PREFIX_TOKENS="${PREFIX_TOKENS:-2048}" \
BRANCHES="${BRANCHES:-4}" \
CASE_COUNT="${CASE_COUNT:-1}" \
CONCURRENCY="${CONCURRENCY:-4}" \
OUTPUT_TOKENS="${OUTPUT_TOKENS:-16}" \
COMMON_ANALYSIS_TOKENS="${COMMON_ANALYSIS_TOKENS:-16}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}" \
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}" \
ENFORCE_EAGER="${ENFORCE_EAGER:-1}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
"${BENCHMARK_DIR}/scripts/run_vllm_benchmark.sh"

disk_files="$(find "${LMCACHE_DISK_PATH}" -type f -name '*.pt' | wc -l)"
disk_bytes="$(find "${LMCACHE_DISK_PATH}" -type f -name '*.pt' -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')"
server_log="${OUTPUT_ROOT}/fork_attn/vllm_server.log"

if ! grep -Eq "disk_cache_mode.*eviction" "${server_log}"; then
  echo "LMCache tiered smoke failed: eviction mode was not loaded." >&2
  tail -n 100 "${server_log}" >&2
  exit 1
fi

if grep -Eq "Failed to write key|Failed to submit disk write|Traceback" "${server_log}"; then
  echo "LMCache tiered smoke failed: the server reported a storage error." >&2
  tail -n 100 "${server_log}" >&2
  exit 1
fi

if ((disk_files == 0)); then
  echo "LMCache tiered smoke failed: no KV chunks were demoted to disk." >&2
  [[ -f "${server_log}" ]] && tail -n 100 "${server_log}" >&2
  exit 1
fi

cat >"${OUTPUT_ROOT}/smoke_summary.txt" <<EOF
status=passed
disk_path=${LMCACHE_DISK_PATH}
disk_files=${disk_files}
disk_bytes=${disk_bytes}
config=${LMCACHE_CONFIG_FILE}
EOF

echo "LMCache tiered smoke passed: ${disk_files} files, ${disk_bytes} bytes."
echo "Summary: ${OUTPUT_ROOT}/smoke_summary.txt"
