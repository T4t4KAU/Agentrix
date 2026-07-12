#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
RUN_SCRIPT="${BENCHMARK_DIR}/scripts/run_vllm_benchmark.sh"
BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}"
OUTPUT_ROOT="${OUTPUT_DIR:-results/offload_backend_comparison}"
CPU_SIZE_GB="${CPU_SIZE_GB:-0.5}"
DISK_SIZE_GB="${DISK_SIZE_GB:-2.0}"

export PYTHONPATH="${RUNTIME_PYTHONPATH:+${RUNTIME_PYTHONPATH}:}${REPO_ROOT}/vllm:${REPO_ROOT}/LMCache${PYTHONPATH:+:${PYTHONPATH}}"
cd "${BENCHMARK_DIR}"

cpu_bytes="$(${BENCHMARK_PYTHON} - "${CPU_SIZE_GB}" <<'PY'
import sys

print(int(float(sys.argv[1]) * 1024**3))
PY
)"

fork_native_config="$(${BENCHMARK_PYTHON} - "${cpu_bytes}" <<'PY'
import json
import sys

print(json.dumps({
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
        "cpu_bytes_to_use": int(sys.argv[1]),
        "fanout_offload": True,
        "fanout_profile": True,
        "fanout_allow_hot_prefix_backup": True,
    },
}))
PY
)"
flash_native_config="$(${BENCHMARK_PYTHON} - "${cpu_bytes}" <<'PY'
import json
import sys

print(json.dumps({
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
        "cpu_bytes_to_use": int(sys.argv[1]),
        "fanout_offload": False,
        "eviction_policy": "lru",
    },
}))
PY
)"
lmcache_config='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_load_failure_policy":"recompute"}'

write_lmcache_config() {
  local name="$1"
  local with_disk="$2"
  local cache_policy="$3"
  local variant_root="${BENCHMARK_DIR}/${OUTPUT_ROOT}/${name}"
  local config_path="${variant_root}/lmcache.yaml"
  mkdir -p "${variant_root}"
  {
    echo "chunk_size: ${LMCACHE_CHUNK_SIZE:-256}"
    echo "local_cpu: true"
    echo "max_local_cpu_size: ${CPU_SIZE_GB}"
    echo "cache_policy: ${cache_policy}"
    if [[ "${with_disk}" == "1" ]]; then
      local disk_path="${variant_root}/lmcache_disk"
      mkdir -p "${disk_path}"
      find "${disk_path}" -mindepth 1 -delete
      echo "local_disk: ${disk_path}"
      echo "max_local_disk_size: ${DISK_SIZE_GB}"
      echo "extra_config:"
      echo "  disk_cache_mode: eviction"
      echo "  disk_io_threads: 2"
    fi
  } >"${config_path}"
  printf '%s\n' "${config_path}"
}

write_storage_summary() {
  local name="$1"
  local disk_path="${BENCHMARK_DIR}/${OUTPUT_ROOT}/${name}/lmcache_disk"
  local disk_files=0
  local disk_bytes=0
  if [[ -d "${disk_path}" ]]; then
    disk_files="$(find "${disk_path}" -type f -name '*.pt' | wc -l)"
    disk_bytes="$(find "${disk_path}" -type f -name '*.pt' -printf '%s\n' | awk '{sum += $1} END {print sum + 0}')"
  fi
  cat >"${BENCHMARK_DIR}/${OUTPUT_ROOT}/${name}/storage_summary.txt" <<EOF
cpu_size_gib=${CPU_SIZE_GB}
disk_size_gib=${DISK_SIZE_GB}
disk_files=${disk_files}
disk_bytes=${disk_bytes}
EOF
}

run_variant() {
  local name="$1"
  local backend="$2"
  local kv_transfer_config="$3"
  local lmcache_config_file="${4:-}"
  echo "Running offload variant ${name}"
  LMCACHE_CONFIG_FILE="${lmcache_config_file}" \
  KV_TRANSFER_CONFIG="${kv_transfer_config}" \
  BACKENDS="${backend}" \
  MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}" \
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-1.7b-offload-comparison}" \
  PREFIX_TOKENS="${PREFIX_TOKENS:-4096}" \
  BRANCHES="${BRANCHES:-8}" \
  CASE_COUNT="${CASE_COUNT:-4}" \
  SAMPLE_COUNT="${SAMPLE_COUNT:-${CASE_COUNT:-4}}" \
  CONCURRENCY="${CONCURRENCY:-32}" \
  OUTPUT_TOKENS="${OUTPUT_TOKENS:-16}" \
  COMMON_ANALYSIS_TOKENS="${COMMON_ANALYSIS_TOKENS:-16}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}" \
  ENFORCE_EAGER="${ENFORCE_EAGER:-1}" \
  OUTPUT_DIR="${OUTPUT_ROOT}/${name}" \
  "${RUN_SCRIPT}"
  write_storage_summary "${name}"
}

default_cpu_config="$(write_lmcache_config lmcache_default_cpu 0 LRU)"
cpu_config="$(write_lmcache_config lmcache_cpu 0 FORK_AWARE)"
tiered_config="$(write_lmcache_config lmcache_tiered 1 FORK_AWARE)"

run_variant no_offload FORK_ATTN ""
run_variant native_cpu FORK_ATTN "${fork_native_config}"
run_variant lmcache_default_cpu FORK_ATTN "${lmcache_config}" "${default_cpu_config}"
run_variant lmcache_cpu FORK_ATTN "${lmcache_config}" "${cpu_config}"
run_variant lmcache_tiered FORK_ATTN "${lmcache_config}" "${tiered_config}"
run_variant flash_no_offload FLASH_ATTN ""
run_variant flash_native_cpu FLASH_ATTN "${flash_native_config}"

"${BENCHMARK_PYTHON}" "${BENCHMARK_DIR}/src/offload_comparison_report.py" \
  "${BENCHMARK_DIR}/${OUTPUT_ROOT}"
