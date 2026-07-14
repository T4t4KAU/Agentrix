#!/usr/bin/env bash
set -Eeuo pipefail

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "${BENCHMARK_DIR}/.." && pwd)"
RUN_SCRIPT="${RUN_SCRIPT:-${BENCHMARK_DIR}/scripts/run_vllm_benchmark.sh}"
BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-${BENCHMARK_DIR}/.venv/bin/python}"

MODE="${MODE:-single_gpu}"
if [[ -z "${EXPERIMENT_PROFILE:-}" ]]; then
  if [[ "${MODE}" == "single_gpu" ]]; then
    EXPERIMENT_PROFILE="fanout_validated"
  else
    EXPERIMENT_PROFILE="legacy"
  fi
fi
if [[ "${EXPERIMENT_PROFILE}" != "fanout_validated" \
  && "${EXPERIMENT_PROFILE}" != "offload_validated" \
  && "${EXPERIMENT_PROFILE}" != "legacy" ]]; then
  echo "EXPERIMENT_PROFILE must be fanout_validated, offload_validated, or legacy." >&2
  exit 1
fi
if [[ "${EXPERIMENT_PROFILE}" == "offload_validated" \
  && "${MODE}" != "single_gpu" ]]; then
  echo "EXPERIMENT_PROFILE=offload_validated requires MODE=single_gpu." >&2
  exit 1
fi
if [[ "${MODE}" == "single_gpu" \
  && "${EXPERIMENT_PROFILE}" == "fanout_validated" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT:-results/main_experiment_v2}"
  OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"
  CASE_COUNT="${CASE_COUNT:-1}"
  BRANCH_ORDER="${BRANCH_ORDER:-case_major}"
  WARM_SHARED_PREFIX="${WARM_SHARED_PREFIX:-1}"
  VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH="${VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH:-1}"
elif [[ "${MODE}" == "single_gpu" \
  && "${EXPERIMENT_PROFILE}" == "offload_validated" ]]; then
  OUTPUT_ROOT="${OUTPUT_ROOT:-results/main_experiment_offload_v2}"
  OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"
  CASE_COUNT="${CASE_COUNT:-4}"
  BRANCH_ORDER="${BRANCH_ORDER:-case_major}"
  WARM_SHARED_PREFIX="${WARM_SHARED_PREFIX:-0}"
  VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH="${VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH:-1}"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-results/main_experiment}"
  OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}"
  CASE_COUNT="${CASE_COUNT:-4}"
  BRANCH_ORDER="${BRANCH_ORDER:-round_robin}"
  WARM_SHARED_PREFIX="${WARM_SHARED_PREFIX:-0}"
  VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH="${VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH:-0}"
fi
DATASETS="${DATASETS:-swebench,agencybench,agentboard,appworld}"
SUFFIX_MEAN="${SUFFIX_MEAN:-256}"
COMMON_ANALYSIS_TOKENS="${COMMON_ANALYSIS_TOKENS:-64}"
MAX_DATASET_RECORDS="${MAX_DATASET_RECORDS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
OFFLOAD_CPU_GIB="${OFFLOAD_CPU_GIB:-8}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
TELEMETRY_INTERVAL_SECONDS="${TELEMETRY_INTERVAL_SECONDS:-0.5}"
RUN_RETRIES="${RUN_RETRIES:-2}"
RUN_RETRY_DELAY_SECONDS="${RUN_RETRY_DELAY_SECONDS:-10}"
NUM_GPU_BLOCKS_OVERRIDE="${NUM_GPU_BLOCKS_OVERRIDE:-}"
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
FANOUT_ADMISSION_WINDOW="${FANOUT_ADMISSION_WINDOW:-16}"
FANOUT_PROFILE="${FANOUT_PROFILE:-0}"

AGENTRIX_GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf unknown)"
VLLM_GIT_COMMIT="$(git -C "${REPO_ROOT}/vllm" rev-parse HEAD 2>/dev/null || printf unknown)"
AGENTRIX_GIT_DIRTY=0
VLLM_GIT_DIRTY=0
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null)" ]]; then
  AGENTRIX_GIT_DIRTY=1
fi
if [[ -n "$(git -C "${REPO_ROOT}/vllm" status --porcelain 2>/dev/null)" ]]; then
  VLLM_GIT_DIRTY=1
fi

if ! [[ "${RUN_RETRIES}" =~ ^[0-9]+$ ]]; then
  echo "RUN_RETRIES must be a non-negative integer." >&2
  exit 1
fi
if ! [[ "${FANOUT_ADMISSION_WINDOW}" =~ ^[0-9]+$ ]]; then
  echo "FANOUT_ADMISSION_WINDOW must be a non-negative integer." >&2
  exit 1
fi
if [[ "${FANOUT_PROFILE}" != "0" && "${FANOUT_PROFILE}" != "1" ]]; then
  echo "FANOUT_PROFILE must be 0 or 1." >&2
  exit 1
fi
if ! [[ "${MAX_DATASET_RECORDS}" =~ ^[0-9]+$ ]]; then
  echo "MAX_DATASET_RECORDS must be a non-negative integer." >&2
  exit 1
fi
if [[ "${VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH}" != "0" \
  && "${VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH}" != "1" ]]; then
  echo "VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH must be 0 or 1." >&2
  exit 1
fi
if [[ "${WARM_SHARED_PREFIX}" != "0" && "${WARM_SHARED_PREFIX}" != "1" ]]; then
  echo "WARM_SHARED_PREFIX must be 0 or 1." >&2
  exit 1
fi

FULL_DATASET=0
SAMPLE_COUNT="${MAX_DATASET_RECORDS}"
if ((MAX_DATASET_RECORDS == 0)); then
  FULL_DATASET=1
  SAMPLE_COUNT="${CASE_COUNT}"
fi

case "${MODE}" in
  single_gpu)
    MODEL_SPECS="${MODEL_SPECS:-qwen3-1.7b|Qwen/Qwen3-1.7B;llama3.2-1b|meta-llama/Llama-3.2-1B-Instruct}"
    PREFIX_LENGTHS="${PREFIX_LENGTHS:-8192,16384}"
    BRANCH_COUNTS="${BRANCH_COUNTS:-8,16}"
    if [[ "${EXPERIMENT_PROFILE}" == "offload_validated" ]]; then
      VARIANT_SPECS="${VARIANT_SPECS:-flash_ordinary_offload|FLASH_ATTN|ordinary|0;fork_ordinary_offload|FORK_ATTN|ordinary|0;fork_scheduled_ordinary_offload|FORK_ATTN|ordinary|0;fork_optimized_offload|FORK_ATTN|optimized|0}"
    elif [[ "${EXPERIMENT_PROFILE}" == "fanout_validated" ]]; then
      VARIANT_SPECS="${VARIANT_SPECS:-flash_no_offload|FLASH_ATTN|none|0;fork_no_offload|FORK_ATTN|none|0}"
    else
      VARIANT_SPECS="${VARIANT_SPECS:-flash_no_offload|FLASH_ATTN|none|0;fork_no_offload|FORK_ATTN|none|0;flash_ordinary_offload|FLASH_ATTN|ordinary|0;fork_ordinary_offload|FORK_ATTN|ordinary|0;fork_optimized_offload|FORK_ATTN|optimized|0}"
    fi
    DP_REPLICAS="${DP_REPLICAS:-1}"
    TP_SIZE="${TP_SIZE:-1}"
    GPU_IDS="${GPU_IDS:-0}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
    ;;
  dp)
    MODEL_SPECS="${MODEL_SPECS:-qwen3-8b|Qwen/Qwen3-8B}"
    PREFIX_LENGTHS="${PREFIX_LENGTHS:-8192,16384}"
    BRANCH_COUNTS="${BRANCH_COUNTS:-8,16,32}"
    VARIANT_SPECS="${VARIANT_SPECS:-flash_dp|FLASH_ATTN|none|0;fork_dp|FORK_ATTN|none|0;fork_prefix_aware_dp|FORK_ATTN|none|1}"
    DP_REPLICAS="${DP_REPLICAS:-2}"
    TP_SIZE="${TP_SIZE:-1}"
    GPU_IDS="${GPU_IDS:-0,1}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
    ;;
  tp_accuracy)
    MODEL_SPECS="${MODEL_SPECS:-qwen3-14b|Qwen/Qwen3-14B}"
    PREFIX_LENGTHS="${PREFIX_LENGTHS:-8192,16384}"
    BRANCH_COUNTS="${BRANCH_COUNTS:-8,16,32}"
    VARIANT_SPECS="${VARIANT_SPECS:-flash_tp|FLASH_ATTN|none|0;fork_tp_run1|FORK_ATTN|none|0;fork_tp_run2|FORK_ATTN|none|0}"
    DP_REPLICAS="${DP_REPLICAS:-1}"
    TP_SIZE="${TP_SIZE:-2}"
    GPU_IDS="${GPU_IDS:-0,1}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
    ;;
  *)
    echo "MODE must be single_gpu, dp, or tp_accuracy." >&2
    exit 1
    ;;
esac

cpu_bytes="$(${BENCHMARK_PYTHON} - "${OFFLOAD_CPU_GIB}" <<'PY'
import sys

print(int(float(sys.argv[1]) * 1024**3))
PY
)"
ordinary_offload="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":${cpu_bytes},\"fanout_offload\":false,\"fanout_admission_window\":0,\"fanout_preemption_enabled\":false,\"fanout_gpu_hotset_enabled\":false,\"eviction_policy\":\"lru\"}}"
fanout_profile_json=false
if [[ "${FANOUT_PROFILE}" == "1" ]]; then
  fanout_profile_json=true
fi
optimized_offload="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":${cpu_bytes},\"fanout_offload\":true,\"fanout_profile\":${fanout_profile_json},\"fanout_allow_hot_prefix_backup\":true,\"eviction_policy\":\"lru\"}}"

write_manifest() {
  local path="$1"
  local model_name="$2"
  local dataset="$3"
  local prefix_tokens="$4"
  local branches="$5"
  local variant="$6"
  local backend="$7"
  local offload="$8"
  local prefix_aware_policy="$9"
  local fanout_admission_window="${10}"
  mkdir -p "$(dirname -- "${path}")"
  "${BENCHMARK_PYTHON}" - \
    "${path}" "${MODE}" "${model_name}" "${dataset}" \
    "${prefix_tokens}" "${branches}" "${variant}" "${backend}" "${offload}" \
    "${DP_REPLICAS}" "${TP_SIZE}" \
    "${AGENTRIX_GIT_COMMIT}" "${AGENTRIX_GIT_DIRTY}" \
    "${VLLM_GIT_COMMIT}" "${VLLM_GIT_DIRTY}" \
    "${NUM_GPU_BLOCKS_OVERRIDE}" "${VLLM_USE_FLASHINFER_SAMPLER}" \
    "${prefix_aware_policy}" "${fanout_admission_window}" \
    "${OFFLOAD_CPU_GIB}" "${MAX_DATASET_RECORDS}" "${FULL_DATASET}" \
    "${EXPERIMENT_PROFILE}" "${BRANCH_ORDER}" "${WARM_SHARED_PREFIX}" \
    "${OUTPUT_TOKENS}" "${CASE_COUNT}" \
    "${VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH}" <<'PY'
import json
import sys
from pathlib import Path

(
    path,
    mode,
    model_name,
    dataset,
    prefix_tokens,
    branches,
    variant,
    backend,
    offload,
    dp_replicas,
    tp_size,
    agentrix_git_commit,
    agentrix_git_dirty,
    vllm_git_commit,
    vllm_git_dirty,
    num_gpu_blocks_override,
    use_flashinfer_sampler,
    prefix_aware_policy,
    fanout_admission_window,
    offload_cpu_gib,
    max_dataset_records,
    full_dataset,
    experiment_profile,
    branch_order,
    warm_shared_prefix,
    output_tokens,
    case_count,
    enable_forest_cudagraph,
) = sys.argv[1:]
Path(path).write_text(
    json.dumps(
        {
            "mode": mode,
            "model_name": model_name,
            "dataset": dataset,
            "prefix_tokens": int(prefix_tokens),
            "branches": int(branches),
            "variant": variant,
            "attention_backend": backend,
            "offload": offload,
            "dp_replicas": int(dp_replicas),
            "tp_size": int(tp_size),
            "agentrix_git_commit": agentrix_git_commit,
            "agentrix_git_dirty": bool(int(agentrix_git_dirty)),
            "vllm_git_commit": vllm_git_commit,
            "vllm_git_dirty": bool(int(vllm_git_dirty)),
            "num_gpu_blocks_override": (
                int(num_gpu_blocks_override) if num_gpu_blocks_override else None
            ),
            "use_flashinfer_sampler": bool(int(use_flashinfer_sampler)),
            "prefix_aware_policy": bool(int(prefix_aware_policy)),
            "fanout_admission_window": int(fanout_admission_window),
            "offload_cpu_gib": float(offload_cpu_gib),
            "max_dataset_records": (
                int(max_dataset_records) if int(max_dataset_records) else None
            ),
            "full_dataset": bool(int(full_dataset)),
            "experiment_profile": experiment_profile,
            "branch_order": branch_order,
            "warm_shared_prefix": bool(int(warm_shared_prefix)),
            "output_tokens": int(output_tokens),
            "case_count": int(case_count),
            "enable_forest_cudagraph": bool(int(enable_forest_cudagraph)),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

run_variant() {
  local model_name="$1"
  local model_path="$2"
  local dataset="$3"
  local prefix_tokens="$4"
  local branches="$5"
  local variant="$6"
  local backend="$7"
  local offload="$8"
  local prefix_routing="$9"
  local prefix_aware_policy=0
  if [[ "${variant}" == "fork_optimized_offload" \
    || "${variant}" == "fork_scheduled_ordinary_offload" \
    || "${variant}" == "fork_prefix_aware_dp" \
    || ( "${variant}" == "fork_no_offload" \
      && "${MODE}" == "single_gpu" \
      && "${EXPERIMENT_PROFILE}" == "fanout_validated" ) ]]; then
    prefix_aware_policy=1
  fi
  local variant_fanout_window=0
  if [[ "${prefix_aware_policy}" == "1" ]]; then
    variant_fanout_window="${FANOUT_ADMISSION_WINDOW}"
  fi
  local run_root="${OUTPUT_ROOT}/${MODE}/${model_name}/${dataset}/p${prefix_tokens}_b${branches}/${variant}"
  local backend_name="${backend,,}"
  local result_path="${BENCHMARK_DIR}/${run_root}/${backend_name}/benchmark_results.csv"
  local kv_transfer_config=""

  if [[ -s "${result_path}" ]]; then
    echo "Skipping completed run: ${result_path}"
    if [[ ! -s "${BENCHMARK_DIR}/${run_root}/manifest.json" ]]; then
      write_manifest \
        "${BENCHMARK_DIR}/${run_root}/manifest.json" \
        "${model_name}" "${dataset}" "${prefix_tokens}" \
        "${branches}" "${variant}" "${backend}" "${offload}" \
        "${prefix_aware_policy}" "${variant_fanout_window}"
    fi
    return
  fi
  case "${offload}" in
    none) kv_transfer_config="" ;;
    ordinary) kv_transfer_config="${ordinary_offload}" ;;
    optimized) kv_transfer_config="${optimized_offload}" ;;
    *) echo "Unknown offload mode: ${offload}" >&2; exit 1 ;;
  esac

  local tokenizer_margin=$((prefix_tokens / 4 + 1024))
  local max_model_len=$((prefix_tokens + tokenizer_margin + SUFFIX_MEAN + OUTPUT_TOKENS + COMMON_ANALYSIS_TOKENS))
  local concurrency=$((branches * CASE_COUNT))
  echo "Running ${MODE}/${model_name}/${dataset}/p${prefix_tokens}_b${branches}/${variant}"
  local attempt=1
  local max_attempts=$((RUN_RETRIES + 1))
  while true; do
    if MODEL_PATH="${model_path}" \
      SERVED_MODEL_NAME="${model_name}" \
      DATASET="${dataset}" \
      FULL_DATASET="${FULL_DATASET}" \
      BACKENDS="${backend}" \
      PREFIX_TOKENS="${prefix_tokens}" \
      BRANCHES="${branches}" \
      BRANCH_GROUP_SIZE="${branches}" \
      BRANCH_ORDER="${BRANCH_ORDER}" \
      WARM_SHARED_PREFIX="${WARM_SHARED_PREFIX}" \
      CASE_COUNT="${CASE_COUNT}" \
      SAMPLE_COUNT="${SAMPLE_COUNT}" \
      CONCURRENCY="${concurrency}" \
      SUFFIX_DISTRIBUTION=lognormal \
      SUFFIX_MEAN="${SUFFIX_MEAN}" \
      OUTPUT_TOKENS="${OUTPUT_TOKENS}" \
      COMMON_ANALYSIS_TOKENS="${COMMON_ANALYSIS_TOKENS}" \
      MAX_MODEL_LEN="${max_model_len}" \
      MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
      GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
      ENFORCE_EAGER="${ENFORCE_EAGER}" \
      DP_DEPLOYMENT=internal \
      DP_REPLICAS="${DP_REPLICAS}" \
      DP_ROUTING=single \
      TP_SIZE="${TP_SIZE}" \
      GPU_IDS="${GPU_IDS}" \
      KV_TRANSFER_CONFIG="${kv_transfer_config}" \
      TELEMETRY_INTERVAL_SECONDS="${TELEMETRY_INTERVAL_SECONDS}" \
      VLLM_FORK_ATTN_ENABLE_FOREST=1 \
      VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH="${VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH}" \
      VLLM_FORK_ATTN_FANOUT_SCHEDULING_ENABLED="${prefix_aware_policy}" \
      VLLM_FORK_ATTN_FANOUT_ADMISSION_WINDOW="${variant_fanout_window}" \
      VLLM_FORK_ATTN_DP_PREFIX_ROUTING="${prefix_routing}" \
      VLLM_FORK_ATTN_DP_RELOAD_REBALANCE=0 \
      OUTPUT_DIR="${run_root}" \
      "${RUN_SCRIPT}"; then
      break
    fi

    local backend_dir="${BENCHMARK_DIR}/${run_root}/${backend_name}"
    if [[ -f "${backend_dir}/vllm_server.log" ]]; then
      cp "${backend_dir}/vllm_server.log" \
        "${backend_dir}/vllm_server.attempt${attempt}.log"
    fi
    if ((attempt >= max_attempts)); then
      echo "Run failed after ${max_attempts} attempts: ${run_root}" >&2
      return 1
    fi
    echo "Run attempt ${attempt}/${max_attempts} failed; retrying in ${RUN_RETRY_DELAY_SECONDS}s..." >&2
    attempt=$((attempt + 1))
    sleep "${RUN_RETRY_DELAY_SECONDS}"
  done

  write_manifest \
    "${BENCHMARK_DIR}/${run_root}/manifest.json" \
    "${model_name}" "${dataset}" "${prefix_tokens}" \
    "${branches}" "${variant}" "${backend}" "${offload}" \
    "${prefix_aware_policy}" "${variant_fanout_window}"
}

IFS=';' read -r -a model_specs <<<"${MODEL_SPECS}"
IFS=',' read -r -a datasets <<<"${DATASETS}"
IFS=',' read -r -a prefix_lengths <<<"${PREFIX_LENGTHS}"
IFS=',' read -r -a branch_counts <<<"${BRANCH_COUNTS}"
IFS=';' read -r -a variant_specs <<<"${VARIANT_SPECS}"

for model_spec in "${model_specs[@]}"; do
  IFS='|' read -r model_name model_path <<<"${model_spec}"
  for dataset in "${datasets[@]}"; do
    for prefix_tokens in "${prefix_lengths[@]}"; do
      for branches in "${branch_counts[@]}"; do
        for variant_spec in "${variant_specs[@]}"; do
          IFS='|' read -r variant backend offload prefix_routing <<<"${variant_spec}"
          run_variant \
            "${model_name}" "${model_path}" "${dataset}" "${prefix_tokens}" \
            "${branches}" "${variant}" "${backend}" "${offload}" \
            "${prefix_routing}"
        done

        if [[ "${MODE}" == "tp_accuracy" ]]; then
          matrix_root="${BENCHMARK_DIR}/${OUTPUT_ROOT}/${MODE}/${model_name}/${dataset}/p${prefix_tokens}_b${branches}"
          reference="${matrix_root}/flash_tp/flash_attn/raw_api_results.json"
          for candidate in fork_tp_run1 fork_tp_run2; do
            candidate_path="${matrix_root}/${candidate}/fork_attn/raw_api_results.json"
            if [[ -s "${reference}" && -s "${candidate_path}" ]]; then
              "${BENCHMARK_PYTHON}" "${BENCHMARK_DIR}/src/accuracy.py" \
                "${reference}" "${candidate_path}" \
                --output-dir "${matrix_root}/${candidate}/agreement_vs_flash"
            fi
          done
          fork_run1="${matrix_root}/fork_tp_run1/fork_attn/raw_api_results.json"
          fork_run2="${matrix_root}/fork_tp_run2/fork_attn/raw_api_results.json"
          if [[ -s "${fork_run1}" && -s "${fork_run2}" ]]; then
            "${BENCHMARK_PYTHON}" "${BENCHMARK_DIR}/src/accuracy.py" \
              "${fork_run1}" "${fork_run2}" \
              --output-dir \
              "${matrix_root}/fork_tp_run2/repeatability_vs_fork_run1"
          fi
        fi
      done
    done
  done
done

"${BENCHMARK_PYTHON}" "${BENCHMARK_DIR}/src/main_experiment_report.py" \
  "${BENCHMARK_DIR}/${OUTPUT_ROOT}/${MODE}" \
  --output "${BENCHMARK_DIR}/${OUTPUT_ROOT}/${MODE}/main_experiment_report.md"
