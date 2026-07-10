#!/usr/bin/env bash
set -Eeuo pipefail

# Strong shared-prefix end-to-end matrix for comparing FlashAttention and
# ForkAttention through the OpenAI-compatible vLLM API.
#
# Each case keeps one long root prefix and launches many concurrent branch
# requests. This is the workload shape where ForkAttention should have a real
# chance to amortize shared-prefix KV loads and QK/PV work.

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SCRIPT="${BENCHMARK_DIR}/scripts/run_vllm_benchmark.sh"

if [[ ! -x "${RUN_SCRIPT}" ]]; then
  echo "benchmark runner does not exist or is not executable: ${RUN_SCRIPT}" >&2
  exit 1
fi

MATRIX_OUTPUT_ROOT="${MATRIX_OUTPUT_ROOT:-results/fanout_matrix_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_MD="${BENCHMARK_DIR}/${MATRIX_OUTPUT_ROOT}/matrix_summary.md"

BACKENDS="${BACKENDS:-FLASH_ATTN,FORK_ATTN}"
DATASET="${DATASET:-swebench}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-0.6B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-0.6b-local}"
DTYPE="${DTYPE:-float16}"
SUFFIX_DISTRIBUTION="${SUFFIX_DISTRIBUTION:-equal}"
COMMON_ANALYSIS_TOKENS="${COMMON_ANALYSIS_TOKENS:-32}"
CASE_COUNT="${CASE_COUNT:-4}"
SAMPLE_COUNT="${SAMPLE_COUNT:-${CASE_COUNT}}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
BRANCH_ORDER="${BRANCH_ORDER:-round_robin}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
CASE_FILTER="${CASE_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"

# ForkAttention profiling is useful evidence: server logs will show whether
# the eager/graph forest path was selected or the request fell back to FA.
PROFILE_FORK="${PROFILE_FORK:-1}"
VLLM_FORK_ATTN_ENABLE_FOREST="${VLLM_FORK_ATTN_ENABLE_FOREST:-1}"
VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH="${VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH:-1}"
VLLM_FORK_ATTN_PREFIX_CHUNK_SIZE="${VLLM_FORK_ATTN_PREFIX_CHUNK_SIZE:-2048}"
VLLM_FORK_ATTN_FOREST_CTA_BUCKETS="${VLLM_FORK_ATTN_FOREST_CTA_BUCKETS:-64,128,256,384,512,768,1024}"

# case_id|prefix_tokens|branches|branch_group_size|suffix_mean|output_tokens|max_num_seqs|max_model_len|gpu_memory_utilization
CASES=(
  "p8k_b16_g16_s128_o128|8192|16|16|128|128|16|32768|0.70"
  "p16k_b32_g32_s128_o128|16384|32|32|128|128|32|32768|0.78"
  "p16k_b64_g32_s128_o128|16384|64|32|128|128|64|32768|0.82"
  "p24k_b32_g32_s128_o128|24576|32|32|128|128|32|32768|0.82"
  "p24k_b64_g32_s256_o128|24576|64|32|256|128|64|32768|0.86"
)

mkdir -p "$(dirname "${SUMMARY_MD}")"
cat >"${SUMMARY_MD}" <<EOF
# Fanout Matrix

Backends: \`${BACKENDS}\`

Dataset: \`${DATASET}\`

Model: \`${MODEL_PATH}\`

Fork profile logging: \`PROFILE_FORK=${PROFILE_FORK}\`

EOF

run_case() {
  local spec="$1"
  local case_id prefix_tokens branches branch_group_size suffix_mean output_tokens
  local max_num_seqs max_model_len gpu_memory_utilization

  IFS="|" read -r \
    case_id \
    prefix_tokens \
    branches \
    branch_group_size \
    suffix_mean \
    output_tokens \
    max_num_seqs \
    max_model_len \
    gpu_memory_utilization <<<"${spec}"

  if [[ -n "${CASE_FILTER}" && "${case_id}" != *"${CASE_FILTER}"* ]]; then
    echo "Skipping ${case_id} due to CASE_FILTER=${CASE_FILTER}"
    return 0
  fi

  local output_dir="${MATRIX_OUTPUT_ROOT}/${case_id}"
  local concurrency=$((branches * CASE_COUNT))

  echo
  echo "==> Running ${case_id}"
  echo "    prefix=${prefix_tokens} branches=${branches} group=${branch_group_size} suffix=${suffix_mean} output=${output_tokens}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    cat <<EOF
BACKENDS=${BACKENDS} MODEL_PATH=${MODEL_PATH} PREFIX_TOKENS=${prefix_tokens} BRANCHES=${branches} \\
BRANCH_GROUP_SIZE=${branch_group_size} SUFFIX_MEAN=${suffix_mean} OUTPUT_TOKENS=${output_tokens} \\
MAX_NUM_SEQS=${max_num_seqs} MAX_MODEL_LEN=${max_model_len} CONCURRENCY=${concurrency} \\
OUTPUT_DIR=${output_dir} ${RUN_SCRIPT}
EOF
    return 0
  fi

  BACKENDS="${BACKENDS}" \
  MODEL_PATH="${MODEL_PATH}" \
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME}" \
  DTYPE="${DTYPE}" \
  DATASET="${DATASET}" \
  SAMPLE_INDEX="${SAMPLE_INDEX}" \
  SAMPLE_COUNT="${SAMPLE_COUNT}" \
  CASE_COUNT="${CASE_COUNT}" \
  PREFIX_TOKENS="${prefix_tokens}" \
  BRANCHES="${branches}" \
  BRANCH_GROUP_SIZE="${branch_group_size}" \
  BRANCH_ORDER="${BRANCH_ORDER}" \
  CONCURRENCY="${concurrency}" \
  SUFFIX_DISTRIBUTION="${SUFFIX_DISTRIBUTION}" \
  SUFFIX_MEAN="${suffix_mean}" \
  OUTPUT_TOKENS="${output_tokens}" \
  COMMON_ANALYSIS_TOKENS="${COMMON_ANALYSIS_TOKENS}" \
  MAX_NUM_SEQS="${max_num_seqs}" \
  MAX_MODEL_LEN="${max_model_len}" \
  GPU_MEMORY_UTILIZATION="${gpu_memory_utilization}" \
  STARTUP_TIMEOUT="${STARTUP_TIMEOUT}" \
  ENFORCE_EAGER="${ENFORCE_EAGER}" \
  OUTPUT_DIR="${output_dir}" \
  PROFILE_FORK="${PROFILE_FORK}" \
  VLLM_FORK_ATTN_ENABLE_FOREST="${VLLM_FORK_ATTN_ENABLE_FOREST}" \
  VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH="${VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH}" \
  VLLM_FORK_ATTN_PREFIX_CHUNK_SIZE="${VLLM_FORK_ATTN_PREFIX_CHUNK_SIZE}" \
  VLLM_FORK_ATTN_FOREST_CTA_BUCKETS="${VLLM_FORK_ATTN_FOREST_CTA_BUCKETS}" \
  "${RUN_SCRIPT}"

  {
    echo
    echo "## ${case_id}"
    echo
    if [[ -f "${BENCHMARK_DIR}/${output_dir}/backend_comparison.md" ]]; then
      sed -n '1,120p' "${BENCHMARK_DIR}/${output_dir}/backend_comparison.md"
    else
      echo "No backend comparison produced."
    fi
    echo
    echo "Server logs:"
    for backend_dir in "${BENCHMARK_DIR}/${output_dir}"/*; do
      [[ -d "${backend_dir}" ]] || continue
      echo "- \`${backend_dir#${BENCHMARK_DIR}/}/vllm_server.log\`"
    done
  } >>"${SUMMARY_MD}"
}

for spec in "${CASES[@]}"; do
  run_case "${spec}"
done

echo
echo "Fanout matrix complete: ${BENCHMARK_DIR}/${MATRIX_OUTPUT_ROOT}"
echo "Summary: ${SUMMARY_MD}"
