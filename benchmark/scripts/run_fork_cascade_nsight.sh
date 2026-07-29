#!/usr/bin/env bash
set -Eeuo pipefail

# Capture matched, UI-openable Nsight Systems reports for ordinary
# FlashAttention, native vLLM Cascade Attention, and ForkAttention. Each arm
# runs in a fresh server process and writes to its own subdirectory.

BENCHMARK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-results/fork_cascade_nsys}"

profile_names=(flash_attention cascade_attention fork_attention)
attention_backends=(FLASH_ATTN FLASH_ATTN FORK_ATTN)
enable_cascade=(0 1 1)
for index in "${!profile_names[@]}"; do
  profile_name="${profile_names[index]}"
  backend="${attention_backends[index]}"
  cascade="${enable_cascade[index]}"
  echo "Capturing ${profile_name}: backend=${backend} cascade=${cascade}"
  PROFILE_NAME="${profile_name}" \
  NSYS_OUTPUT_NAME="${profile_name}" \
  ATTENTION_BACKEND="${backend}" \
  ENABLE_CASCADE_ATTN="${cascade}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
    "${BENCHMARK_DIR}/scripts/run_fork_attention_nsight.sh"
done

echo "Nsight Systems UI reports are under ${BENCHMARK_DIR}/${OUTPUT_DIR}/"
