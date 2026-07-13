# Main Experiment: Shared-Prefix Performance and Accuracy

## Scope

This document is the canonical record for the Agentrix main experiment. The
matrix uses all four bundled datasets: SWE-bench Verified, AgencyBench,
AgentBoard, and AppWorld. Every workload expands each dataset prompt into a
long common prefix followed by multiple branch suffixes.

The experimental KV reload rebalance feature is disabled throughout. The DP
comparison isolates the stable prefix-aware routing path.

## Experiment Groups

### Single GPU

Models:

- Qwen3-1.7B
- Llama-3.2-1B

Variants:

- FlashAttention without offload
- ForkAttention without offload
- FlashAttention with ordinary LRU CPU offload
- ForkAttention with ordinary LRU CPU offload
- ForkAttention with prefix-aware optimized CPU offload

Both offload variants use vLLM's native `OffloadingConnector`, identical CPU
capacity, and LRU eviction. The ordinary variant disables fanout admission,
preemption, GPU hotset protection, and connector planning. The optimized
variant enables that complete prefix-aware policy. LMCache and disk storage
are excluded from this matrix.

The default host matrix uses 4K and 8K prefixes with 8 and 16 branches. Longer
settings can be selected through `PREFIX_LENGTHS` and `BRANCH_COUNTS`.

### Data Parallel

Model: Qwen3-8B. The default server matrix uses 8K and 16K prefixes with 8, 16,
and 32 branches on two internal DP ranks.

Variants:

- FlashAttention with ordinary DP and no offload
- ForkAttention with ordinary DP and no offload
- ForkAttention with prefix-aware DP and no offload

Ordinary internal DP and prefix-aware DP are both routed by vLLM. The ordinary
Fork variant disables the fanout scheduler, while the prefix-aware variant
enables it together with prefix-affinity routing. The client does not force a
data-parallel rank. Forced rank headers remain exclusive to the separate
experimental skew/reload benchmark.

### Tensor Parallel Accuracy Guardrail

Model: Qwen3-14B with TP=2 and no offload.

Variants:

- FlashAttention reference
- ForkAttention run 1
- ForkAttention run 2

The two ForkAttention runs measure both backend agreement and repeatability.
Accuracy in this document means normalized exact match, token F1, and text
similarity against greedy FlashAttention output. It is not task success in an
interactive SWE-bench, AgentBoard, or AppWorld environment.

Fork run 2 is also compared directly with Fork run 1. Every run manifest
records the Agentrix and vLLM Git commits, whether either worktree was dirty,
and the sampler backend. The matrix fixes `VLLM_USE_FLASHINFER_SAMPLER=0` so
runtime sampler JIT is not a hidden variable in the attention comparison.

## Recorded Metrics

- Logical KV read volume and reduction relative to branch-local attention
- Sampled GPU KV cache usage and configured GPU KV capacity
- Sampled CPU offload-cache occupancy and peak total physical KV occupancy
- KV offload load/store traffic
- Request, input-token, output-token, and total-token throughput
- TTFT and TPOT mean, P50, P95, and P99
- End-to-end request latency mean, P50, P95, and P99
- GPU compute utilization from NVIDIA `utilization.gpu`
- Memory bandwidth activity proxy from NVIDIA `utilization.memory`
- FlashAttention-to-ForkAttention output agreement for the TP guardrail

Logical KV read reduction estimates repeated shared-prefix reads avoided by
ForkAttention; it does not claim that vLLM stores duplicate physical prefix
blocks. Physical GPU KV reduction is reported separately from sampled vLLM
occupancy and is compared with the named baseline for each variant.

`utilization.memory` measures memory-controller activity. It is not a direct
measurement of HBM bandwidth in GB/s.

## Reproduction

Run from the `benchmark` directory and provide machine-specific model paths
through `MODEL_SPECS` rather than editing tracked scripts:

```bash
MODE=single_gpu MODEL_SPECS='qwen3-1.7b|/path/to/Qwen3-1.7B;llama3.2-1b|/path/to/Llama-3.2-1B' \
  ./scripts/run_main_experiment.sh

MODE=dp MODEL_SPECS='qwen3-8b|/path/to/Qwen3-8B' \
  ./scripts/run_main_experiment.sh

MODE=tp_accuracy MODEL_SPECS='qwen3-14b|/path/to/Qwen3-14B' \
  ./scripts/run_main_experiment.sh
```

The runner resumes by skipping completed result files. Generated Markdown and
CSV reports are stored under `benchmark/results/main_experiment/<mode>/`.

For policy comparisons, pin the same physical GPU KV capacity across variants
with `NUM_GPU_BLOCKS_OVERRIDE`. The host run recorded here uses 1700 blocks
(27,200 tokens at the default 16-token block size) after calibration against
the lowest observed backend capacity.

## Validation Status

The measurement path has passed a CUDA 13.0 host smoke test with Qwen3-1.7B.
Streaming TTFT/TPOT, latency percentiles, GPU telemetry, Prometheus KV gauges,
CPU offload occupancy, offload traffic, policy provenance, and server-profile
merging were populated for all five single-GPU variants. The smoke used a 1K
prefix, eight branches, a 160-block GPU KV capacity, and a 0.125 GiB CPU cache;
the offload variants reached 100% sampled CPU-cache occupancy. These values
validate instrumentation and pressure generation only. They are not main
experiment performance claims.

Full matrix results will be inserted below from the generated reports after
the host and server runs complete.

## Results

The fair full matrix is running. Earlier diagnostic checkpoints are excluded:
their ordinary Fork variants still inherited the default GPU hotset policy, and
one host checkpoint used backend-dependent GPU KV capacities. Both sets remain
archived under `benchmark/results/main_experiment_archive/` for debugging, but
they are not main experiment evidence.
