# Main Experiment: Shared-Prefix Performance and Accuracy

## Scope

This document is the canonical record for the completed high-value experiment.
The complete 304-run factorial launch procedure is documented separately in
[`main_experiment_matrix.md`](main_experiment_matrix.md).

The high-value run uses all four bundled datasets: SWE-bench Verified,
AgencyBench, AgentBoard, and AppWorld. It retains the complete Qwen3-1.7B
8K/16K by 8/16-branch grid on AgentBoard and AppWorld, adds cross-dataset
16K/16 stress replications, and validates Llama-3.2-1B on the same stress
shape. The complete two-rank Qwen3-8B DP grid covers 8K/16K prefixes and
8/16/32 branches. Qwen3-14B TP accuracy uses the maximum 16K/32 shape.

The Qwen single-GPU core grid consumes all 9 AgentBoard and 13 AppWorld
records. Its AgencyBench 8K/8 cell uses 32 records; all 16K/16 stress
replications use eight records. DP consumes all AgentBoard, AppWorld, and
AgencyBench records plus the first 32 SWE-bench records. TP accuracy uses the
first eight records per dataset. Every comparison within a cell uses identical
prompts, and reported cross-cell means are unweighted.

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

The default host matrix uses 8K and 16K prefixes with 8 and 16 branches. Other
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
The default `MAX_DATASET_RECORDS=32` and whether a run is uncapped are included
in its provenance manifest.

For policy comparisons, pin the same physical GPU KV capacity across variants
with `NUM_GPU_BLOCKS_OVERRIDE`. The host run recorded here uses 1700 blocks
(27,200 tokens at the default 16-token block size) after calibration against
the lowest observed backend capacity.

## Executed Coverage

| Group | Hardware | CUDA | Executed coverage |
|---|---|---|---|
| Single GPU | RTX 5070 12 GiB | 13.0 build | 75 runs: Qwen complete grid on AgentBoard/AppWorld; Qwen and Llama 16K/16 stress replications on four datasets |
| Data parallel | 2x RTX 5090 32 GiB | 12.8 | Complete 72-run Qwen3-8B grid |
| TP accuracy | 2x RTX 5090 32 GiB | 12.8 | 12 standard Qwen3-14B 16K/32 runs plus two Flash repeatability controls |

Single-GPU comparisons pin 1,700 GPU blocks and an 8 GiB CPU offload cache.
DP comparisons pin 3,852 blocks per rank. The reload-rebalance experiment is
disabled. Earlier diagnostic checkpoints with inherited hotset policy or
backend-dependent capacities are excluded.

The clean result manifests span Agentrix commits `a6db65e`, `26f63d9`, and
`c95508e`, which changed experiment orchestration, provenance, documentation,
or submodule registration. Every reported cell uses the same clean vLLM commit
`287304ad68ce`.

## Single-GPU Offload

The table compares optimized ForkAttention offload against ordinary
ForkAttention LRU offload. Positive throughput is an improvement; negative
TTFT and KV load values are reductions.

| Model | Dataset | Prefix | Branches | Throughput | TTFT P50 | KV load |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | AgentBoard | 8K | 8 | +57.2% | -54.8% | -75.5% |
| Qwen3-1.7B | AgentBoard | 8K | 16 | +84.6% | -67.1% | -87.3% |
| Qwen3-1.7B | AgentBoard | 16K | 8 | +88.9% | -62.7% | -82.1% |
| Qwen3-1.7B | AgentBoard | 16K | 16 | +129.4% | -70.8% | -85.1% |
| Qwen3-1.7B | AppWorld | 8K | 8 | +78.9% | -67.8% | -80.2% |
| Qwen3-1.7B | AppWorld | 8K | 16 | +102.3% | -71.3% | -86.6% |
| Qwen3-1.7B | AppWorld | 16K | 8 | +80.3% | -58.0% | -80.6% |
| Qwen3-1.7B | AppWorld | 16K | 16 | +128.6% | -73.5% | -84.9% |
| Qwen3-1.7B | AgencyBench | 8K | 8 | +78.0% | -60.8% | -82.1% |
| Qwen3-1.7B | AgencyBench | 16K | 16 | +150.6% | -70.6% | -86.1% |
| Qwen3-1.7B | SWE-bench | 16K | 16 | +141.7% | -70.5% | -85.5% |
| Llama-3.2-1B | AgentBoard | 16K | 16 | +77.9% | -55.5% | -86.0% |
| Llama-3.2-1B | AppWorld | 16K | 16 | +58.5% | -50.3% | -86.7% |
| Llama-3.2-1B | AgencyBench | 16K | 16 | +73.6% | -56.8% | -85.0% |
| Llama-3.2-1B | SWE-bench | 16K | 16 | +73.4% | -58.5% | -85.2% |

Across the 11 complete Qwen cells, the unweighted mean improvement is 101.9%
for output throughput and 66.2% for TTFT P50, while CPU-to-GPU KV load falls
83.3%. The sampled total physical GPU+CPU KV peak changes by only -0.5% on
average. This is expected: the policy changes residency and reload frequency,
not the amount of useful KV retained. ForkAttention's logical KV read reduction
is reported separately and does not imply duplicate physical vLLM blocks.
Across the four Llama stress cells, throughput rises 70.9%, TTFT P50 falls
55.3%, and KV load falls 85.7%, confirming that the offload result is not
specific to Qwen3.

## Data-Parallel Results

Prefix-aware DP is compared directly with ordinary ForkAttention DP across 24
dataset/shape cells.

| Scope | Throughput | Wins | TTFT P50 | Wins | Peak GPU KV | Wins |
|---|---:|---:|---:|---:|---:|---:|
| All cells | +26.5% | 22/24 | -56.2% | 24/24 | -17.7% | 20/24 |
| 8K prefix | +6.9% | 10/12 | -46.0% | 12/12 | -21.0% | 11/12 |
| 16K prefix | +46.1% | 12/12 | -66.4% | 12/12 | -14.4% | 9/12 |

The average logical KV read reduction is 93.8%. The 16K result is the primary
claim: every throughput cell improves, while 8K retains two small negative
throughput outliers. Lower sampled GPU compute and memory-controller activity
for prefix-aware DP reflect less redundant work; they should not be interpreted
as lower hardware capability.

At the maximum 16K/32 shape, throughput improves over ordinary Fork DP by
32.3% on AgencyBench, 31.0% on AgentBoard, 101.1% on AppWorld, and 20.9% on
SWE-bench. TTFT P50 falls by 64.9%, 63.9%, 74.7%, and 33.7%, respectively.

## TP Accuracy Guardrail

Each dataset cell compares 264 outputs: 256 branches and eight common-analysis
outputs. Values below are unweighted means over the four 16K/32 cells.

| Comparison | Exact match | Token F1 | Text similarity |
|---|---:|---:|---:|
| Fork run 1 vs Flash | 91.10% | 98.25% | 97.16% |
| Fork run 2 vs Flash | 91.76% | 98.60% | 97.60% |
| Fork run 2 vs Fork run 1 | 94.03% | 98.59% | 97.65% |

AgentBoard Flash repeatability is 92.80% exact match and 98.40% token F1,
which is comparable to Fork repeatability on that cell (91.29% and 98.10%).
SWE-bench is the strict-exact outlier: both Flash and Fork are internally
repeatable (98.48% and 98.86%), but Flash-to-Fork exact match is 85.61% while
token F1 remains 98.45%. The implementation therefore passes a strong
token-level agreement guardrail but is not bitwise or text-exact equivalent.
Because the bundled snapshots do not provide a common executable evaluator,
this experiment does not establish environment-level task accuracy.

## Artifacts

Detailed P50/P95/P99 latency, TPOT, throughput, compute utilization,
memory-controller utilization, physical/logical KV metrics, offload traffic,
and provenance are preserved in the committed report snapshots:

- [Single-GPU report](experiment_results/single_gpu.md) ([CSV](experiment_results/single_gpu.csv))
- [Data-parallel report](experiment_results/dp.md) ([CSV](experiment_results/dp.csv))
- [TP accuracy report](experiment_results/tp_accuracy.md) ([CSV](experiment_results/tp_accuracy.csv))

Request-level JSON, telemetry samples, and server logs remain in the ignored
`benchmark/results/main_experiment/` tree on the machines that ran each group.

The complete factorial commands and expected run counts are in
[`main_experiment_matrix.md`](main_experiment_matrix.md).
