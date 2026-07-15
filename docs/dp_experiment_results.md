# Capacity-Aware Data Parallel Experiment Results

## Objective

This document records only the current prefix-aware DP implementation at vLLM
commit `0cff6d857`. Results produced by the earlier routing policy have been
removed because its affinity ordering, cohort balancing, and short-prompt
behavior no longer represent the implementation.

The current comparison isolates routing behavior: both variants use the same
ForkAttention backend, model, two-rank vLLM deployment, CUDA Graph settings,
and physical KV capacity. CPU and disk KV offload are disabled.

## Environment

| Item | Value |
|---|---|
| Validation dates | 2026-07-14 and 2026-07-15 |
| GPUs | 2 x NVIDIA GeForce RTX 5090, 32 GiB each |
| CUDA toolkit | 12.8.93 |
| GPU architecture | SM120 |
| Model | Qwen3-8B, float16 |
| DP deployment | vLLM internal DP, two replicas |
| Attention | ForkAttention for both variants |
| GPU KV capacity | 3,852 blocks per rank |
| Maximum sequences | 64 per rank |
| CUDA Graphs | Prefix Forest CUDA Graphs enabled |
| Offloading | Disabled |

Each run contains four cases, 16 branches per case, concurrency 64, a
deterministic lognormal suffix distribution with mean 256, 256 branch output
tokens, 64 common-analysis tokens, and seed 2026. Ordinary DP and the final
prefix-aware router differ only in routing policy.

## Current Router Behavior

The corrected router:

- compares total prefix match depth first and uses active, live, or warm
  residency only as a tie-breaker at the same depth;
- balances a new long-prefix owner before applying prefix affinity;
- keeps a deep-prefix cohort together under a cumulative skew budget derived
  from matched prefix size, per-rank KV capacity, and `max_num_seqs`;
- relaxes work balance only by the amount of prefix recomputation avoided;
- sends prompts smaller than 20% of per-rank KV capacity through native
  ordinary DP without prefix hashing or an arrival wave; and
- uses a 10 ms arrival wave for eligible long-prefix requests.

The small-prompt bypass is part of the result, not an inactive configuration:
it prevents router overhead and unnecessary cohort delay when both ranks have
enough KV capacity.

## Warm8K Bypass Validation

Warm8K uses an exact warm prefix and case-major request order. It has three
same-session paired repetitions with alternating variant order.

| Variant | Branch tok/s runs | Median | Branch phase | TTFT P50 | TPOT P50 | Peak KV | Max waiting | Preemptions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fork ordinary DP | 2,142.3 / 2,124.5 / 2,014.0 | 2,124.5 | 7,711.8 ms | 1,002.3 ms | 24.73 ms | 76.40% | 14 | 0 |
| Final prefix-aware router | 2,210.3 / 2,116.7 / 2,021.3 | 2,116.7 | 7,740.2 ms | 1,003.4 ms | 23.82 ms | 76.53% | 14 | 0 |

All 72 requests per run take the native bypass, with zero arrival waves and
zero prefix-routing time. Relative to ordinary DP, the final policy changes
branch throughput by -0.37%, branch-phase time by +0.37%, TTFT P50 by +0.11%,
and TPOT P50 by -3.67%. The result is neutral within measured variance.

## Pressure16K Validation

Pressure16K uses no explicit warmup and a deterministic four-case shuffle.
Each variant has three repetitions.

| Variant | Branch tok/s runs | Median | Branch phase | TTFT P50 | TPOT P50 | Peak KV | Max waiting | Preemptions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fork ordinary DP | 443.6 / 385.8 / 454.3 | 443.6 | 36,937.9 ms | 17,253.2 ms | 21.66 ms | 88.28% | 56 | 0 |
| Final prefix-aware router | 1,677.2 / 1,689.4 / 1,716.0 | 1,689.4 | 9,698.4 ms | 1,432.5 ms | 31.32 ms | 75.19% | 2 | 0 |

Relative to ordinary DP, the final router:

- improves median branch throughput by 280.87%, or 3.81x;
- reduces branch-phase time by 73.74%;
- reduces TTFT P50 by 91.70%;
- lowers peak KV usage by 13.09 percentage points; and
- reduces the maximum waiting queue from 56 requests to two.

Every final run records 64/64 affinity routes, 64/64 cohort-locked routes,
four balanced bootstrap routes, a 34/34 rank split, peak KV usage between
75.14% and 75.50%, and zero preemptions.

The higher TPOT P50 does not contradict the throughput result. Ordinary DP
leaves most requests waiting before decode, which produces a 17.3-second TTFT
P50. The final router admits the shared cohort promptly and keeps both ranks
working, so aggregate throughput and completion time improve even though the
active decode interval per token is longer.

## Interpretation

The supported claim is deliberately scoped:

- For bypassed 8K traffic with sufficient KV capacity, the current router is
  approximately neutral and avoids imposing routing work.
- For shuffled 16K cohorts under memory pressure, preserving deep-prefix
  ownership prevents duplicated prefix residency and delivers a large,
  repeatable throughput and TTFT improvement.
- The result does not imply that prefix-aware routing accelerates every DP
  request. Its benefit depends on prefix depth, cohort visibility, and KV
  pressure.

The generated common-analysis text is not guaranteed to be byte-identical
across DP schedules even with temperature zero. Compared runs use identical
request counts, input/output token totals, length distribution, seed, model,
KV capacity, attention backend, and CUDA Graph settings. Performance claims
use measured timing rather than logical KV savings.

## Validation

- All current-router runs completed without preemption or request failure.
- The final pressure runs remained within a narrow 1,677.2-1,716.0 tok/s
  branch-throughput band.
- The validation combined 31 prefix-router tests with 22 targeted offload
  admission/planner tests; all 53 passed.
- DP routing and offload changes touch disjoint implementation files, and no
  offload connector was enabled in these runs.
- Shutdown-time `EngineDeadError` messages occur after all HTTP responses,
  metrics, and result files have been collected.

## Reproduction

Run from the repository root with portable model paths. The following commands
use only the two current comparison variants.

Warm8K:

```bash
cd benchmark

MODE=dp \
MODEL_SPECS='qwen3-8b|/path/to/Qwen3-8B' \
DATASETS=agencybench \
PREFIX_LENGTHS=8192 \
BRANCH_COUNTS=16 \
VARIANT_SPECS='fork_dp|FORK_ATTN|none|0;fork_prefix_aware_dp|FORK_ATTN|none|1' \
CASE_COUNT=4 \
MAX_DATASET_RECORDS=4 \
BRANCH_ORDER=case_major \
WARM_SHARED_PREFIX=1 \
SUFFIX_MEAN=256 \
OUTPUT_TOKENS=256 \
COMMON_ANALYSIS_TOKENS=64 \
MAX_NUM_SEQS=64 \
NUM_GPU_BLOCKS_OVERRIDE=3852 \
SEED=2026 \
PROFILE_FORK=1 \
FANOUT_ADMISSION_WINDOW=0 \
VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH=1 \
VLLM_FORK_ATTN_DP_ARRIVAL_WAVE_MS=10 \
OUTPUT_ROOT=results/dp_capacity_aware_warm8k \
./scripts/run_main_experiment.sh
```

Pressure16K changes only the prefix construction, order, and output root:

```bash
cd benchmark

MODE=dp \
MODEL_SPECS='qwen3-8b|/path/to/Qwen3-8B' \
DATASETS=agencybench \
PREFIX_LENGTHS=16384 \
BRANCH_COUNTS=16 \
VARIANT_SPECS='fork_dp|FORK_ATTN|none|0;fork_prefix_aware_dp|FORK_ATTN|none|1' \
CASE_COUNT=4 \
MAX_DATASET_RECORDS=4 \
BRANCH_ORDER=shuffle \
WARM_SHARED_PREFIX=0 \
SUFFIX_MEAN=256 \
OUTPUT_TOKENS=256 \
COMMON_ANALYSIS_TOKENS=64 \
MAX_NUM_SEQS=64 \
NUM_GPU_BLOCKS_OVERRIDE=3852 \
SEED=2026 \
PROFILE_FORK=1 \
FANOUT_ADMISSION_WINDOW=0 \
VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH=1 \
VLLM_FORK_ATTN_DP_ARRIVAL_WAVE_MS=10 \
OUTPUT_ROOT=results/dp_capacity_aware_pressure16k \
./scripts/run_main_experiment.sh
```

Use a new output root for each repetition and alternate variant order for the
paired Warm8K runs. Result paths remain repository-relative under
`benchmark/results/`.
