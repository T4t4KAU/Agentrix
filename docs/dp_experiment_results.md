# Capacity-Aware Data Parallel Experiment Results

## Objective

This document records the latest adaptive Prefix Forest CUDA Graph validation
of the current prefix-aware DP implementation. It uses vLLM commit
`e139c755b` with a server-side graph-capacity fix. Superseded measurements have
been removed.

The two ForkAttention variants isolate routing behavior by using the same
backend, model, two-rank vLLM deployment, CUDA Graph settings, and physical KV
capacity. The FlashAttention baseline uses the same workload and capacity.
CPU and disk KV offload are disabled.

## Environment

| Item | Value |
|---|---|
| Validation date | 2026-07-15 |
| GPUs | 2 x NVIDIA GeForce RTX 5090, 32 GiB each |
| CUDA toolkit | 12.8.93 |
| GPU architecture | SM120 |
| Model | Qwen3-8B, float16 |
| DP deployment | vLLM internal DP, two replicas |
| Attention | ForkAttention routing comparison; FlashAttention follow-up baseline |
| GPU KV capacity | 3,852 blocks per rank |
| Maximum sequences | 64 per rank |
| CUDA Graphs | Prefix Forest CUDA Graphs enabled |
| Offloading | Disabled |

Adaptive16K contains four cases with 16 branches each. Pressure32K/32 contains
two cases with 32 branches each. Both use concurrency 64, a deterministic
lognormal suffix distribution with mean 256, 256 branch output tokens, 64
common-analysis tokens, seed 2026, no explicit shared-prefix warmup, and
shuffled arrival order.

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

## Current Adaptive Forest Results

The current checkout's adaptive CTA splitting exposed a graph-capacity
regression: the dispatcher could select `forest-64` before adaptive planning
expanded the runtime forest beyond 64 CTAs. A server-side fix makes forest
bucket selection adaptive-aware and constrains the final metadata plan to the
selected CTA and split capacities. A 64-CTA workspace can retain the valid
base plan, while a larger workspace still retains the adaptive plan; the fix
does not disable adaptive execution globally.

The table uses measured branch-only output throughput.

| Scenario / variant | Branch tok/s runs | Median | Branch phase | TTFT P50 | TPOT P50 | Peak KV | Preemptions |
|---|---|---:|---:|---:|---:|---:|---:|
| Adaptive16K, Flash ordinary DP | 496.3 | 496.3 | 33,009.4 ms | 16,357.7 ms | 23.45 ms | 89.04% | 0 |
| Adaptive16K, Fork ordinary DP | 424.1 | 424.1 | 38,634.7 ms | 18,212.5 ms | 22.77 ms | 87.60% | 0 |
| Adaptive16K, Fork prefix-aware DP | 1,529.1 | 1,529.1 | 10,714.6 ms | 1,413.3 ms | 35.05 ms | 75.10% | 0 |
| Pressure32K/32, Flash ordinary DP | 394.7 / 212.6 | 303.7 | 59,284.8 ms | 32,462.9 ms | 25.47 ms | 59.01% | 0 |
| Pressure32K/32, Fork ordinary DP | 264.8 / 377.9 | 321.3 | 52,619.6 ms | 29,645.3 ms | 21.88 ms | 59.74% | 0 |
| Pressure32K/32, Fork prefix-aware DP | 1,570.1 / 1,611.9 | 1,591.0 | 10,299.8 ms | 2,153.2 ms | 26.82 ms | 72.62% | 0 |

Adaptive16K completed all three variants without another graph mismatch. In
that run, prefix-aware ForkAttention delivered 3.08x the branch throughput of
Flash ordinary DP and 3.61x that of Fork ordinary DP.

Pressure32K/32 requested 32,768 prefix tokens and measured 32,850 after chat
formatting. All 66 requests completed. The two prefix owners bootstrapped one
per rank, the remaining 64 requests used prefix affinity and cohort locking,
and the final rank allocation was exactly 33/33 in both prefix-aware runs.
Prefix-aware branch throughput stayed within 1,570.1-1,611.9 tok/s, while both
ordinary baselines varied substantially. Relative to ordinary ForkAttention,
the paired improvement ranged from 4.27x to 5.93x. The stable result is the
narrow prefix-aware band and balanced cohort placement; no single ratio against
the unstable baselines should be treated as a universal headline.

The 32K benchmark script conservatively requested `max_model_len=42560` because
it adds a 25% tokenizer margin. `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` allowed that
service configuration, but measured request lengths remained below Qwen3-8B's
declared 40,960-token position limit. No run used KV offload. The adaptive
forest fix was validated as an uncommitted server-side patch and must be
committed and synchronized before these exact results are reproducible from a
clean checkout.

## Interpretation

The supported claim is deliberately scoped:

- For the repaired Adaptive16K path, preserving deep-prefix ownership improves
  branch throughput and TTFT relative to both ordinary-DP baselines.
- For Pressure32K/32, prefix-aware placement keeps the two long-prefix cohorts
  balanced and produces a narrow throughput band despite substantial baseline
  variance.
- The result does not imply that prefix-aware routing accelerates every DP
  request. Its benefit depends on prefix depth, cohort visibility, and KV
  pressure.
- Pressure32K/32 is a two-repeat stress result rather than a broad workload
  average.

The generated common-analysis text is not guaranteed to be byte-identical
across DP schedules even with temperature zero. Compared runs use identical
request counts, input/output token totals, length distribution, seed, model,
KV capacity, attention backend, and CUDA Graph settings. Performance claims
use measured timing rather than logical KV savings.

## Validation

- All current-router runs completed without preemption or request failure.
- Pressure32K/32 prefix-aware branch throughput remained within a narrow
  1,570.1-1,611.9 tok/s band.
- The current adaptive forest validation passed 75 graph-dispatch, ForkAttention
  backend, GPU-workspace, and prefix-router tests. LangGraph was outside this
  validation scope.
- No offload connector was enabled in these runs.
