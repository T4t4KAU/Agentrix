# Capacity-Aware Data Parallel Experiment Results

## Objective

This document records the latest adaptive Prefix Forest CUDA Graph validation
of the current prefix-aware DP implementation. It uses vLLM commit
`e139c755b` with a server-side graph-capacity fix. Superseded measurements have
been removed.

FlashAttention with ordinary DP is the primary baseline. ForkAttention with
ordinary DP is an ablation that separates the attention-backend effect from
the prefix-aware routing effect. All variants use the same model, two-rank
deployment, workload, graph-enabled execution policy, and physical KV
capacity. FlashAttention uses standard vLLM CUDA Graphs; ForkAttention uses
Prefix Forest CUDA Graphs. CPU and disk KV offload are disabled.

## Environment

| Item | Value |
|---|---|
| Validation date | 2026-07-15 |
| GPUs | 2 x NVIDIA GeForce RTX 5090, 32 GiB each |
| CUDA toolkit | 12.8.93 |
| GPU architecture | SM120 |
| Model | Qwen3-8B, float16 |
| DP deployment | vLLM internal DP, two replicas |
| Attention | FlashAttention ordinary-DP baseline; ForkAttention ordinary and prefix-aware variants |
| GPU KV capacity | 3,852 blocks per rank |
| Maximum sequences | 64 per rank |
| CUDA Graphs | Standard vLLM graphs for FlashAttention; Prefix Forest graphs for ForkAttention |
| Offloading | Disabled |

Adaptive16K contains four cases with 16 branches each. Pressure32K/32 contains
two cases with 32 branches each. Both use concurrency 64, a deterministic
lognormal suffix distribution with mean 256, 256 branch output tokens, 64
common-analysis tokens, seed 2026, no explicit shared-prefix warmup, and
shuffled arrival order.

## Request Construction and Fairness Audit

The Pressure32K/32 request stream is constructed as follows:

1. The runner deterministically selects records 0 and 1 from the named bundled
   dataset; it does not select records based on measured performance. Each
   record is converted to a dataset-specific Agent prompt containing its real
   task, instructions, tools, demonstrations, or bug description.
2. Each source prompt is repeated with a fixed separator or truncated to
   exactly 32,768 harness-tokenizer tokens. The harness uses `o200k_base` for
   the local `qwen3-8b` served name. After the Qwen chat template and generated
   shared analysis are included, the measured shared prefix is roughly 32.8K
   tokens; complete branch inputs remain below the model's declared
   40,960-token limit.
3. Before fanout, each case issues one temperature-zero common-analysis request
   with a 64-token output limit. Its output is appended to that case's root to
   form the shared branch context. This is the natural bootstrap stage of the
   Agent workflow.
4. Each case then creates 32 branches. Suffix budgets use a seed-2026 lognormal
   distribution with sigma 0.75 and are rescaled to a mean of 256 tokens. The
   32 branches share a group seed for approximately half of each suffix budget
   and use branch-specific text for the remainder, producing a nested shared
   prefix followed by private work. Every branch requests at most 256 output
   tokens with temperature zero.
5. The 64 branch requests from both cases are shuffled with the fixed runner
   seed and submitted together with concurrency 64 and zero client-side arrival
   interval. The order and suffix budgets are identical for all three variants.
   The client uses one internal-DP endpoint and sends no rank-selection header,
   so rank placement is entirely server-controlled.

This is not a native dataset task-score evaluation. The dataset supplies real
Agent root content, while the harness deliberately controls prefix length and
constructs the multi-branch suffixes. That controlled transformation targets
Agentrix's long-prefix fanout design point and is applied identically to every
baseline; conclusions are limited to systems performance on that shape.

`WARM_SHARED_PREFIX=0` means the runner sends no additional one-token request
to warm the exact final shared branch context. It does not make the workload
fully cold: the required common-analysis request has already processed the
32K root before fanout. This staged bootstrap is identical across variants and
models an Agent that analyzes a root task before spawning branches.

Each variant starts from a fresh vLLM service, so no KV cache or CUDA Graph
state is carried from another variant. All variants pin 3,852 KV blocks per
rank, use `max_num_seqs=64`, disable CPU/disk offload, and complete the same
66 requests (two common-analysis plus 64 branch requests). Flash ordinary DP
and Fork ordinary DP use native internal-DP placement. Only the prefix-aware
variant enables server-side prefix routing; the request client is otherwise
unchanged.

The headline branch throughput is `16,384 generated branch tokens / measured
branch-phase wall time`. It excludes service startup, graph capture, and the
two common-analysis requests. TTFT and TPOT are measured from streamed branch
responses. This is therefore a fanout-stage systems result, not end-to-end
Agent task throughput. The raw report retains case-total and request-level
timings for audit.

The generated common-analysis bytes are not forced to be identical across
separately launched variants, even at temperature zero. For each dataset, the
record IDs, branch suffix templates, request counts, input-token totals,
output-token totals, suffix budgets, and arrival order are identical across
variants; the observed common-analysis wording may differ. This is a
limitation for exact output comparison, but it does not give the prefix-aware
variant a different length or request-count workload.

The cross-dataset run was launched from `benchmark/` with the following
settings; only the server-specific model path and output root are portable
substitutions:

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
MODE=dp \
MODEL_SPECS='qwen3-8b|/path/to/Qwen3-8B' \
DATASETS='agentboard,appworld,swebench' \
PREFIX_LENGTHS=32768 BRANCH_COUNTS=32 \
VARIANT_SPECS='flash_dp|FLASH_ATTN|none|0;fork_dp|FORK_ATTN|none|0;fork_prefix_aware_dp|FORK_ATTN|none|1' \
CASE_COUNT=2 MAX_DATASET_RECORDS=2 BRANCH_ORDER=shuffle \
WARM_SHARED_PREFIX=0 SUFFIX_MEAN=256 OUTPUT_TOKENS=256 \
COMMON_ANALYSIS_TOKENS=64 MAX_NUM_SEQS=64 \
NUM_GPU_BLOCKS_OVERRIDE=3852 GPU_MEMORY_UTILIZATION=0.70 \
DP_REPLICAS=2 TP_SIZE=1 GPU_IDS=0,1 \
FANOUT_ADMISSION_WINDOW=0 \
VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH=1 \
VLLM_FORK_ATTN_DP_ARRIVAL_WAVE_MS=10 SEED=2026 \
OUTPUT_ROOT=results/dp_pressure32k_b32_crossdatasets_r1 \
./scripts/run_main_experiment.sh
```

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
ordinary baselines varied substantially. Relative to the primary Flash
ordinary-DP baseline, the paired improvement was 3.98x and 7.58x. Relative to
the Fork ordinary-DP ablation, it was 5.93x and 4.27x. The stable result is the
narrow prefix-aware band and balanced cohort placement; no single ratio against
the unstable baselines should be treated as a universal headline.

## Cross-Dataset Pressure32K/32 Validation

The same Pressure32K/32 configuration was then run once on each of the other
three bundled datasets. The source record changes, while prefix length, branch
count, suffix distribution, output length, seed, arrival order, KV capacity,
and server settings remain fixed. Flash ordinary DP remains the primary
baseline; Fork ordinary DP is the routing ablation.

| Dataset | Flash ordinary branch tok/s | Fork ordinary branch tok/s | Fork prefix-aware branch tok/s | Gain vs Flash | Gain vs Fork | Prefix-aware branch phase | Prefix-aware TTFT P50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| AgentBoard | 231.4 | 233.7 | 1,007.1 | 4.35x | 4.31x | 16,268.9 ms | 2,135.6 ms |
| AppWorld | 173.8 | 163.1 | 1,033.4 | 5.95x | 6.33x | 15,854.8 ms | 2,131.5 ms |
| SWE-bench Verified | 185.8 | 174.4 | 1,032.7 | 5.56x | 5.92x | 15,865.9 ms | 1,964.3 ms |

All nine runs completed 66/66 requests with zero preemptions. Every
prefix-aware run bootstrapped two long-prefix owners, recorded 64 affinity
routes and 64 cohort locks, and finished with an exact 33/33 rank split. The
prefix-aware throughput band across the three new datasets is
1,007.1-1,033.4 branch tok/s. This is a single-repeat cross-dataset check, so
the per-dataset ratios are evidence of consistency on this controlled shape,
not estimates of universal average speedup.

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
  balanced and substantially outperforms the primary Flash ordinary-DP
  baseline on all four tested dataset sources.
- The result does not imply that prefix-aware routing accelerates every DP
  request. Its benefit depends on prefix depth, cohort visibility, and KV
  pressure.
- AgencyBench Pressure32K/32 has two repeats; the other three datasets have one
  run per variant. This is not a broad workload average.

The generated common-analysis text is not guaranteed to be byte-identical
across DP schedules even with temperature zero. Compared runs use identical
request counts, input/output token totals, length distribution, seed, model,
KV capacity, and graph-enabled policy. The primary comparison changes the
attention backend, backend-specific graph implementation, and routing together;
the Fork ordinary ablation isolates the routing contribution. Performance
claims use measured timing rather than logical KV savings.

## Validation

- All current-router runs completed without preemption or request failure.
- Pressure32K/32 prefix-aware branch throughput remained within a narrow
  1,570.1-1,611.9 tok/s band.
- The three cross-dataset prefix-aware runs remained within
  1,007.1-1,033.4 branch tok/s and each finished with a 33/33 rank split.
- The current adaptive forest validation passed 75 graph-dispatch, ForkAttention
  backend, GPU-workspace, and prefix-router tests. LangGraph was outside this
  validation scope.
- No offload connector was enabled in these runs.
