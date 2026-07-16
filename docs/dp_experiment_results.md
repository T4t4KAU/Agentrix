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

## Representative Route-Ownership Snapshots

One diagnostic repeat of the AgentBoard Pressure32K/32 Fork prefix-aware
variant and one of the Flash ordinary-DP baseline were used to inspect actual
per-rank placement. These are illustrative routing runs, not additional
performance repeats. Both used the same seeded client workload and submission
configuration as the formal runs. Concurrent HTTP handling and the
prefix-aware arrival wave can change server dispatch order, so intermediate
rows describe each run's observed order rather than paired requests.

After each routing decision, diagnostic-only logging hashed the first 4,096
prompt token IDs into an opaque case fingerprint and recorded the selected DP
rank. The fingerprint was computed after rank selection and was not an input
to either router. The two stable fingerprints are labeled A and B below. Each
log contained exactly 66 contiguous records: two distinct bootstrap requests,
then 32 A branches and 32 B branches. Both runs completed 66/66 requests with
zero preemptions. Because synchronous hashing and logging can perturb later
load observations, diagnostic-run throughput is not used in the result tables.

Fanout rows are cumulative and exclude the two bootstrap requests.

| Variant / dispatch point | A to rank 0 | A to rank 1 | B to rank 0 | B to rank 1 | Total by rank |
|---|---:|---:|---:|---:|---:|
| Prefix-aware: bootstrap | 1 | 0 | 0 | 1 | 1 / 1 |
| Prefix-aware: first 16 fanout | 12 | 0 | 0 | 4 | 12 / 4 |
| Prefix-aware: first 32 fanout | 17 | 0 | 0 | 15 | 17 / 15 |
| Prefix-aware: first 48 fanout | 31 | 0 | 0 | 17 | 31 / 17 |
| Prefix-aware: all 64 fanout | 32 | 0 | 0 | 32 | 32 / 32 |
| Flash ordinary: bootstrap | 0 | 1 | 1 | 0 | 1 / 1 |
| Flash ordinary: first 16 fanout | 5 | 6 | 3 | 2 | 8 / 8 |
| Flash ordinary: first 32 fanout | 10 | 9 | 6 | 7 | 16 / 16 |
| Flash ordinary: first 48 fanout | 13 | 13 | 11 | 11 | 24 / 24 |
| Flash ordinary: all 64 fanout | 18 | 14 | 14 | 18 | 32 / 32 |

Both variants therefore finish with a superficially perfect 33/33 total rank
split after bootstrap. The composition is different. Prefix-aware routing
keeps all 32 A branches on rank 0 and all 32 B branches on rank 1. Flash
ordinary DP balances request counts but places both roots on both replicas:
rank 0 receives 18 A and 14 B branches, while rank 1 receives 14 A and 18 B
branches. Aggregate rank counts alone therefore cannot establish prefix
locality.

The prefix-aware run's observed dispatch order contained substantial transient
skew, including 12/4 and 31/17 cumulative rank splits, rather than an
artificially alternating order. It still preserved the two prefix owners.
Flash ordinary DP instead tracked its load score closely,
reaching 8/8, 16/16, 24/24, and 32/32 while progressively mixing both cases
across both ranks.

Representative server samples confirm concurrent activity. During the
prefix-aware dispatch window, rank 0 reported 31 running requests at 63.9% KV
usage and rank 1 reported 16 at 60.9%, with both GPUs at 97% compute
utilization. During the Flash run, rank 0 reported 2 running and 29 waiting at
57.5% KV usage and rank 1 reported 1 running and 30 waiting at 55.9%, with both
GPUs at 100% utilization. These records prove coordinator route ownership and
concurrent device activity; they are not single-clock CTA or physical KV-block
maps, and one diagnostic run must not be generalized to every ordinary-DP
schedule.

## Why Branch TTFT Drops

The route snapshots above show placement, not the amount of KV actually reused.
The formal AgencyBench Pressure32K/32 runs also exported vLLM's scheduler-derived
prompt-token source counters, which provide the missing prefill evidence. All
four runs processed the same 2,191,629 prompt tokens and used no external KV
transfer.

| Variant / branch tok/s | Local prompt compute | Local cache hit | Cache-hit share | Cumulative prefill time | Cumulative queue time |
|---|---:|---:|---:|---:|---:|
| Flash ordinary / 394.7 | 190,173 | 2,001,456 | 91.32% | 66.202 s | 1,002.421 s |
| Flash ordinary / 212.6 | 279,757 | 1,911,872 | 87.24% | 66.690 s | 2,379.357 s |
| Fork prefix-aware / 1,570.1 | 76,285 | 2,115,344 | 96.52% | 15.988 s | 0.000595 s |
| Fork prefix-aware / 1,611.9 | 76,285 | 2,115,344 | 96.52% | 16.001 s | 0.000644 s |

`Local prompt compute` is the number of prompt tokens that the scheduler says
must be prefilled locally after prefix-cache lookup; `Local cache hit` is the
number skipped through resident KV reuse. The time columns sum per-request
durations across all 66 concurrent requests and therefore are not experiment
wall time.

Prefix-aware placement reduces actual prompt computation by 60-73% relative to
the two Flash ordinary runs and cumulative prefill time by about 76%. The much
larger TTFT change is the resulting queueing amplification: ordinary DP spends
large cumulative time waiting behind long or repeated prefills, whereas the
prefix-aware runs admit the shared-prefix cohorts without a material scheduler
queue. Thus the causal description is not simply "each prefill becomes 30
seconds faster"; reduced prefill work and KV pressure collapse the queue in
front of the median request.

The capacity point explains why the amplification is strong. Each rank has
61,632 tokens of physical GPU KV capacity. The two common requests contain
32,854 and 32,889 prompt tokens, or 65,743 tokens together, before branch
suffixes and outputs. When ordinary DP mixes both roots across a rank, as the
diagnostic snapshot demonstrates, that rank cannot retain both complete roots
plus active branch state. Prefix-aware DP keeps one root cohort per rank and
avoids this capacity cliff.

The latency measurements are internally consistent with this explanation.
Branch TTFT is measured from immediately before the streaming API call until
the first non-empty generated content and includes HTTP handling, the arrival
wave, scheduler waiting, and prefill. The common-analysis bootstrap happens
before fanout and is not part of branch TTFT. Fork ordinary DP remains close to
Flash ordinary DP at 29.65 versus 32.46 seconds TTFT P50, while prefix-aware DP
falls to 2.15 seconds. TPOT does not improve: it changes from 25.47 ms for Flash
ordinary to 26.82 ms for Fork prefix-aware. The large gain is therefore a
prefill-locality and queueing result, not faster steady-state decode.

This mechanism is credible for the controlled Agent fanout shape, but the exact
TTFT ratio is not universal. The benchmark intentionally places two long roots
near a per-rank KV-capacity boundary, and the two Flash repeats show substantial
run-to-run variation. It also measures fanout after the required common-analysis
stage, not cold end-to-end latency from the first root request.

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

## H20 DP=4 Pressure32K/32 Validation

An additional DP=4 validation was run on 2026-07-16 on a four-GPU H20
(SM90) server. This run increases both the model size and the number of DP
replicas relative to the preceding RTX 5090 DP=2 results. The raw server-side
artifacts are retained at:

```text
/test__02/hwx/Agentrix/benchmark/results/
  h20_dp4_pressure32k_b32_qwen3_32b_r1/dp/
```

### Environment and Workload

The hardware inventory below was queried directly from the benchmark server
after the run.

#### Hardware Configuration

| Hardware item | Observed configuration |
|---|---|
| GPU accelerators | 4 x NVIDIA H20-3e |
| GPU memory | 143,771 MiB per GPU (approximately 140.4 GiB); approximately 561.6 GiB total |
| GPU compute capability | 9.0 (SM90) |
| GPU power limit | 500 W per GPU |
| GPU interconnect | Every GPU pair reports `NV18` connectivity in `nvidia-smi topo -m` |
| GPU PCI bus IDs | `04:00.0`, `23:00.0`, `44:00.0`, `63:00.0` |
| Host CPU | 2 x AMD EPYC 9654 96-Core Processor; 192 physical cores, 384 logical CPUs, SMT2 |
| CPU frequency range | 1.5 GHz minimum; approximately 3.708 GHz maximum |
| NUMA topology | 2 NUMA nodes; all four GPUs report NUMA-node-0 CPU affinity (`0-95,192-287`) |
| Host memory | 1.5 TiB RAM; no swap configured |
| Workspace storage | GPFS mounted at `/test__02`; 563 TiB total and 539 TiB available when recorded |
| Operating system | Ubuntu 22.04.5 LTS, Linux `5.15.0-60-generic`, x86-64 |

#### Software and Benchmark Configuration

| Item | Value |
|---|---|
| Validation date | 2026-07-16 |
| NVIDIA driver | 550.144.03 |
| CUDA toolkit | 12.9.86 |
| PyTorch | 2.11.0+cu129 |
| Agentrix commit | `5793805` |
| vLLM submodule commit | `b5086c8b7` |
| Model | Qwen3-32B, float16 |
| Parallelism | vLLM internal DP=4, TP=1 |
| Dataset / shape | AgencyBench, four cases, 32 branches per case |
| Measured shared prefix | Approximately 32,835 tokens per case |
| Requests | 4 common bootstrap + 128 branches = 132 total |
| Branch output | 256 tokens per branch; 32,768 tokens total |
| Arrival policy | All cases concurrent, shuffled with seed 2026 |
| Maximum sequences | 64 per rank |
| GPU KV capacity | 3,852 blocks per rank = 61,632 tokens per rank |
| CUDA Graphs | Enabled for both variants; Prefix Forest graphs for ForkAttention |
| Offloading | Disabled |
| Reload rebalance | Disabled |

Both variants used the same fixed physical KV capacity, request construction,
branch order, output-token count, and freshly started service. The comparison
contains only the requested primary arms: FlashAttention with native ordinary
DP and ForkAttention with prefix-aware DP. It does not contain an H20
ForkAttention ordinary-DP ablation, which limits attribution between backend
and routing.

The branch-phase timer surrounds the concurrent fanout requests and excludes
the four common bootstrap requests. The case-level wall time starts before the
common bootstrap stage and ends after fanout, but still excludes server launch
and startup graph capture. Therefore the 12.21x result below is a fanout-stage
systems result; 7.71x is the more conservative measured case-level result.

### Performance Results

All 132 requests completed in both variants with no request failure and no
preemption.

| Variant | Branch wall | Branch output tok/s | Case wall | Case output tok/s | TTFT P50 / P95 | TPOT P50 / P95 | Peak GPU KV |
|---|---:|---:|---:|---:|---:|---:|---:|
| FlashAttention ordinary DP | 470.6866 s | 69.6174 | 497.1948 s | 65.9058 | 208.228 / 450.098 s | 36.701 / 63.856 ms | 55.97% |
| ForkAttention prefix-aware DP | 38.5567 s | 849.8656 | 64.4482 s | 508.4397 | 6.3109 / 8.8839 s | 117.131 / 130.769 ms | 74.63% |

This corresponds to:

- 12.21x higher branch-only output throughput and 91.81% lower branch wall
  time;
- 7.71x higher case-level output throughput and 87.04% lower measured case
  wall time; and
- 33.00x lower branch TTFT P50, but 3.19x worse branch TPOT P50.

The identical total prompt-token source counters make it possible to inspect
where the wall-time difference comes from. The time columns below are sums of
per-request scheduler durations across 132 concurrent requests, not elapsed
experiment wall time.

| Variant | Total prompt source | Local prompt compute | Local cache hit | Cache-hit share | Cumulative prefill | Cumulative queue | Cumulative decode |
|---|---:|---:|---:|---:|---:|---:|---:|
| FlashAttention ordinary DP | 4,402,747 | 1,632,251 | 2,770,496 | 62.93% | 1,540.571 s | 26,643.441 s | 1,185.411 s |
| ForkAttention prefix-aware DP | 4,402,747 | 153,467 | 4,249,280 | 96.51% | 338.468 s | 0.003621 s | 3,930.522 s |

Prefix-aware placement therefore avoids 1,478,784 locally computed prompt
tokens, reducing local prompt computation by 90.60% and cumulative prefill
time by 78.02%. The higher cumulative decode time and worse TPOT show that the
headline speedup comes from the DP mechanism measured here: prefix residency,
admission, and the resulting removal of repeated prefill and queueing. This
experiment does not make a claim that the ForkAttention decode kernel is
faster than FlashAttention.

The prefix-aware router recorded four bootstrap routes, 128 affinity routes,
128 cohort locks, and a final route allocation of `[33, 33, 33, 33]`. Its
average routing time was 12,570.9 microseconds per request. It observed 129
arrival waves for 132 requests, so large arrival-wave batching was not the
source of the throughput improvement. The final allocation demonstrates that
prefix affinity did not sacrifice aggregate request-count balance in this
four-cohort/four-rank shape.

ForkAttention physical execution was also active rather than bypassed:

| Physical execution metric | Value |
|---|---:|
| Observed steps | 1,520 |
| Fork-active steps | 1,026 (67.50%) |
| Shared CTAs | 56,901 |
| Singleton CTAs | 32,290 |
| Shared-CTA share | 63.80% |

These counters prove that the shared-prefix kernel path executed, but they do
not make kernel speed the basis of the 12.21x DP result. Backend-level kernel
comparison is outside this experiment's attribution scope.

### Why TPOT Gets Worse While Throughput Improves

The H20 TPOT regression is not inconsistent with the throughput gain. The
benchmark does not timestamp every generated token. For a streamed request it
computes average post-first-token latency as
`(request latency - TTFT) / (output tokens - 1)`. Consequently, TPOT excludes
the queueing before the first token but includes scheduling gaps and batched
decode time after the first token; it is not a decode-kernel-only measurement.

Ordinary DP reaches the multi-root KV-capacity cliff and leaves many requests
waiting before their first token. This produces very poor TTFT and aggregate
throughput, but a request that has entered decode can share the GPU with a
smaller active set and therefore observe a shorter interval between output
tokens. Prefix-aware DP removes almost all of that queue and can admit many
siblings from the resident cohort concurrently. Each larger decode step takes
longer, so an individual request receives its next token less frequently, but
the step produces tokens for many more requests. The increase in active
sequences is larger than the increase in step time, yielding 12.21x higher
aggregate branch throughput despite 3.19x worse median TPOT. This is a
latency-throughput tradeoff caused primarily by admission and effective batch
size, rather than a contradiction in the measurements.

The cumulative decode durations above reinforce the concurrency change but do
not isolate kernel speed: they sum per-request residence time across concurrent
requests rather than measuring GPU decode wall time. The current DP comparison
also changes the attention backend, so backend behavior remains a confounding
variable rather than a claimed source of acceleration. If kernel attribution
is needed later, it requires a separate fixed-resident-KV experiment with
matched active sequence counts and batch shapes; it is not required for the
present conclusion about DP placement and admission.

### Why the Improvement Is So Large

This experiment deliberately operates at a KV-capacity boundary. One measured
root is approximately 32.8K tokens, while a rank has room for 61,632 tokens.
Two roots require roughly 65.7K tokens before any branch suffix or generated
KV is included, so one root fits comfortably but two do not.

Ordinary DP balances request counts without preserving content ownership.
With four shuffled cohorts, sibling branches can be distributed to ranks that
do not retain their root, and a rank can alternate among multiple roots whose
combined working set exceeds its KV capacity. Prefix-cache reuse is rank-local,
so this placement causes repeated long-prefix prefill, cache displacement, and
scheduler waiting even when the global request count is balanced.

Prefix-aware DP instead keeps each 32-branch cohort with its resident root. In
this symmetric case, four roots map naturally to four ranks. Each rank stores
one root once and admits that root's short private suffixes and outputs. The
result is a nonlinear effect:

1. deep-prefix hits remove about 1.48 million repeated prompt-token computes;
2. the smaller resident working set stays below the per-rank capacity cliff;
3. the scheduler can admit many more sibling branches concurrently; and
4. the ordinary-DP queue collapses rather than merely becoming proportionally
   shorter.

This also explains why prefix-aware DP reports higher peak KV usage: it is
using the available cache for concurrently admitted, useful branch state.
FlashAttention ordinary DP reports lower instantaneous occupancy while many
requests remain queued behind repeated prefill work.

The supported causal claim is therefore that prefix-aware residency and
admission remove repeated prefill and queue amplification. ForkAttention's
forest execution may reduce redundant shared-prefix reads during decode, but
the present result does not quantify a positive kernel-level contribution.

### Credibility and Scope

The mechanism is internally consistent and credible for this controlled
shape: both arms completed identical request and token counts, the
scheduler-derived prompt sources reconcile exactly, no preemption or retry
explains the difference, route ownership is balanced, and the cache-hit,
prefill, queue, TTFT, and wall-time changes all point to the same capacity-cliff
explanation.

The exact 12.21x ratio is not yet a general performance claim:

- this H20 comparison currently has one run per variant;
- it changes both attention backend and DP routing, without the H20
  ForkAttention ordinary-DP ablation;
- four equally sized cohorts on four ranks are an ideal ownership mapping;
- the fixed 61,632-token KV capacity intentionally makes one root fit while
  two do not; and
- the branch-only ratio excludes the required bootstrap stage, for which the
  case-level result is the more conservative 7.71x.

The next validation should add ForkAttention ordinary DP, repeat all arms at
least three times with randomized variant order, sweep KV capacity across the
one-root/two-root boundary, and test two, four, and eight uneven or staggered
cohorts. A resident-KV decode-only Nsight/NCU sweep is separately required to
isolate the H20 ForkAttention kernel from routing, prefill, and queueing.
