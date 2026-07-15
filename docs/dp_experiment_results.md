# Prefix-Aware Data Parallel Experiment Results

## Objective

This experiment compares three two-replica vLLM deployments without KV
offloading:

1. FlashAttention with ordinary vLLM internal data parallel routing.
2. ForkAttention with ordinary vLLM internal data parallel routing.
3. ForkAttention with the Agentrix prefix-aware data parallel router.

The purpose is to separate the attention backend effect from the routing
effect and to measure throughput, logical KV footprint, physical GPU KV
capacity, prefix-cache effectiveness, and router overhead.

## Environment

| Item | Value |
|---|---|
| Date | 2026-07-12 |
| GPUs | 2 x NVIDIA GeForce RTX 5090, 32 GiB each |
| Driver | 595.71.05 |
| CUDA toolkit | 12.8.93 |
| PyTorch | 2.11.0+cu128 |
| GPU architecture | SM120 |
| Model | Qwen3-8B, float16 |
| Dataset | AgencyBench, all 32 records |
| DP deployment | vLLM internal DP, 2 replicas |
| Offloading | Disabled |

Each run contains four batches. A batch contains up to eight dataset cases,
eight branches per case, an 8,192-token target prefix, a 128-token equal
suffix, and 64 generated tokens. Maximum request concurrency is 64. The
server is restarted between variants.

## Throughput

Throughput below is weighted across all four batches as total generated branch
tokens divided by total measured wall time. This is preferable to averaging
per-batch throughput values when batch durations differ.

### End-to-End Output Throughput

| Variant | Run 1 | Run 2 | Run 3 | Median |
|---|---:|---:|---:|---:|
| FlashAttention, ordinary DP | 399.78 | 421.33 | 419.20 | 419.20 tok/s |
| ForkAttention, ordinary DP | 392.35 | 416.70 | 418.94 | 416.70 tok/s |
| ForkAttention, prefix-aware DP | 496.01 | 581.07 | 568.83 | **568.83 tok/s** |

### Branch-Phase Output Throughput

| Variant | Run 1 | Run 2 | Run 3 | Median |
|---|---:|---:|---:|---:|
| FlashAttention, ordinary DP | 612.30 | 664.40 | 662.18 | 662.18 tok/s |
| ForkAttention, ordinary DP | 601.85 | 660.40 | 667.45 | 660.40 tok/s |
| ForkAttention, prefix-aware DP | 882.24 | 1199.51 | 1147.86 | **1147.86 tok/s** |

The paired median improvements of prefix-aware DP are:

| Baseline | E2E throughput | Branch throughput |
|---|---:|---:|
| FlashAttention ordinary DP | **+35.69%** | **+73.35%** |
| ForkAttention ordinary DP | **+35.78%** | **+71.98%** |

ForkAttention with ordinary DP is approximately neutral relative to
FlashAttention ordinary DP: the paired median differences are -1.10% E2E and
-0.60% branch throughput. The large gain therefore comes from prefix-aware
routing rather than from changing the attention backend alone.

## KV Cache Capacity and Logical Footprint

The model uses 147,456 logical KV bytes per token. The GPU cache capacity was
identical for all three variants within each run:

| Run | Tokens per replica | GiB per replica | Total tokens | Total GiB |
|---|---:|---:|---:|---:|
| 1 | 61,456 | 8.440 | 122,912 | 16.879 |
| 2 | 68,720 | 9.437 | 137,440 | 18.875 |
| 3 | 68,720 | 9.437 | 137,440 | 18.875 |

Run 1 had a cold torch compilation cache and left less memory for KV blocks.
Runs 2 and 3 reused the compilation cache. This does not bias comparisons
within a run because every variant received the same capacity.

Across the complete 32-record workload, independent branch-local KV would
require 2,171,848 token entries. The shared-prefix representation requires
96,313 token entries:

| Branch-local KV | Shared KV | Saved tokens | Saved GiB | Reduction |
|---:|---:|---:|---:|---:|
| 2,171,848 | 96,313 | 2,075,535 | 285.031 | **95.57%** |

This is a workload-derived logical footprint. It is not a claim that 285 GiB
was simultaneously resident on the two GPUs.

## Prefix Cache Effectiveness

The vLLM Prometheus counters report queried and hit prefix-cache token volume.
The table uses the median of three runs. "Miss GiB equivalent" multiplies miss
tokens by the model's logical KV bytes per token; it represents avoided KV
construction/admission demand, not PCIe traffic because offloading is disabled.

| Variant | Query tokens | Hit tokens | Miss tokens | Hit rate | Miss GiB equivalent |
|---|---:|---:|---:|---:|---:|
| FlashAttention, ordinary DP | 2,992,384 | 2,386,176 | 589,872 | 80.29% | 81.007 |
| ForkAttention, ordinary DP | 4,854,542 | 2,618,208 | 2,278,181 | 53.47% | 312.861 |
| ForkAttention, prefix-aware DP | 2,437,469 | 2,115,312 | 322,157 | **86.78%** | **44.242** |

Prefix-aware DP reduces prefix-cache miss token volume by a paired median of
45.39% relative to FlashAttention ordinary DP and 83.05% relative to
ForkAttention ordinary DP.

## Router Activity

The optimized router was active in every run rather than merely enabled:

| Metric | Run 1 | Run 2 | Run 3 | Median |
|---|---:|---:|---:|---:|
| Routing decisions | 288 | 288 | 288 | 288 |
| Prefix-affinity routes | 245 | 252 | 253 | 252 |
| Graph-bound routes | 20 | 8 | 10 | 10 |
| Rank 0 / rank 1 routes | 141 / 147 | 142 / 146 | 148 / 140 | balanced |
| Average route time | 998.9 us | 742.0 us | 927.5 us | 927.5 us |

The median affinity rate is 87.5%, while rank assignment remains balanced.
The approximately 0.93 ms routing cost is small compared with multi-second
8K-prefix request latency and is included in the end-to-end measurements.

## Validation and Caveats

- The benchmark suite passed all 23 unit tests.
- Two targeted ForkAttention GPU tests covering partial suffix masks and
  interleaved KV pages passed on SM120.
- All API requests completed successfully. `EngineDeadError` messages that
  appear after result collection are generated by the runner's zero-timeout
  forced shutdown, after all HTTP 200 responses and metric collection.
- Results cover one model, one dataset, two replicas, and three repetitions.
- The first run includes colder compilation and allocation caches. Paired
  comparisons and medians are used to reduce that effect.
- `PROFILE_FORK=1` was enabled for both ForkAttention variants to collect
  CUDA Graph and router telemetry.
- No CPU or disk KV offloading was enabled in any variant.

Raw results are stored on the AutoDL host under:

```text
/root/autodl-tmp/Agentrix/benchmark/results/dp_agencybench_qwen3_8b_cu128_r1
/root/autodl-tmp/Agentrix/benchmark/results/dp_agencybench_qwen3_8b_cu128_r2
/root/autodl-tmp/Agentrix/benchmark/results/dp_agencybench_qwen3_8b_cu128_r3
```

## Capacity-Aware Router Follow-up

The prefix-aware router was reworked and retested on 2026-07-14 and
2026-07-15 after focused pressure runs exposed two sources of instability.
This follow-up is implemented by vLLM commit `0cff6d857` and does not enable
or modify the offload policy.

The first issue was the affinity ordering. The old score compared active,
live, and warm residency classes before comparing total match depth. A shallow
but active common ancestor could therefore beat a deeper, case-specific warm
prefix on another rank. The second issue was request-local balancing: moving
only two branches of a 16K prefix cohort to the other rank duplicated roughly
32K prefix tokens, pushed peak KV usage above 99.8%, and caused 13 to 14
preemptions in repeated runs.

The corrected policy now:

- compares prefix match depth first and uses active/live/warm residency only
  to break ties at the same depth;
- balances a new long-prefix owner before applying prefix affinity;
- keeps a deep-prefix cohort together under a cumulative skew budget derived
  from matched prefix size, per-rank KV capacity, and `max_num_seqs`;
- relaxes work balance only by the amount of prefix recomputation avoided;
- sends prompts smaller than 20% of per-rank KV capacity through the native
  ordinary-DP path without building prefix hashes or delaying them in an
  arrival wave; and
- uses a 10 ms arrival wave for eligible long-prefix requests.

### Focused Test Configuration

Both ordinary and prefix-aware variants used the same ForkAttention backend,
Prefix Forest CUDA Graphs, Qwen3-8B float16 model, two internal DP ranks,
`max_num_seqs=64`, and 3,852 KV blocks per rank. Offload, fanout admission,
reload rebalance, and the FlashInfer sampler were disabled. Each run contained
four cases, 16 branches per case, concurrency 64, a deterministic lognormal
suffix distribution with mean 256, 256 branch output tokens, 64 common-analysis
tokens, and seed 2026.

| Scenario | Prefix construction | Branch input tokens/run | Branch output tokens/run |
|---|---|---:|---:|
| Warm8K | exact warm prefix, case-major order | 554,291 | 16,384 |
| Pressure16K | no explicit warmup, deterministic four-case shuffle | 1,086,003 | 16,384 |

Warm8K used three same-session paired repetitions with alternating variant
order: ordinary then final, final then ordinary, and ordinary then final. This
controls for the substantial run-to-run variance observed on the host. The
Pressure16K table reports three repetitions for each variant.

### Warm8K Paired Result

| Variant | Branch tok/s runs | Median | Branch phase | TTFT P50 | TPOT P50 | Peak KV | Max waiting | Preemptions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fork ordinary DP | 2,142.3 / 2,124.5 / 2,014.0 | 2,124.5 | 7,711.8 ms | 1,002.3 ms | 24.73 ms | 76.40% | 14 | 0 |
| Final prefix-aware policy | 2,210.3 / 2,116.7 / 2,021.3 | 2,116.7 | 7,740.2 ms | 1,003.4 ms | 23.82 ms | 76.53% | 14 | 0 |

The final policy differs from ordinary DP by -0.37% branch throughput, +0.37%
branch-phase time, +0.11% TTFT P50, and -3.67% TPOT P50. All 72 requests per
run took the native bypass, with zero arrival waves and zero prefix-routing
time. The short-prefix result is therefore neutral within measured variance
rather than the double-digit regression produced by the earlier router.

### Pressure16K Result

| Variant | Branch tok/s runs | Median | Branch phase | TTFT P50 | TPOT P50 | Peak KV | Max waiting | Median preemptions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fork ordinary DP | 443.6 / 385.8 / 454.3 | 443.6 | 36,937.9 ms | 17,253.2 ms | 21.66 ms | 88.28% | 56 | 0 |
| Legacy prefix-aware DP | 1,642.9 / 1,654.4 / 1,193.2 | 1,642.9 | 9,972.7 ms | 1,338.8 ms | 32.56 ms | 75.72% | 2 | 0 |
| Final prefix-aware DP | 1,677.2 / 1,689.4 / 1,716.0 | 1,689.4 | 9,698.4 ms | 1,432.5 ms | 31.32 ms | 75.19% | 2 | 0 |

Relative to ordinary DP, the final router improves branch throughput by
280.87% (3.81x), reduces branch-phase time by 73.74%, reduces TTFT P50 by
91.70%, and lowers peak KV usage by 13.09 percentage points. Its higher TPOT
than ordinary DP does not contradict the throughput result: ordinary DP leaves
many requests waiting before decode, as shown by its 56-request maximum queue
and 17.3-second TTFT.

Relative to the legacy prefix-aware router, the final router improves median
throughput by 2.83% and branch-phase time by 2.75%. More importantly, the
legacy third run fell to 1,193.2 tok/s with 13 preemptions, while all three
final runs completed without preemption. Every final run recorded 64/64
affinity routes, 64/64 cohort-locked routes, four balanced bootstrap routes,
a 34/34 rank split, peak KV usage between 75.14% and 75.50%, and a maximum
waiting queue of two.

The generated common-analysis text is not guaranteed to be byte-identical
across different DP schedules even with temperature zero. The compared runs
do, however, use identical request counts, input/output token totals, length
distribution, seed, model, KV capacity, backend, and CUDA Graph settings. The
claims above use measured end-to-end timing rather than logical KV or CTA
savings estimates.

The host validation combined 31 prefix-router tests with the 22 targeted
offload admission/planner tests introduced by the preceding offload commit;
all 53 passed. The DP and offload patches touch disjoint implementation files.

## Experimental KV Reload Rebalance

An additional experiment evaluated the default-off KV-reload Prefix Forest
rebalance path. Both variants used ForkAttention prefix-aware internal DP,
the same two GPUs and Qwen3-8B model, a fresh LMCache MP server, and LMCache's
default `LRU` policy. The only policy difference was
`VLLM_FORK_ATTN_DP_RELOAD_REBALANCE`.

The stress workload used two 8,192-token cases, nine branches per case, 768
maximum output tokens, concurrency 18, and 608 GPU KV blocks per replica. It
deliberately skewed each prefix forest across the two ranks and created 28 to
31 scheduler preemptions.

| Variant | Branch tok/s | E2E tok/s | Preemptions | Logical KV saved | GPU-local reload saved |
|---|---:|---:|---:|---:|---:|
| Default LMCache LRU | 233.64 | 221.21 | 28 | 19.464 GiB | 0 GiB |
| Reload rebalance enabled | 231.84 | 219.58 | 31 | 19.464 GiB | 0 GiB |

The measured differences were -0.77% branch throughput and -0.74% end-to-end
throughput. No reload intents, plans, or committed handoffs occurred. The
workload preempted running requests, but vLLM's local prefix cache retained
their blocks, so those requests never entered LMCache's external reload path.
The small throughput difference is therefore run-to-run noise rather than
evidence of either a gain or a regression caused by a handoff.

This result is important for interpreting the feature: running preemption
alone is not a valid trigger. The implementation now waits for an actual
external LMCache hit, validates target-side physical GPU residency in a
prepare phase, and rejects a target that has no local-prefix gain over the
source. A workload that forces local APC eviction while preserving the same
prefix on another DP rank is still required to quantify end-to-end benefit.

Reproduce the paired run with:

```bash
cd /root/autodl-tmp/Agentrix/benchmark
OUTPUT_ROOT=results/dp_reload_comparison_external_final \
  ./scripts/run_vllm_dp_reload_comparison.sh
```

As in the earlier experiments, the shutdown-time `EngineDeadError` was emitted
after all HTTP responses, metrics, and result files had been collected.
