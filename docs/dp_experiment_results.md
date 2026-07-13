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
