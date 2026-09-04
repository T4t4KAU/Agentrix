# Prefix-Aware Data-Parallel Routing Profile

## Summary

This experiment isolates the benefit of routing a repeated long prefix back to
the data-parallel rank whose local KV cache already contains that prefix. On a
two-rank deployment, prefix-aware routing raised the repeated-prefix cache hit
rate from 0% to 100% and increased median batch throughput from 8.43 to 96.08
requests/s (11.40x).

The result is intentionally a strong-affinity microbenchmark. It demonstrates
that the routing mechanism works and quantifies its upper-bound benefit; it is
not a general production-workload throughput claim.

## Environment

- Date: 2026-09-04
- Host: AutoDL container, four NVIDIA GeForce RTX 5090 GPUs (32,607 MiB each)
- GPUs used: GPU 0 and GPU 1
- Model: `Qwen3-VL-8B-Instruct`
- vLLM base: 0.25.0 with Agentrix ForkAttention and unified CUDA Graph support
- Data parallel size: 2 (one model replica per GPU)
- Attention backend: `FORK_ATTN`
- Dtype: BF16
- Maximum model length: 4,096 tokens
- Maximum sequences: 16
- Maximum batched tokens: 8,192
- Prefix caching: enabled
- Async scheduling: disabled
- CUDA Graph capture sizes: 1, 2, 4, 8, and 16
- FlashInfer sampler: disabled because the installed FlashInfer capability
  probe does not recognize the RTX 5090 SM 12.x device correctly

The baseline and treatment used identical arguments. The only routing change
was `VLLM_FORK_ATTN_DP_PREFIX_ROUTING=0` versus
`VLLM_FORK_ATTN_DP_PREFIX_ROUTING=1`.

## Workload

Each trial used 15 logical documents. A document consists of a deterministic,
document-specific 3,072-token prefix and an 8-token suffix.

1. Reset the vLLM prefix cache and wait one second.
2. Submit one warm-up request for each document sequentially, with a 150 ms
   settling interval between requests.
3. Submit 15 revisit requests concurrently. Each revisit retains the same
   3,072-token document prefix but uses a different 8-token suffix.
4. Generate one token with greedy sampling and record streamed TTFT, end-to-end
   latency, cached-token accounting, and batch makespan.

Five independent trials were executed for each configuration. GPU power,
utilization, and framebuffer use were sampled once per second with
`nvidia-smi dmon`. The workload generator is
`benchmark/scripts/benchmark_prefix_aware_dp.py`.

## Results

All latency and throughput values below are medians across five trials. The
range column contains the minimum and maximum trial values.

| Metric | Native DP | Prefix-aware DP | Change |
| --- | ---: | ---: | ---: |
| Cache-hit requests | 0/15 | 15/15 | +15 requests |
| Cached-token rate | 0.00% | 99.74% | +99.74 pp |
| Mean TTFT | 1,162.7 ms | 110.6 ms | -90.5% |
| P50 TTFT | 1,402.3 ms | 115.4 ms | -91.8% |
| P95 TTFT | 1,760.0 ms | 143.4 ms | -91.9% |
| Batch makespan | 1,780.2 ms | 156.1 ms | -91.2% |
| Batch throughput | 8.43 requests/s | 96.08 requests/s | 11.40x |

Observed stability:

| Metric | Native DP range | Prefix-aware DP range |
| --- | ---: | ---: |
| P50 TTFT | 1,395.1-1,412.2 ms | 108.4-116.6 ms |
| P95 TTFT | 1,755.2-1,763.8 ms | 129.7-168.9 ms |
| Batch makespan | 1,775.7-1,784.1 ms | 140.3-189.4 ms |
| Throughput | 8.41-8.45 requests/s | 79.21-106.90 requests/s |

The native load balancer deterministically sent every revisit to a rank that
did not hold its document prefix in these clean trials. Prefix-aware routing
sent every revisit back to the resident rank. Each treatment request therefore
reused 3,072 of 3,080 prompt tokens; the remaining eight suffix tokens explain
the 99.74% cached-token rate.

## Router CPU Cost

A separate in-process microbenchmark measured 5,000 routing decisions using
the same 3,080-token prompt length, 15 resident prefixes, and two ranks:

| Statistic | Routing time |
| --- | ---: |
| Mean | 52.6 us |
| P50 | 49.9 us |
| P95 | 67.2 us |
| P99 | 74.9 us |

This includes prefix-chain construction and rank selection. In this workload,
the roughly 53 us frontend cost is negligible compared with the approximately
1.05-second reduction in mean TTFT.

## GPU Sampling

Both configurations reached 97-100% sampled SM utilization on both GPUs. Peak
sampled framebuffer allocation was approximately 26,839 MiB for native DP and
26,073 MiB for prefix-aware DP. These one-second samples span both warm-up and
revisit phases; they are useful for detecting gross imbalance or OOM risk but
are too coarse to attribute kernel-level utilization or memory savings to the
router.

## Validation and Limitations

- The dedicated router test suite passed: 19 tests passed in 3.78 seconds.
- The server log confirmed that prefix-aware routing was enabled with block
  size 16, minimum prefix depth 4 blocks, load slack 32, and warm-hint TTL 30 s.
- All server processes were stopped after profiling; all four GPUs returned to
  0 MiB used and 0% utilization.
- One-token completions intentionally isolate prefix-prefill and routing cost.
  A follow-up workload should use longer decoding to measure TPOT and steady
  decode throughput.
- The experiment represents strong, evenly reusable affinity. Mixed reuse,
  highly skewed hot prefixes, rank overload, multiple API frontends, and cache
  eviction require separate profiles.
