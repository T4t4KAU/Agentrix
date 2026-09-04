# Prefix-Aware Data-Parallel Routing: KV-Affinity Profiling

## Summary

This experiment profiles Agentrix's prefix-aware data-parallel router in a
workload where long prefixes are already resident on specific model replicas.
The router increases the measured revisit cache-hit rate from 0% to 100% in a
controlled placement stress test and from a median 53.3% to 100% with shuffled
revisit arrivals.

The controlled scenario is an intentionally favorable ceiling, not a general
serving-speed claim. The shuffled scenario is the more representative result:
its median batch completion time improves from 922.4 ms to 121.5 ms (7.59x),
while median P50 TTFT improves from 702.2 ms to 96.0 ms (7.31x).

## Test System

- Date: 2026-09-04
- GPU: 2 x NVIDIA GeForce RTX 5090, 32 GiB each
- Model: Qwen3-VL-8B-Instruct
- vLLM: 0.25.0 with the Agentrix ForkAttention and prefix-aware DP changes
- PyTorch: 2.11.0+cu130
- CUDA runtime: 13.0
- Model dtype: BF16
- Data parallel size: 2, tensor parallel size: 1
- API frontend count: 1
- Attention backend: `FORK_ATTN`
- Automatic prefix caching: enabled
- Asynchronous scheduling: disabled
- Maximum model length: 4096
- Maximum sequences: 16
- Maximum batched tokens: 8192
- GPU memory utilization: 0.80
- CUDA Graph capture sizes: 1, 2, 4, 8, and 16
- CUDA Graph mode: full and piecewise; reported capture memory was 0.09 GiB
  per rank

The FlashInfer sampler was disabled in both arms because its installed build
misdetected SM 12.0 during JIT capability checking. Both arms used the same
native vLLM sampler, so this does not change the routing comparison.

## Workload

Each trial uses 15 independent synthetic documents. A document consists of a
3072-token deterministic high-entropy prefix and an 8-token query suffix. The
warm request and revisit request use the same document prefix but different
suffixes. Every request generates one token, which makes the experiment
primarily a prefill and KV-reuse measurement.

The trial procedure is:

1. Reset the physical prefix caches and the router's residency hints.
2. Submit one warm request per document sequentially, with 150 ms between
   requests.
3. Submit all 15 revisit requests concurrently, with a 1 ms launch gap.
4. Record streamed TTFT, end-to-end latency, batch makespan, and the
   request-level `cached_tokens` field.

Every reported arm contains five trials. Warm-up requests are excluded from
the latency statistics. The server and benchmark client run on the same host.

Two revisit orders are used:

- **Controlled placement:** revisit documents in the same order as the warm
  phase. With DP=2 and an odd document count, native tie rotation sends every
  revisit to the opposite replica. This deliberately measures the upper bound
  of benefit from correct affinity.
- **Shuffled arrival:** shuffle revisit order with a different deterministic
  seed in each trial. Native routing then finds a resident prefix by chance on
  roughly half of requests.

## Results

Values below are medians across five trials. Speedup is baseline divided by
prefix-aware latency, or prefix-aware divided by baseline throughput.

### Controlled Placement

| Metric | Native DP | Prefix-aware DP | Improvement |
|---|---:|---:|---:|
| Revisit requests with a cache hit | 0 / 15 | 15 / 15 | 100 percentage points |
| Cached prompt-token rate | 0.00% | 99.74% | 99.74 percentage points |
| Mean TTFT | 1162.7 ms | 94.6 ms | 12.29x |
| P50 TTFT | 1402.3 ms | 104.7 ms | 13.40x |
| P95 TTFT | 1760.0 ms | 117.5 ms | 14.98x |
| Batch makespan | 1780.2 ms | 134.4 ms | 13.24x |
| Revisit throughput | 8.43 req/s | 111.59 req/s | 13.24x |

The native results are highly stable: batch makespan ranges from 1775.7 to
1784.1 ms. Prefix-aware batch makespan ranges from 84.4 to 153.8 ms.

### Shuffled Arrival

| Metric | Native DP | Prefix-aware DP | Improvement |
|---|---:|---:|---:|
| Revisit requests with a cache hit | 8 / 15 median (8-10) | 15 / 15 | 5-7 more hits |
| Cached prompt-token rate | 53.19% median | 99.74% | 46.55 percentage points |
| Mean TTFT | 651.6 ms | 93.9 ms | 6.94x |
| P50 TTFT | 702.2 ms | 96.0 ms | 7.31x |
| P95 TTFT | 856.8 ms | 107.7 ms | 7.95x |
| Batch makespan | 922.4 ms | 121.5 ms | 7.59x |
| Revisit throughput | 16.26 req/s | 123.43 req/s | 7.59x |

The prefix-aware shuffled batch makespan ranges from 76.4 to 150.8 ms. The
native range is 762.1 to 971.4 ms.

## Correctness and Load-Balance Evidence

- Across the controlled runs, all 75 native revisit requests report zero
  cached tokens. All 75 prefix-aware revisits report exactly 3072 cached
  tokens. Every warm request reports zero cached tokens after cache reset.
- The controlled prefix-aware server processes 80 total requests on rank 0 and
  70 on rank 1. The 75 cache-hit revisits split 40/35 between the ranks. This
  is the expected 53.3%/46.7% split from five odd-sized batches, not a collapse
  onto one replica.
- In the shuffled native runs, the two ranks process exactly 75 requests each
  and each records 21 cached revisits.
- Server logs contain no inference errors, aborts, preemptions, or attention
  backend fallbacks during the measured runs.
- Each final arm is validated from server logs to contain exactly 150 requests
  and five successful cache resets. Earlier overlapping exploratory runs were
  discarded and are not present in the result CSV.

## Router CPU Cost

A separate 5000-decision microbenchmark on the server used the same 3080-token
request shape, 15 resident prefixes, and two ranks. It includes prefix-chain
construction and rank selection but excludes construction of the already
available engine request object.

| Statistic | Routing time |
|---|---:|
| Mean | 46.6 us |
| P50 | 45.9 us |
| P95 | 48.3 us |
| P99 | 76.3 us |

This frontend cost is less than 0.05% of the controlled prefix-aware mean TTFT
and is negligible relative to the approximately 1.07-second mean TTFT saved by
avoiding a 3072-token recomputation.

## Interpretation

The router is effective when the expensive portion of a request is a long
prefix resident on only one DP replica. Queue-only routing cannot distinguish
the replicas in that state, so a wrong choice recomputes 3072 tokens. The
prefix-aware router selects the resident replica while its load and estimated
work remain within configured bounds. In this experiment the resident
documents are already balanced across both ranks, so locality does not trade
away parallelism.

This result does not imply a 7-15x gain on general traffic. It is intentionally
prefill-heavy, uses one output token, and has very high reusable-prefix
coverage. The expected gain decreases with shorter prefixes, longer decode,
low revisit probability, shared KV storage across replicas, or queue imbalance
large enough to trigger the router's load bounds.

## Reproduction

The benchmark client is
`benchmark/scripts/benchmark_prefix_aware_dp.py`. The two server arms differ
only in this environment variable:

```bash
# Native internal DP routing
VLLM_FORK_ATTN_DP_PREFIX_ROUTING=0

# Prefix-aware internal DP routing
VLLM_FORK_ATTN_DP_PREFIX_ROUTING=1
```

Example client invocation:

```bash
benchmark/.venv/bin/python benchmark/scripts/benchmark_prefix_aware_dp.py \
  --base-url http://127.0.0.1:8130 \
  --model qwen3-vl \
  --documents 15 \
  --prefix-tokens 3072 \
  --suffix-tokens 8 \
  --output-tokens 1 \
  --trials 5 \
  --revisit-order shuffled \
  --output result.json
```

The per-trial measurements are stored in
`docs/experiment_results/prefix_aware_dp_affinity_profile.csv`.
