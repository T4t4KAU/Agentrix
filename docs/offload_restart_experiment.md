# ForkAttention Cohort-Aware Three-Level Offload Experiment

## Conclusion

This report replaces the earlier two-level CPU-only result. The current
experiment enables the complete native vLLM hierarchy:

```text
GPU KV cache <-> pinned CPU KV cache <-> filesystem tier
```

Both policies use ForkAttention, the same fanout scheduler, and the same
three-level storage. The only policy difference is ordinary LRU offload versus
cohort-aware proactive backup.

The initial 16K-prefix/16-branch case exercised all three levels. Cohort-aware
offload reduced GPU reload traffic by **38.20%**, filesystem read traffic by
**6.22%**, and branch P50 TTFT by **6.96%**, but throughput was **0.97% lower**.
Instrumentation identified fragmented promotion I/O as the main remaining
problem: 83 load jobs versus 43 for ordinary offload, with average completion
latency increasing from 78.55 ms to 164.43 ms.

A first follow-up coalesced both submission and completion across sibling
requests. Repeated testing exposed head-of-line blocking and this design was
rejected. The retained implementation batches queue submission and interleaves
blocks across requests, but preserves independent request completion.

The final safe-batching design was evaluated with three paired 16K/16 runs,
including a reversed policy order. Mean physical Disk read fell from
2.938 to 2.595 GiB (**-11.65%**) and mean TTFT fell from 1,571.10 to
1,526.29 ms (**-2.85%**). However, throughput changed from 251.26 to
249.59 tok/s (**-0.66%**) and variability was high. The credible conclusion is
therefore a measured I/O-volume benefit, not a statistically established
end-to-end acceleration.

The 8K/8 case generated no filesystem reads and is a negative control. The
optimized policy was 4.25% slower initially and 2.10% slower with safe batching,
confirming that proactive backup should be gated when CPU capacity is sufficient
and no later restore is likely.

## Compared Policies

- **Ordinary:** ForkAttention, case-major fanout scheduling, native tiered
  `OffloadingConnector`, LRU CPU tier, and filesystem secondary tier;
  `fanout_offload=false`.
- **Cohort-aware:** the same configuration, plus `fanout_offload=true` and
  `fanout_allow_hot_prefix_backup=true`. Shared, long, active root prefixes are
  eligible for early GPU-to-CPU backup and physical-key reuse.

The filesystem tier is not treated as an independent accelerator. It is a
capacity tier used after the bounded CPU cache. A filesystem hit is useful only
relative to recomputing an evicted long prefix; it is expected to be slower than
a CPU-tier hit.

## Three-Level Semantics

The native `TieringOffloadingSpec` uses CPU as the primary offload tier and the
filesystem as a secondary tier. Stores are write-through: a GPU-to-CPU Store is
also submitted to the filesystem. Loads search CPU first; a filesystem hit is
loaded into CPU and then copied to GPU. Consequently:

- filesystem writes alone do not prove that L3 was needed for serving;
- non-zero filesystem **load** bytes prove an actual L3-to-L2 restore;
- filesystem logical Store bytes can exceed physical bytes because identical
  physical KV block files are deduplicated;
- GPU Load bytes and filesystem Load bytes need not be equal, because a Disk
  restore may repopulate CPU for a larger set than the immediately copied GPU
  subset.

## Experimental Setup

| Setting | Value |
|---|---|
| Validation date | 2026-07-18 |
| GPU | NVIDIA GeForce RTX 5070, 12,227 MiB |
| Model | Qwen3-1.7B, FP16 |
| Dataset | First two deterministic AgentBoard records |
| Matrix | 8K/8 negative control; 16K/16 L3-pressure case |
| Root cases | 2 per cell |
| Total requests | 18 for 8K/8; 34 for 16K/16 |
| Common/branch output | 64 tokens each |
| Branch suffix | Lognormal, mean target 256 tokens |
| Request order | Case-major |
| Explicit warm-up | Disabled |
| GPU KV capacity | 1,700 blocks, 27,200 tokens, 2.905 GiB |
| CPU KV capacity | 2 GiB pinned memory |
| KV block | 16 tokens, 1,835,008 bytes |
| Filesystem workers | 4 read and 4 write threads |
| Prefix cache | Enabled |
| Fanout admission window | 16 |
| CUDA Graph | Enabled, including Prefix Forest graphs |
| Sampling | Greedy; FlashInfer sampler disabled |
| Repetitions | 16K/16 safe batching: 3 paired; 8K/8 control: 1 paired |

The CPU tier is deliberately smaller than the previous 8 GiB setup. At 2 GiB,
it can absorb the 8K working set but cannot retain the complete two-root 16K
working set, forcing real filesystem restoration in the pressure case.

All four runs used identical token counts within each paired cell. Disk roots
were separate and initially empty. Each `.bin` file represents one physical
offload block. The final file count and byte total were independently checked
against the internal physical-write counters.

## Test Method

Each paired measurement follows the same procedure:

1. Create a fresh filesystem root so neither policy inherits cached KV files.
2. Start a new vLLM process with Qwen3-1.7B and wait for the health endpoint.
3. Submit two common/root requests, followed by their case-major sibling
   branches at the configured concurrency.
4. Keep input records, generated-token limits, branch suffixes, GPU block
   count, CPU capacity, filesystem threads, and scheduler settings identical.
5. Scrape Prometheus after all requests finish but before server shutdown.
6. Record request TTFT/TPOT/latency, GPU telemetry at 250 ms intervals, CPU KV
   usage, logical connector traffic, and physical filesystem traffic.
7. Stop the server and independently count final KV files and their bytes.
8. Compare ordinary and cohort-aware policies as a pair. The final 16K/16
   result has three pairs; pair 2 reverses policy execution order.

The baseline is not FlashAttention and does not disable scheduling. It is
**ForkAttention + the same fanout scheduler + ordinary native tiered LRU
offload**. The optimized side changes only cohort-aware offload admission; the
retained safe I/O batching implementation is enabled for both sides.

The primary end-to-end metric is output tokens/s. TTFT isolates initial prefix
availability, while TPOT covers steady decode. Physical Disk-read bytes are
the primary mechanism metric. A performance claim is accepted only when it is
consistent with the relevant tier traffic; Store-only activity is not counted
as evidence that Disk served a request.

## Disk-Tier Correctness and Optimizations

Two disk paths are maintained in this repository, and they serve different
integration modes:

### Native vLLM filesystem tier used by this experiment

- Store and Load callbacks report actual bytes returned by `write`/`readv`.
- Existing KV files skip physical Store and are counted separately from
  logical submitted bytes, exposing the real deduplication ratio.
- Prometheus records physical blocks/bytes, jobs, failures, lookups, in-flight
  work, and submission-to-completion latency by direction.
- Safe batching submits independent per-request Load jobs under one queue lock
  and interleaves their block tasks round-robin. Completion and failure remain
  per request, avoiding cross-request head-of-line blocking.
- The filesystem uses direct I/O where supported. Disk is a capacity tier, not
  an accelerator; a Disk hit is compared with prefix recomputation, not with a
  CPU hit.

Aggressive cross-request completion coalescing was tested and rejected. It
reduced job counts in favorable runs but made request readiness depend on the
slowest sibling and caused recomputation in repeated tests.

### LMCache CPU + Disk correctness path

LMCache eviction mode previously allowed a Disk lookup to promise more chunks
than the CPU staging allocator could hold. A concurrent CPU-to-Disk demotion
could also retain the victim memory briefly, making the first Disk-load staging
allocation fail and producing a partial retrieve.

The fix caps advertised/retrieved chunks to the CPU allocator budget and uses
a bounded staging-allocation retry. The asynchronous wait runs outside the
event-loop thread so pending demotions can complete and release memory. It does
not make Disk faster; it makes the third tier usable without overcommitting the
CPU staging pool.

Validation covers retry-after-demotion, zero-timeout behavior, lookup-budget
capping, independent batched-load failures, physical byte metrics, and Store
deduplication. Focused results are **42 passed** for LMCache storage and
**39 passed** for native vLLM tiering/filesystem tests.

## Instrumentation

The filesystem tier now exports the following labeled Prometheus metrics for
`tier="fs"` and `direction="load|store"`:

- submitted jobs, blocks, and logical bytes;
- physically transferred blocks and bytes;
- deduplicated Store blocks and bytes;
- completed and failed jobs;
- submission-to-completion latency histogram;
- lookup hit, miss, and asynchronous retry observations;
- in-flight jobs.

`store_block()` returns the bytes actually written and reports zero when an
existing physical block file avoids a write. `load_block()` reports bytes
actually read. These counters therefore distinguish policy-level logical
traffic from filesystem-level physical traffic. The existing connector metrics
continue to measure GPU-to-CPU Store and CPU-to-GPU Load traffic.

Metric implementation tests cover physical Store/Load bytes, duplicate Store
suppression, labels, and definitions. The complete tiering test selection
passed: **39 passed**.

## Initial Single-Pair End-to-End Diagnostic

Lower is better for TTFT, TPOT, and latency.

| Prefix / branches | Policy | Output tok/s | Branch TTFT P50 | Branch TPOT P50 | Branch latency P50 |
|---|---|---:|---:|---:|---:|
| 8K / 8 | Ordinary | 288.16 | 291.86 ms | 16.55 ms | 1,334.15 ms |
| 8K / 8 | Cohort-aware | 275.92 | 298.11 ms | 18.19 ms | 1,443.82 ms |
| 16K / 16 | Ordinary | 276.49 | 1,173.98 ms | 12.91 ms | 2,107.25 ms |
| 16K / 16 | Cohort-aware | 273.80 | 1,092.23 ms | 14.15 ms | 2,079.22 ms |

| Prefix / branches | Throughput | TTFT | TPOT | Branch latency |
|---|---:|---:|---:|---:|
| 8K / 8 | **-4.25%** | +2.14% | +9.90% | +8.22% |
| 16K / 16 | **-0.97%** | **-6.96%** | +9.66% | **-1.33%** |

The 16K result separates initial restoration latency from steady decode. The
shared-prefix-aware policy improves TTFT and total branch latency slightly, but
TPOT regresses and offsets the gain at cohort throughput level.

## Initial Single-Pair Per-Tier KV Traffic

All byte values below are GiB (2^30 bytes). “GPU Store” is GPU-to-CPU logical
traffic; “GPU Load” is CPU-to-GPU traffic. Disk columns are actual physical I/O.

| Prefix / branches | Policy | GPU Store | GPU Load | Disk write | Disk read | Store dedup |
|---|---|---:|---:|---:|---:|---:|
| 8K / 8 | Ordinary | 2.123 | 0 | 2.123 | 0 | 0% |
| 8K / 8 | Cohort-aware | 6.075 | 0 | 2.094 | 0 | 65.54% |
| 16K / 16 | Ordinary | 6.648 | 1.781 | 4.155 | 2.475 | 37.51% |
| 16K / 16 | Cohort-aware | 7.549 | 1.101 | 4.098 | 2.321 | 45.71% |

At 16K/16, cohort awareness changes the storage path as follows:

- CPU-to-GPU Load falls from 1.781 to 1.101 GiB: **-38.20%**;
- physical Disk read falls from 2.475 to 2.321 GiB: **-6.22%**;
- physical Disk write falls from 4.155 to 4.098 GiB: **-1.36%**;
- logical GPU Store rises 13.56%, from 6.648 to 7.549 GiB;
- Store dedup rises from 37.51% to 45.71%, preventing the extra logical
  proactive Stores from becoming proportional physical writes.

The final physical disk contents exactly match the transferred-byte counters:
2,431 files / 4.155 GiB for ordinary and 2,398 files / 4.098 GiB for
cohort-aware. This provides an independent check that the byte counters are not
merely logical request sizes.

## Initial Single-Pair I/O Granularity and Latency

| Prefix / branches | Policy | Store jobs / blocks | Avg Store job | Load jobs / blocks | Avg Load job | Failed jobs |
|---|---|---:|---:|---:|---:|---:|
| 8K / 8 | Ordinary | 26 / 1,242 | 106.79 ms | 0 / 0 | - | 0 |
| 8K / 8 | Cohort-aware | 253 / 3,555 logical | 19.76 ms | 0 / 0 | - | 0 |
| 16K / 16 | Ordinary | 50 / 3,890 logical | 111.14 ms | 43 / 1,448 | 78.55 ms | 0 |
| 16K / 16 | Cohort-aware | 292 / 4,417 logical | 37.19 ms | 83 / 1,358 | 164.43 ms | 0 |

The optimized 16K run reads fewer blocks but uses 1.93x as many Load jobs. Its
average job contains 16.4 blocks versus 33.7 for ordinary offload. This
fragmentation explains why lower Disk bytes do not translate into higher
throughput. The latency metric includes queueing and scheduler polling, so it
is a request-path completion latency rather than raw device service time.

## Safe I/O Batching and Benefit Analysis

### Design selected after negative testing

The initial implementation already batches promotion blocks by
`(tier, request)`. Two more aggressive variants were evaluated:

1. **Cross-request completion coalescing:** one job and one completion barrier
   for all sibling requests in a scheduler step.
2. **Bounded cross-request completion coalescing:** the same design, capped at
   64 blocks per job.

Repeated runs showed that both can delay an already-read request behind other
requests in the batch. In the bounded n=3 diagnostic, cohort-aware throughput
was 13.79%, 6.70%, and 0.35% lower/higher than ordinary, respectively; two runs
performed no CPU-to-GPU Load after long Disk promotions and fell back to
recomputation. This design was rejected rather than used as positive evidence.

The retained **safe batching** changes only submission mechanics:

- the manager submits all independent per-request jobs to the filesystem tier
  in one call;
- the thread pool acquires its queue lock once and interleaves block tasks
  round-robin across requests;
- every request keeps an independent completion state and failure boundary;
- a missing block in one job does not fail or delay completion bookkeeping for
  another job.

This removes repeated queue-lock/notification work and improves fairness
without changing when a request becomes ready. OffloadKey identity, CPU slot
allocation, KV contents, and recompute-on-failure behavior remain unchanged.

### Three paired 16K/16 runs

Both policies used the same safe-batching code. Run 2 reversed execution order
(`cohort-aware -> ordinary`) to reduce fixed ordering bias.

| Run | Policy | Output tok/s | TTFT P50 | Disk read | Load jobs | Avg load job | CPU->GPU Load |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Ordinary | 272.84 | 1,286.23 ms | 3.355 GiB | 39 | 105.64 ms | 1.784 GiB |
| 1 | Cohort-aware | 275.26 | 1,179.22 ms | 1.601 GiB | 40 | 35.32 ms | 1.109 GiB |
| 2 | Ordinary | 239.00 | 1,753.77 ms | 2.504 GiB | 22 | 101.47 ms | 0 GiB |
| 2 | Cohort-aware | 236.84 | 1,755.22 ms | 4.117 GiB | 24 | 164.87 ms | 0 GiB |
| 3 | Ordinary | 241.94 | 1,673.31 ms | 2.955 GiB | 30 | 125.11 ms | 0 GiB |
| 3 | Cohort-aware | 236.67 | 1,644.43 ms | 2.068 GiB | 24 | 90.26 ms | 0 GiB |

Mean and sample standard deviation across the three runs are:

| Metric | Ordinary | Cohort-aware | Mean change | Paired direction |
|---|---:|---:|---:|---:|
| Output throughput | 251.26 +/- 18.75 tok/s | 249.59 +/- 22.23 tok/s | -0.66% | improved 1/3 |
| TTFT P50 | 1,571.10 +/- 249.97 ms | 1,526.29 +/- 305.64 ms | **-2.85%** | improved 2/3 |
| TPOT P50 | 13.68 +/- 0.45 ms | 14.72 +/- 1.43 ms | +7.54% | improved 1/3 |
| Branch latency P50 | 2,492.63 +/- 317.42 ms | 2,520.70 +/- 353.84 ms | +1.13% | improved 1/3 |
| Physical Disk read | 2.938 +/- 0.426 GiB | 2.595 +/- 1.338 GiB | **-11.65%** | improved 2/3 |
| Load jobs | 30.33 +/- 8.50 | 29.33 +/- 9.24 | -3.30% | fewer 1/3 |
| Avg load-job latency | 110.74 +/- 12.62 ms | 96.82 +/- 65.02 ms | -12.57% | improved 2/3 |
| GPU-to-CPU Store | 6.370 +/- 1.728 GiB | 5.578 +/- 1.376 GiB | -12.44% | lower 2/3 |
| I/O failures | 0 | 0 | no change | 3/3 clean |

The standard deviations are large relative to the mean changes. With only
three pairs, confidence intervals would span zero for the end-to-end metrics.
Accordingly, the table establishes observed direction and variance; it does
not establish statistical significance.

### What benefit is directly supported

The strongest benefit is lower L3 read volume in the target regime:

- mean Disk read falls by 0.342 GiB per two-root cohort;
- normalized by 32 branches, it falls from 94.01 to 83.05 MiB per branch;
- normalized by 2,176 generated tokens, it falls from 1.382 to 1.221 MiB per
  output token;
- mean TTFT is 44.81 ms lower, consistent with less initial storage pressure.

This benefit does **not** yet become stable throughput improvement. Mean
throughput is 0.66% lower, TPOT is 7.54% higher, and the benefit direction is
not consistent in every pair. The likely remaining causes are asynchronous
CPU eviction timing and the much larger number of proactive Store submissions;
these are hypotheses consistent with the counters, not directly proven causes.

The 8K/8 negative control strengthens the regime claim: it performs zero Disk
loads and cohort-aware throughput is 2.10% lower in the safe-completion
single-run comparison. Proactive backup should therefore be gated off when no
later L3 restore is predicted.

Output agreement for the three final pairs is 32/34, 33/34, and 33/34 exact,
with mean character similarities of 96.67%, 97.72%, and 98.95%. This is
consistent with concurrent greedy-serving variation; the batching path does
not modify KV data or model execution.

### Evidence triangulation

Several independent observations support the storage-volume claim:

- workload token counts and request counts are identical within each pair;
- both policies use the same model, ForkAttention backend, scheduler, CPU/Disk
  capacities, filesystem worker counts, and batching implementation;
- non-zero physical Load counters prove that L3 was used, while zero failed
  jobs rule out error-driven byte differences;
- `transferred_blocks * 1,835,008 bytes` exactly equals the reported physical
  read bytes in every run;
- the 8K control has zero Disk reads and no corresponding performance benefit;
- physical filesystem file totals independently match Store-byte counters.

These checks make the byte-level mechanism credible. They do not remove the
need for more repetitions and controlled eviction traces before making an
end-to-end acceleration claim.

Lookup observations provide additional evidence that the filesystem tier is
active:

| Prefix / branches | Policy | FS hit | FS miss | Async retry observations |
|---|---|---:|---:|---:|
| 8K / 8 | Ordinary | 0 | 18 | 1,255 |
| 8K / 8 | Cohort-aware | 0 | 18 | 2,818 |
| 16K / 16 | Ordinary | 1,498 | 1,280 | 8,074 |
| 16K / 16 | Cohort-aware | 1,358 | 1,145 | 6,865 |

Retries are polling observations while asynchronous existence checks are in
flight; they are not unique blocks and must not be interpreted as requests.

## Initial Single-Pair Capacity and Hardware Telemetry

| Prefix / branches | Policy | Peak GPU KV usage | Peak CPU active usage | Peak CPU occupancy | Peak GPU memory | Mean GPU compute | Mean memory BW |
|---|---|---:|---:|---:|---:|---:|---:|
| 8K / 8 | Ordinary | 76.69% | 43.76% | 100% | 8,810 MiB | 81.23% | 44.62% |
| 8K / 8 | Cohort-aware | 76.99% | 21.88% | 100% | 9,021 MiB | 88.62% | 39.77% |
| 16K / 16 | Ordinary | 75.75% | 100% | 100% | 9,186 MiB | 83.64% | 36.71% |
| 16K / 16 | Cohort-aware | 75.46% | 53.76% | 100% | 8,947 MiB | 94.93% | 45.00% |

CPU occupancy reaches 100% because the allocator has assigned the bounded
offload region; active usage reflects blocks still considered useful. GPU KV
peak is essentially unchanged, which is expected: proactive Store creates a
backup and does not evict the active GPU copy. The optimization targets future
reload/recompute cost, not immediate GPU capacity reduction.

## Interpretation and Next Optimization

The experiment supports three claims:

1. The three-level implementation is functional: 16K/16 performs non-zero,
   successful Disk-to-CPU restores with byte-accurate physical counters.
2. Across the final three pairs, cohort awareness reduces mean Disk-read volume
   by 11.65% and mean TTFT by 2.85%, but neither direction is universal.
3. Merging completion across requests is unsafe because it introduces
   head-of-line blocking. Batching only queue submission preserves correctness,
   but does not by itself establish an end-to-end speedup.

The next low-risk optimization should focus on admission and Store-side
coalescing:

- gate proactive backup on predicted CPU eviction and expected sibling reuse;
- apply the same safe batching principle to proactive Store submissions;
- impose minimum blocks/bytes per proactive job;
- cap outstanding Store jobs so they cannot delay a later Load;
- prioritize active-cohort Loads over speculative Stores;
- record unique physical keys and backup-ready-before-eviction rate.

The final target cell has three pairs, which is enough to reveal variance but
not enough for a narrow confidence interval. A report-quality evaluation should
use more roots and repetitions, randomize every policy order, pin or record
filesystem/device state, and include a device-level I/O trace. The present
result should be described as a conditional storage-traffic reduction, not an
end-to-end acceleration of Disk offload.

## Artifacts and Reproduction

Raw API responses, per-request CSVs, server logs, telemetry, Prometheus dumps,
and physical KV files are retained under:

```text
benchmark/results/offload_l3_detailed_20260718/
benchmark/results/offload_l3_iomerge_20260718/
benchmark/results/offload_l3_bounded_iomerge_20260718/
benchmark/results/offload_l3_safe_batch_20260718/
```

The essential connector configuration is:

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use": 2147483648,
    "block_size": 16,
    "eviction_policy": "lru",
    "offload_prompt_only": true,
    "fanout_offload": true,
    "fanout_allow_hot_prefix_backup": true,
    "spec_name": "TieringOffloadingSpec",
    "secondary_tiers": [{
      "type": "fs",
      "root_dir": "/absolute/path/to/kv_fs",
      "n_read_threads": 4,
      "n_write_threads": 4
    }]
  }
}
```

For the ordinary baseline, set `fanout_offload=false` and disable the other
fanout-specific offload options. Keep ForkAttention fanout scheduling enabled
for both policies so that the comparison changes only the offload policy.
