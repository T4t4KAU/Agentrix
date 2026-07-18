# ForkAttention Cohort-Aware KV Offload and Eviction

## Current Status

This document supersedes the earlier proactive-backup-only report and its old
measurements. The comparison in this revision contains exactly one baseline:

- **Baseline:** ForkAttention + native three-level offload + ordinary CPU LRU.
- **Optimized:** the identical stack, with CPU eviction changed to
  `cohort_lru`.

Both sides enable the same fanout scheduler and proactive backup path so that
the experiment isolates eviction order. There is intentionally no
FlashAttention + offload result: changing both the attention operator and the
offload policy would confound the comparison and can be misleading.

The new 16K-prefix, 16-sibling-per-root pressure experiment confirms that the
policy changes cache retention behavior, but **does not yet establish an
end-to-end performance gain**. Compared with ordinary LRU, `cohort_lru`:

- reduced logical GPU-to-CPU Store traffic by **26.79%**;
- reduced Store-admission failures from 28 to 16 (**-42.86%**);
- raised the reported external prefix-cache hit rate from 16.3% to 19.2%
  (**+2.9 percentage points**);
- but increased physical filesystem reads by **19.59%** and reduced end-to-end
  throughput by **8.91%** in this single paired run.

The feature is therefore opt-in and ordinary `lru` remains the default. The
current result is a useful negative boundary: preserving shared prefix blocks
is not sufficient by itself; the eviction score must also account for the cost
and probability of a later Disk restore.

## Motivation

Ordinary LRU knows only when a CPU KV block was last touched. In a
ForkAttention workload, equal-age blocks can have very different future value:
a long root-prefix block may be shared by 16 live siblings, while a suffix
block may belong to one nearly finished request. Evicting the root first can
destroy many potential reuse opportunities.

The existing fanout policy addresses a different decision: **when and what to
back up from GPU to CPU**. The new policy addresses **which CPU-resident block
to remove when the bounded CPU tier is full**. The two mechanisms are
complementary:

```text
fanout planner: GPU -> CPU Store admission and timing
cohort_lru:     victim selection inside the full CPU tier
filesystem:    CPU overflow/capacity tier, not a faster execution tier
```

## Design

### Metadata flow

The scheduler continuously reports metadata for blocks that have already been
admitted, rather than stopping observation after the first Store. For each
physical KV key, the CPU offload manager can receive:

- lifecycle: `HOT`, `COOLING`, or `COLD`;
- current reuse count;
- sibling fanout;
- residency steps;
- normalized prefix position;
- whether a valid copy is known to exist in the secondary tier.

Secondary-backed state is recorded only after a filesystem Store succeeds. A
successful secondary promotion restores that state. This avoids preferring a
victim on the false assumption that Disk already contains a recoverable copy.

### Victim ordering

Only blocks with `ref_cnt == 0` are evictable. `cohort_lru` sorts eligible
victims lexicographically, from cheapest to retain to most valuable to retain:

1. unobserved/cold before cooling before hot;
2. later prefix position before earlier root-prefix position;
3. secondary-backed before blocks without a verified Disk copy;
4. lower current reuse before higher reuse;
5. lower fanout before higher fanout;
6. shorter residency before longer residency;
7. ordinary LRU rank as the deterministic final tie-breaker.

This order deliberately protects active, early, high-fanout root prefixes and
evicts cold/private suffixes first. It does not pin blocks: under sufficient
pressure every unreferenced block remains evictable.

An experimental root-atomic variant was rejected. Evicting a whole root at
once amplified churn and head-of-line effects in the pressure benchmark. The
retained implementation ranks individual physical blocks and uses cohort
metadata only as priority signals.

### Complexity and compatibility

The policy is implemented as a separate CPU eviction policy and selected by
configuration. It does not change model code, attention kernels, KV tensor
layout, or the default LRU path. Metadata updates are dictionary operations;
victim selection sorts only the currently evictable CPU blocks. Unknown or
partially observed blocks fall back naturally to LRU tie-breaking.

## How to Enable

The default is unchanged:

```json
{
  "kv_connector_extra_config": {
    "eviction_policy": "lru"
  }
}
```

Enable cohort-aware eviction in the native vLLM `OffloadingConnector` with:

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 2147483648,
    "block_size": 16,
    "eviction_policy": "cohort_lru",
    "offload_prompt_only": true,
    "fanout_offload": true,
    "fanout_profile": true,
    "fanout_allow_hot_prefix_backup": true,
    "secondary_tiers": [
      {
        "type": "fs",
        "root_dir": "/path/to/kv-cache",
        "n_read_threads": 4,
        "n_write_threads": 4
      }
    ]
  }
}
```

For a clean baseline, keep every field identical and change only:

```json
"eviction_policy": "lru"
```

`fanout_offload` should not be disabled in this A/B test. Disabling it would
simultaneously change Store admission/timing and eviction, making it impossible
to attribute the result to the new eviction policy. In this report, “ordinary
offload” means the ordinary CPU LRU replacement policy within the otherwise
identical ForkAttention offload stack.

## Experimental Setup

| Setting | Value |
|---|---|
| Validation date | 2026-07-18 |
| GPU | NVIDIA GeForce RTX 5070, 12,227 MiB |
| Model | local Qwen3-1.7B, FP16 |
| Attention backend | ForkAttention on both sides |
| Root cases | 2 |
| Siblings | 16 per root, 32 branch requests total |
| Configured prefix | 16,384 tokens |
| Realized prefix | 16,454 tokens including template overhead |
| Realized mean suffix | 1,067.9 tokens |
| Output | 64 common + 64 per branch |
| Request order / concurrency | case-major / 32 |
| GPU memory utilization | 0.60 |
| Max model length / sequences | 21,888 / 32 |
| GPU KV capacity | 32,480 baseline; 32,560 optimized tokens |
| CPU KV tier | 2 GiB pinned memory |
| KV block size | 16 tokens |
| Secondary tier | filesystem, 4 read + 4 write threads |
| Prefix caching | enabled |
| Fanout backup | enabled identically on both sides |
| Changed variable | `lru` versus `cohort_lru` only |
| Repetitions | one paired diagnostic run |

The small difference in reported GPU KV capacity comes from separate engine
starts and is disclosed rather than normalized away. It slightly favors the
optimized run, so it cannot explain its observed slowdown.

### Method

1. Start a fresh vLLM process from the same source revision and an empty,
   policy-specific filesystem root.
2. Submit the two roots and their 16 siblings in case-major order.
3. Hold model, prompts, outputs, concurrency, CPU capacity, filesystem worker
   count, fanout configuration, and scheduler configuration fixed.
4. Change only `eviction_policy` between `lru` and `cohort_lru`.
5. Scrape Prometheus after all requests complete and before shutdown.
6. Collect streaming latency, wall time, connector bytes/operations,
   filesystem physical bytes/jobs/latency, cache hit rate, preemptions, and
   Store-admission warnings.

No warm shared-prefix pass was used. Disk roots began empty. Thus filesystem
Load bytes represent restoration caused within the measured run, not a cache
primed by an earlier run.

## New Results

### End-to-end performance

Lower is better for latency; higher is better for throughput.

| Metric | ForkAttention + ordinary LRU | ForkAttention + `cohort_lru` | Change |
|---|---:|---:|---:|
| E2E output throughput | 191.40 tok/s | 174.34 tok/s | **-8.91%** |
| Branch throughput | 339.64 tok/s | 288.83 tok/s | **-14.96%** |
| E2E wall time | 10,699.95 ms | 11,747.12 ms | +9.79% |
| Common latency | 4,493.52 ms | 4,507.60 ms | +0.31% |
| Branch mean latency | 4,636.85 ms | 5,370.50 ms | +15.82% |
| TTFT mean | 2,413.74 ms | 2,799.10 ms | +15.97% |
| TTFT P50 | 2,471.70 ms | 2,884.84 ms | +16.72% |
| TTFT P95 | 4,346.32 ms | 5,106.54 ms | +17.49% |
| TPOT mean | 35.29 ms | 40.82 ms | +15.67% |
| TPOT P95 | 44.19 ms | 52.70 ms | +19.27% |
| Preemptions | 0 | 0 | unchanged |

This single pair is diagnostic, not a statistically powered throughput result.
The direction is nevertheless unambiguously negative in this run, so the
report makes no acceleration claim.

### KV and filesystem traffic

Logical connector traffic describes GPU/CPU transfers and Store submissions.
Filesystem traffic below is the actual byte count reported by completed I/O.

| Metric | Ordinary LRU | `cohort_lru` | Change |
|---|---:|---:|---:|
| GPU-to-CPU logical Store | 15,087,435,776 B | 11,044,913,152 B | **-26.79%** |
| CPU-to-GPU Load | 1,045,954,560 B | 1,157,890,048 B | +10.70% |
| Physical Disk write | 5,262,802,944 B | 5,262,802,944 B | 0.00% |
| Physical Disk read | 3,813,146,624 B | 4,559,994,880 B | **+19.59%** |
| Disk Load jobs | 40 | 53 | +32.50% |
| Mean Disk Load completion | 463.92 ms | 625.56 ms | +34.84% |
| Store-admission failures | 28 | 16 | **-42.86%** |
| External prefix-cache hit rate | 16.3% | 19.2% | **+2.9 pp** |

The equal 5.263 GB physical write volume shows that the final unique on-disk
working set was effectively the same. The lower logical Store volume and fewer
admission failures show that `cohort_lru` changed CPU residency and reduced
repeated Store pressure. However, it selected a residency mix that later
required 13 more filesystem Load jobs and 747 MB more reads. That additional
L3-to-L2 restoration is consistent with the TTFT/TPOT regression.

Disk is not expected to accelerate a CPU hit. It exists to avoid recomputing KV
that no longer fits in GPU or CPU. Therefore a useful policy must minimize the
sum of recomputation and restore cost, not maximize Disk hits. In this run,
`cohort_lru` overprotected high-fanout prefix blocks while underestimating the
near-term restore cost of displaced blocks.

## Policy-Level Unit Experiment

A deterministic capacity test isolates victim ordering without model or I/O
noise. The CPU cache contains 16 old `HOT` shared-prefix keys and 48 cold keys;
32 new keys are then inserted into a 64-key cache.

| Policy | Shared HOT keys evicted | Shared HOT keys retained | Potential sibling hit opportunities |
|---|---:|---:|---:|
| LRU | 16 / 16 | 0 / 16 | 0 / 256 |
| `cohort_lru` | 0 / 16 | 16 / 16 | 256 / 256 |

This proves the implementation performs the intended prioritization. The
end-to-end result above separately proves that the current heuristic is not yet
cost-optimal under three-level pressure.

## Validation

Focused vLLM tests cover policy registration, default-LRU compatibility,
lifecycle ordering, prefix-position ordering, verified-secondary preference,
fanout/reuse/residency ordering, reference-count safety, metadata updates, and
tiering callbacks.

```text
119 passed, 2 existing warnings
ruff-check: passed
ruff-format: passed
git diff --check: passed
```

## Interpretation and Next Steps

The new design fixes the missing mechanism in the earlier implementation:
cohort awareness now influences actual CPU victim selection, not only early
backup. It is minimally invasive and safe to disable. The experiments support
three narrower conclusions:

1. ForkAttention cohort metadata can preserve shared root prefixes under CPU
   pressure; the policy-level test demonstrates this directly.
2. The real workload shows less Store churn and a higher external hit rate.
3. Those benefits do not outweigh increased Disk restore traffic in the tested
   case, so `cohort_lru` must remain opt-in.

The next revision should add a restore-cost gate to the score: protect a shared
prefix only when predicted future sibling savings exceed the expected Disk
read or recomputation cost of its victim. Repeated, order-reversed trials should
then cover multiple CPU capacities and prefix/fanout cells before considering a
default change.
