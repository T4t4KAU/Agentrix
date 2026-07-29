# Branch-Aware GPU KV Lifecycle Scheduling and Reload Experiment

## Current Status

This document describes the current branch-aware KV scheduling path and
supersedes the earlier offload-restart and CPU/Disk eviction report.

The current optimization targets a narrower and directly measurable problem:
when GPU prefix-cache capacity is under pressure, retain high-value shared
Agent KV on GPU, evict cold private KV first, and reduce later
CPU/offload-medium-to-GPU reload work.

The controlled three-run A/B experiment confirms that the scheduler changes
GPU victim selection and improves the later multi-branch revisit:

- reloaded KV blocks fall from 298 to 42 (**-85.91%**);
- CPU-to-GPU traffic falls from 521.5 MiB to 73.5 MiB (**-85.91%**);
- measured connector load time falls from 13.79 ms to 2.13 ms
  (**-84.54%**);
- revisit P95 TTFT falls from 81.3 ms to 54.2 ms (**-33.28%**);
- revisit branch throughput rises from 98.6 to 146.5 turns/s
  (**+48.52%**).

The experiment uses CPU as the offload medium. It does not claim a Disk-tier
result. Both modes use the same CPU `cohort_lru` policy, model, request order,
cache sizes, and attention backend. The only A/B variable is lifecycle-aware
GPU idle-prefix eviction.

## Problem

Native GPU prefix-cache LRU knows when an idle physical block entered the free
queue, but it does not know the Agent structure that produced the block. Two
equally idle blocks can have very different future value:

- a 4K root may be shared by eight branches and reused by the next Agent turn;
- a suffix may be private to one completed branch;
- a new pressure request may have no observed reuse at all.

Ordinary LRU can evict the shared root because it is older. If its CPU copy is
still available, the next branch wave reloads the root. If the CPU copy has
also disappeared, the root must be recomputed. Both outcomes discard
cross-branch reuse already identified by the fanout scheduler.

The optimization carries branch-derived lifecycle value to the actual GPU
victim-selection boundary.

## Current Scheduling Design

### End-to-end metadata flow

For every observed offload key, the scheduler derives:

- lifecycle state: `HOT`, `COOLING`, or `COLD`;
- current and historical branch fanout;
- reuse score;
- residency value;
- normalized prefix position;
- every physical GPU block ID currently representing the key.

The same lifecycle observation can inform two separate decisions:

```text
request and branch observations
              |
              v
      lifecycle classification
        HOT / COOLING / COLD
          |                 |
          v                 v
GPU -> CPU Store plan   idle GPU victim selection
          |                 |
          v                 v
 CPU cohort_lru        COLD before shared HOT
```

The CPU policy controls which offloaded copies survive in the bounded CPU
tier. The new GPU policy controls which idle cached blocks are reclaimed for
new allocations. The experiment in this document holds the CPU policy fixed
and isolates the second decision.

### Lifecycle definition

A block becomes `HOT` when current fanout and reuse observations meet the
configured shared-prefix thresholds. When the hot condition disappears, the
block enters `COOLING` for the configured cooldown interval. It becomes
`COLD` only after that interval expires.

This transition is important for Agent workloads. A shared prefix can have no
live request while a tool runs or while the scheduler processes unrelated
work, yet still be valuable to the next turn. Immediate HOT-to-COLD
transition would collapse the useful retention window.

### GPU victim ordering

Lifecycle-aware allocation remains restricted to blocks that are already
idle (`ref_cnt == 0`). Active request KV is never selected.

Eligible cached blocks are ranked from first to evict to last to evict:

1. unclassified and `COLD` before `COOLING` before `HOT`;
2. later suffix position before earlier root-prefix position;
3. lower reuse score before higher reuse score;
4. lower fanout before higher fanout;
5. lower residency value before higher residency value;
6. ordinary free-queue LRU order as the deterministic tie-breaker.

Unhashed free blocks are still allocated before cached blocks. Lifecycle
priority is soft rather than a pin: if all lower-value candidates are
exhausted, an idle HOT block can still be reclaimed, so allocation cannot
deadlock.

The normal linked-list LRU path remains unchanged when lifecycle-aware GPU
eviction is disabled.

### Lifecycle after request release

The scheduler retains lifecycle metadata after the final request releases a
cached prefix. This is the state needed during an Agent tool gap: the request
is gone, but its cached root may still be reused.

Each remembered physical block ID is stored with its expected block hash.
Before publishing an eviction priority, the scheduler validates that the
physical block still contains that hash. A block that has been recycled for a
different prefix cannot inherit stale HOT or COOLING state.

Idle metadata continues to age:

```text
HOT at shared branch execution
  -> COOLING while idle but within TTL
  -> COLD after the cooldown interval
```

### Post-load minimum residency

When an offload load completes, its destination GPU blocks receive a short
soft minimum-residency window. This prevents a just-loaded prefix from being
selected immediately by the next allocation and avoids reload/evict/reload
thrashing.

The protection uses the existing
`fanout_hot_prefix_min_residency_steps` setting. It expires automatically and
remains reclaimable under hard capacity pressure.

### Concurrent reload coalescing

The connector tracks offload keys currently being loaded. If another request
needs the same key, it waits behind that in-flight load and then rechecks the
local prefix cache. It does not issue a second physical transfer for the same
shared root.

This explains why the measured load-operation reduction is smaller than the
block reduction. The baseline already loads the 256-block root in one
coalesced operation. Retaining the root removes one operation but removes 256
individual block transfers.

### Metrics

The connector now reports direct lifecycle-aware GPU and reload metrics:

| Metric | Meaning |
|---|---|
| `vllm:kv_offload_load_bytes` | Bytes loaded from the offload medium to GPU |
| `vllm:kv_offload_load_time` | Connector load time |
| `vllm:kv_offload_load_size` | Load-operation histogram; `_count` is operation count |
| `vllm:kv_offload_reload_blocks{lifecycle}` | Reloaded blocks classified at load admission |
| `vllm:kv_offload_gpu_evicted_blocks{lifecycle}` | GPU cached blocks evicted by lifecycle |
| `vllm:kv_offload_gpu_resident_blocks{lifecycle}` | Current classified GPU residents |
| `vllm:kv_offload_load_protected_blocks` | Loaded blocks granted minimum residency |
| `vllm:kv_offload_coalesced_load_waits` | Requests waiting behind an in-flight shared load |

These metrics are recorded at scheduler-stat intervals, so one interval can
contain multiple completed loads.

## Configuration

Enable the current scheduler in the native vLLM `OffloadingConnector`:

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "cpu_bytes_to_use": 2147483648,
    "eviction_policy": "cohort_lru",
    "fanout_offload": true,
    "fanout_gpu_lifecycle_eviction": true,
    "fanout_profile": true,
    "fanout_budget_blocks": 512,
    "fanout_allow_hot_prefix_backup": true,
    "fanout_hot_prefix_min_fanout": 4,
    "fanout_hot_prefix_min_reuse_blocks": 128,
    "fanout_hot_prefix_min_residency_steps": 4,
    "fanout_hot_prefix_cooldown_steps": 1024,
    "fanout_high_pressure_threshold": 0.9,
    "fanout_critical_pressure_threshold": 0.97
  }
}
```

For the clean GPU-policy baseline, keep every field identical and change only:

```json
"fanout_gpu_lifecycle_eviction": false
```

The option defaults to enabled when fanout offload is enabled, but it has no
effect when the connector does not bind a GPU block pool. Disabling it
preserves ordinary GPU LRU behavior.

## Experiment

### Question

Under a fixed GPU KV budget, does branch-aware GPU victim selection keep the
shared root resident during unrelated pressure and reduce the next
multi-branch turn's physical reload work?

### Agent workflow

Each run uses deterministic valid token IDs and the following controlled
workflow:

```text
4K shared root
  -> fork to 8 running branches: root becomes HOT
  -> retire 6 branches and keep 2 survivors
  -> run six independent 2K pressure prompts
  -> re-fork the original root to 8 branches
  -> finish every shared branch: root becomes idle
  -> run six new independent 2K prompts to force GPU eviction
  -> age the cache with short unrelated pulses
  -> revisit the root with 8 short branches
```

The post-finish pressure phase is essential. The earlier workflow applied
pressure while two root-sharing requests were still active, so the root could
not legally be evicted and reload remained zero.

### Capacity isolation

A preliminary 0.5 GiB CPU-cache run forced the shared root out of both GPU and
CPU, producing recomputation rather than CPU-to-GPU reload. The formal
comparison uses a 2 GiB CPU cache so the offload copy remains available while
the GPU copy is evicted.

This isolates the intended path:

```text
GPU victim choice -> CPU copy remains -> physical CPU-to-GPU reload or GPU hit
```

### Setup

| Setting | Value |
|---|---|
| Validation date | 2026-07-29 |
| GPU | NVIDIA GeForce RTX 5070, 12,227 MiB |
| Driver | 590.48.01 |
| Model | local Qwen3-0.6B, FP16 |
| Attention backend | FlashAttention on both sides |
| Shared root | 4,096 tokens / 256 KV blocks |
| Branch suffix | 128 tokens |
| Maximum branches / survivors | 8 / 2 |
| Initial pressure | 6 prompts × 2,048 tokens |
| Post-finish pressure | 6 new prompts × 2,048 tokens |
| GPU KV capacity | 512 blocks |
| CPU offload cache | 2 GiB |
| KV block size | 16 tokens |
| Cooling interval | 1,024 scheduler steps |
| Re-fork output | 8 tokens per new branch |
| Revisit output | 1 token per branch |
| GPU sample interval | 50 ms |
| CPU eviction policy | `cohort_lru` in both modes |
| Changed variable | GPU LRU versus lifecycle-aware GPU eviction |
| Repetitions | 3 per mode |

The controlled token workload is intentional. It holds shared-root length,
fanout, pressure volume, and reuse timing constant. AppWorld is the appropriate
next workload for measuring the distribution of these effects across real
Agent trajectories, not for establishing the victim-selection mechanism.

## Results

Values are mean ± sample standard deviation over three matched runs:

| Metric | Original GPU LRU | Lifecycle-aware GPU eviction | Change |
|---|---:|---:|---:|
| CPU-to-GPU load operations | 7.0 ± 0.0 | 6.0 ± 0.0 | -14.29% |
| CPU-to-GPU load volume | 521.5 ± 0.0 MiB | 73.5 ± 0.0 MiB | **-85.91%** |
| Connector load time | 13.79 ± 0.25 ms | 2.13 ± 0.01 ms | **-84.54%** |
| Reloaded KV blocks | 298.0 ± 0.0 | 42.0 ± 0.0 | **-85.91%** |
| Reloaded COOLING blocks | 256.0 ± 0.0 | 0.0 ± 0.0 | **-100.00%** |
| Reloaded COLD blocks | 42.0 ± 0.0 | 42.0 ± 0.0 | 0.00% |
| Revisit P95 TTFT | 81.3 ± 6.2 ms | 54.2 ± 0.8 ms | **-33.28%** |
| Revisit branch throughput | 98.6 ± 7.9 turns/s | 146.5 ± 1.9 turns/s | **+48.52%** |

In every optimized run, the scheduler recorded:

- 1,507 COLD GPU-eviction occurrences;
- zero COOLING GPU-eviction occurrences;
- zero HOT GPU-eviction occurrences.

These are eviction occurrences, not unique keys. Cold pressure blocks can
cycle repeatedly through the fixed 512-block GPU pool.

The direct mechanism is visible in the reload composition:

```text
Original GPU LRU:
  256 COOLING shared-root blocks + 42 COLD suffix blocks = 298 reloads

Lifecycle-aware GPU eviction:
  0 shared-root blocks + 42 COLD suffix blocks = 42 reloads
```

The optimized policy retains the shared root, so all eight revisit branches
hit it in GPU prefix cache. Only their branch-specific suffix KV is restored.

The raw per-run table is
[`experiment_results/gpu_lifecycle_reload.csv`](experiment_results/gpu_lifecycle_reload.csv).

## Actual Memory-Event Timelines

The following figures are reconstructed from events emitted on the actual
cache implementation path, rather than from a schematic model:

- the first lane is the physical ForkAttention query cohort size per step;
- the second lane is the number of full KV blocks currently cached on GPU;
- the third lane marks GPU-to-CPU stores, GPU evictions, and CPU-to-GPU loads;
- the fourth lane is the number of KV blocks resident in the CPU offload cache;
- gold denotes the shared prefix, distinct colors denote branch suffixes, and
  gray denotes cold pressure traffic.

A GPU-to-CPU store creates an offload copy; it does not immediately remove the
GPU copy. Therefore, seeing the same color in both residency lanes means that
the KV is available in both tiers. The figure counts full, hash-addressable
prefix-cache blocks; partial active blocks are intentionally excluded. The
operator lane is taken from the physical ForkAttention plan and reports the
maximum number of queries served by one shared CTA, not the number of requests
merely present in the scheduler.

The query-aggregation stress case interleaves eight shared-prefix branches
with eight private requests and limits the engine to eight sequence slots.
This makes FCFS split the shared cohort while preserving the same root, suffix,
GPU capacity, CPU capacity, and pressure phases. Three repeats produced:

| Mode | Reload | Active shared-CTA cohort | Max cohort | Revisit P95 TTFT | Revisit goodput |
|---|---:|---:|---:|---:|---:|
| GPU LRU + FCFS | 521.5 MiB | 3.98 ± 0.02 | 4.0 ± 0.0 | 272.4 ± 7.4 ms | 291.6 ± 2.1 tok/s |
| Lifecycle only | 73.5 MiB | 3.97 ± 0.00 | 4.0 ± 0.0 | 281.5 ± 0.2 ms | 277.8 ± 0.4 tok/s |
| Lifecycle + hot-prefix query join | 73.5 MiB | **8.00 ± 0.00** | **8.0 ± 0.0** | **54.4 ± 7.0 ms** | **481.7 ± 31.9 tok/s** |

Lifecycle-only is an important ablation: retaining the prefix removes reload
traffic but does not by itself prevent FCFS from splitting the eight queries
into two four-query cohorts. Query join uses the resident HOT/COOLING prefix's
historical fanout as a bounded arrival hint, waits at most two scheduler steps,
then promotes ready reload siblings together. The raw 12-run table is
[`experiment_results/fork_query_aggregation.csv`](experiment_results/fork_query_aggregation.csv).
The query-join-only ablation reached maximum cohorts of 4, 8, and 7 across its
three repeats: once the prefix is evicted, there is no resident lifecycle hint
to make the arrival window deterministic. It is therefore not presented as a
standalone improvement.

### Baseline: Original GPU LRU

![Baseline actual KV memory-event timeline](assets/kv_memory_timeline_baseline.png)

After the branches become idle, cold pressure evicts all 256 shared-prefix
blocks from GPU. The CPU copy remains available, but revisit must load the
entire shared prefix plus 42 branch blocks: 298 blocks, 521.5 MiB, and seven
load operations.

### Optimized: Lifecycle-Aware Eviction and Query Join

![Optimized actual KV memory-event timeline](assets/kv_memory_timeline_optimized.png)

The lifecycle-aware policy directs eviction to cold pressure and branch
suffixes while the shared prefix stays on GPU. Revisit therefore loads only
42 branch blocks: 73.5 MiB and six load operations. The same HOT/COOLING
metadata supplies an expected fanout of eight, so the bounded join window
forms one eight-query physical ForkAttention cohort instead of two
four-query cohorts.

## Validation

Relevant vLLM tests cover:

- lifecycle ordering and original-LRU fallback;
- COLD-before-COOLING-before-HOT victim selection;
- prefix-position and LRU tie-breaking;
- minimum post-load residency and protection expiry;
- physical block-hash validation after request release;
- HOT-to-COOLING-to-COLD aging of idle cached prefixes;
- scheduler-to-GPU-block-pool binding and opt-out;
- completed-load protection;
- concurrent lookup suppression for the same prefix;
- ready-reload cohort promotion and bounded HOT-prefix arrival waiting;
- physical shared-query and maximum-CTA-cohort telemetry;
- metric serialization and reduction.

Validation result:

```text
198 passed, 17 dependency deprecation warnings
ruff-check: passed
python compile checks: passed
git diff --check: passed
```

The missing `tblib` test dependency was installed into `vllm/.venv` from the
Tsinghua PyPI mirror. The warnings are existing third-party deprecation
warnings from Torch, SWIG, Transformers, and tokenization dependencies.

## Reproduction

On the current machine, set the source and dependency paths:

```bash
export PYTHONPATH="$PWD/vllm:$PWD/vllm/.venv.broken-root-20260715/lib/python3.12/site-packages"
```

Run three baseline trials:

```bash
for trial in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 vllm/.venv/bin/python \
    benchmark/scripts/benchmark_fanout_lifecycle_timeline.py \
    --policy cohort_lru \
    --no-gpu-lifecycle-eviction \
    --attention-backend FORK_ATTN \
    --no-fork-query-join \
    --num-gpu-blocks 512 \
    --max-num-seqs 8 \
    --cpu-cache-gib 2.0 \
    --hot-prefix-cooldown-steps 1024 \
    --pressure-sessions 6 \
    --post-finish-pressure-sessions 6 \
    --wave-output-tokens 8 \
    --revisit-output-tokens 16 \
    --revisit-distractors 8 \
    --gpu-sample-ms 50 \
    --output "benchmark/results/gpu_lifecycle_reload/baseline_${trial}.json"
done
```

Run three optimized trials with the same arguments:

```bash
for trial in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 vllm/.venv/bin/python \
    benchmark/scripts/benchmark_fanout_lifecycle_timeline.py \
    --policy cohort_lru \
    --gpu-lifecycle-eviction \
    --attention-backend FORK_ATTN \
    --fork-query-join \
    --num-gpu-blocks 512 \
    --max-num-seqs 8 \
    --cpu-cache-gib 2.0 \
    --hot-prefix-cooldown-steps 1024 \
    --pressure-sessions 6 \
    --post-finish-pressure-sessions 6 \
    --wave-output-tokens 8 \
    --revisit-output-tokens 16 \
    --revisit-distractors 8 \
    --gpu-sample-ms 50 \
    --output "benchmark/results/gpu_lifecycle_reload/optimized_${trial}.json"
done
```

Generate the two actual memory-event timelines:

```bash
benchmark/.venv/bin/python \
  benchmark/scripts/plot_kv_memory_event_timeline.py \
  --baseline benchmark/results/gpu_lifecycle_reload/baseline_1.json \
  --optimized benchmark/results/gpu_lifecycle_reload/optimized_1.json \
  --baseline-output docs/assets/kv_memory_timeline_baseline.png \
  --optimized-output docs/assets/kv_memory_timeline_optimized.png
```

## Implementation Map

| Area | Source |
|---|---|
| GPU lifecycle metadata and ranked victim heap | [`../vllm/vllm/v1/core/block_pool.py`](../vllm/vllm/v1/core/block_pool.py) |
| Scheduler lifecycle publication, idle aging, and hot-prefix fanout hint | [`../vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py`](../vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py) |
| Query admission, reload regrouping, and bounded join | [`../vllm/vllm/v1/core/sched/scheduler.py`](../vllm/vllm/v1/core/sched/scheduler.py) |
| Physical query-cohort construction and telemetry | [`../vllm/vllm/v1/attention/backends/fork_attn.py`](../vllm/vllm/v1/attention/backends/fork_attn.py) |
| GPU block-pool binding | [`../vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py`](../vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py) |
| Reload and lifecycle metrics | [`../vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py`](../vllm/vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py) |
| Experiment runner | [`../benchmark/scripts/benchmark_fanout_lifecycle_timeline.py`](../benchmark/scripts/benchmark_fanout_lifecycle_timeline.py) |
| Actual memory-event timelines | [`../benchmark/scripts/plot_kv_memory_event_timeline.py`](../benchmark/scripts/plot_kv_memory_event_timeline.py) |

## Interpretation and Limits

The experiment establishes the intended scheduling mechanism:

1. branch lifecycle value survives the request-release boundary;
2. cold pressure absorbs GPU evictions while the shared root remains resident;
3. the later multi-branch revisit avoids rereading that root;
4. reduced reload traffic becomes lower TTFT and higher short-turn throughput.

It does not establish a universal throughput gain. The current evidence covers
one model, one GPU, one branch topology, one GPU/CPU capacity ratio, and a
deterministic mechanism workload. It does not measure Disk restoration.

The next experiment should replay AppWorld trajectories and report the
distribution of:

- shared-root HOT and COOLING lifetimes;
- GPU-resident reuse distance;
- reload blocks and bytes avoided per Agent turn;
- protection-window expiry and forced HOT eviction;
- TTFT and completed branch turns under different concurrency levels;
- CPU hit, Disk hit, and recomputation outcomes when a third tier is enabled.
