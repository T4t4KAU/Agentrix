# KV residency, placement, and proactive backup

## Scope

The residency index remains observational. Placement supports both a shadow
planner and a separately gated active policy. The active policy changes only
the order in which already-free cached blocks are recycled; it does not alter
prefix lookup or add policy fields to vLLM cache blocks. An independent,
opt-in proactive backup path can create a lower-tier LMCache copy before GPU
pressure forces an eviction.

The existing public KV event stream describes logical hashes, but does not
include physical block IDs, reference-count transitions, cache hits, or block
reuse generations. Those fields are needed to:

- identify concurrently or repeatedly shared prefixes;
- reject stale asynchronous offload acknowledgements after a block ID is
  reused;
- prove that a lower-tier copy exists before a GPU copy becomes demotable;
- classify idle blocks without scanning the complete cache.

## Intrusion boundary

`BlockPool` depends only on the `KVCacheObserver` protocol. The residency state
machine is an optional implementation attached by `KVCacheManager`; no
Agentrix-specific state is added to `KVCacheBlock`, Scheduler output, the
external KV event schema, or public metrics.

The observer is disabled by default. When disabled, the existing block-pool
path only executes an observer-null check at each lifecycle hook.

The implementation keeps O(1) state counts and intrusive ordered lists only
for idle states. Active, unhashed, and free blocks are not linked into victim
queues. Aging is limited by a configurable transition budget and rotates
between the four aging queues, so one scheduler step performs
`O(aging_budget)` rather than draining every simultaneously expired block.
Snapshot construction is O(1); the expensive consistency check is test-only.

## State and safety model

The current lifecycle segments are:

```text
FREE -> UNHASHED -> ACTIVE -> WARM -> COOLING -> COLD
                       |         shared variants       |
                       +---- ACTIVE/WARM/..._SHARED ---+
```

A prefix becomes shared when either its concurrent request fanout reaches two
or its request reuse count reaches the configured threshold. `BlockPool`
exposes separate request `touch/free_blocks` and internal `pin/unpin` paths;
temporary offload references therefore affect physical liveness without
inflating either sharing signal. Shared classification lasts for the physical
block generation and both signals reset when the generation changes.

Every physical block allocation or key invalidation advances a 32-bit
generation. Asynchronous backup operations carry
`(block_id, generation, tier, operation_id)`; an acknowledgement for an old,
unknown, already completed, or wrong-tier operation is rejected. Concurrent
operations on the same block remain independently in flight. CPU and remote
backup bits are kept separately from lifecycle state. The backup APIs are
wired to the optional LMCache coordinator described below.

## Placement planning

The optional placement planner runs immediately before an allocation that
would recycle cached blocks. In shadow mode its result does not alter the
allocation or LRU order.

Victim selection is separated from residency tracking in
`vllm/v1/core/kv_placement.py`. The first policy deliberately uses only state
that has already been validated:

- shared idle prefixes are never returned as victims;
- blocks with a CPU or remote copy can release their GPU copy;
- cold, never-reused blocks can be discarded;
- other unshared victims must first be backed up to CPU;
- blocks with an in-flight backup are deferred.

Cold, cooling, and warm queues are checked in that order, and each queue keeps
oldest-first order. The scan budget is shared by all allocations in one
scheduler step, rather than being renewed per request. The last plan contains
generation-stamped decisions, unresolved demand, protected-prefix pressure,
and budget-exhaustion status. Cumulative planner counters are available from
`KVCacheManager.get_placement_stats()`.

When active placement is enabled, the planner subtracts uncached free blocks
and the current request's not-yet-touched, window-resident cache hits from
allocation demand. Local-attention hits outside the current window remain
valid eviction candidates.
It ranks candidates as follows:

1. blocks with an existing CPU or remote copy;
2. cold blocks that have never been reused;
3. remaining non-shared blocks, discarded as a progress-preserving fallback.

The final category may require recomputation on a future hit, but prevents an
unbacked warm cache from stalling admission. Proactive backup reduces how
often this fallback is needed. Current-request cache hits are explicitly
excluded. Every selected generation is revalidated immediately before
`BlockPool` atomically moves the complete victim set to the front of its free
queue. If the bounded plan is incomplete or stale, allocation returns `None`
to the existing scheduler admission/preemption path; it never falls back to a
shared prefix. Explicit cache reset and correctness invalidation remain
allowed to remove shared entries.

## Proactive LMCache backup

The coordinator is implemented in LMCache rather than in vLLM's allocator.
vLLM exposes only the block-pool observer and thin connector delegation hooks.
At request completion, LMCache converts complete chunks into
generation-stamped physical-block candidates. It then:

- rejects an entire request atomically if retaining it would exceed the
  configured block bound;
- hard-excludes any chunk that has become shared;
- validates every block generation and existing lower-tier copy;
- pins selected GPU blocks and records a tier- and operation-specific backup;
- executes D2H only in a connector-only engine step, on the model runner's
  CUDA-owning thread;
- aggregates acknowledgements from all tensor- and pipeline-parallel model
  workers before marking a CPU copy and
  releasing both the pin and the finished request's blocks.

MLA workers configured with `save_only_first_rank` still participate in the
acknowledgement barrier, but passive ranks report an explicit no-op success;
only the leader performs the store. Local CPU removals are returned in worker
metadata by chunk hash. The scheduler keeps the corresponding physical block
IDs and generations and clears the CPU residency bit through `drop_backup()`.
Thus placement can treat a lower-tier copy as durable only until LMCache
actually evicts it. Eviction feedback is deduplicated per worker step and its
callback runs outside the local CPU cache lock.

Using vLLM's existing delayed-free contract guarantees that a completed
request remains alive until its backup acknowledgement. Persistent failures
release all pins and do not retry in a tight loop. The candidate scan, batch,
and total retained-block counts are independently bounded. Ordinary
request-driven LMCache save is disabled while this policy is active, so each
chunk follows one storage path.

Requests are retained only while occupancy is at or above the watermark. If
pressure falls after registration, candidates are released within the same
per-step scan budget. A candidate already owned by another backup is also
released instead of being requeued indefinitely.

Admission is proportional to pressure rather than all-or-nothing per request.
The scheduler converts the number of occupied blocks above the watermark into
whole LMCache chunks, subtracts chunks already retained, and admits only that
prefix of a finished request. The existing batch-block limit is the per-step
transfer budget, while the retained-block limit remains an atomic request-level
backpressure guard. This keeps all work bounded without adding a second
candidate index.

The default D2H path remains synchronous. Setting
`VLLM_AGENTRIX_KV_PROACTIVE_ASYNC=1` instead enqueues the copy on LMCache's
connector-owned store stream after waiting for current-stream KV writes. A
CUDA event fences publication to the CPU cache. Newly registered chunks wait
one planning step and are then revalidated, allowing concurrent prefix reuse
to promote them to protected shared state before any copy starts. Event checks
do not synchronize CUDA; once a copy completes, the bounded CPU-cache publish
runs before ACKing the scheduler and releasing the pinned GPU generation.
Async mode may attach a bounded backup to a model-bearing step, permitting D2H
to overlap forward execution without moving CUDA work to a Python executor
thread.

## Configuration

```bash
VLLM_AGENTRIX_KV_RESIDENCY_SHADOW=1
VLLM_AGENTRIX_KV_WARM_SECONDS=10
VLLM_AGENTRIX_KV_COOLING_SECONDS=100
VLLM_AGENTRIX_KV_SHARED_REUSE_THRESHOLD=2
VLLM_AGENTRIX_KV_AGING_BUDGET=256
VLLM_AGENTRIX_KV_PLACEMENT_SHADOW=1
VLLM_AGENTRIX_KV_PLACEMENT_ACTIVE=0
VLLM_AGENTRIX_KV_PLACEMENT_SCAN_BUDGET=64
VLLM_AGENTRIX_KV_PROACTIVE_BACKUP=1
VLLM_AGENTRIX_KV_PROACTIVE_ASYNC=0
VLLM_AGENTRIX_KV_BACKUP_HIGH_WATERMARK=0.8
VLLM_AGENTRIX_KV_BACKUP_SCAN_BUDGET=32
VLLM_AGENTRIX_KV_BACKUP_BATCH_BLOCKS=64
VLLM_AGENTRIX_KV_BACKUP_MAX_INFLIGHT_BLOCKS=256
LMCACHE_CONFIG_FILE=benchmark/configs/lmcache_proactive_backup.yaml
```

The warm threshold must not exceed the cooling threshold. Shadow mode is off
unless explicitly enabled. The aging budget is a positive maximum number of
state transitions performed synchronously per scheduler step.
Enabling either placement mode also enables the required residency index. Its
scan budget is a separate positive per-step limit. Active mode may inspect at
least one entry per requested allocation block, so its work is
`O(allocation)` rather than `O(cache size)` even when an allocation exceeds the
configured budget. Proactive backup also enables the residency index and
requires non-layerwise local CPU storage without CacheBlend. Asynchronous D2H
is separately gated so the synchronous path remains the default and can serve
as an A/B baseline.

## Initial profiling

Environment: one RTX 5090, Qwen3-VL-8B-Instruct, BF16, 8 concurrent sessions,
2,048 shared-prefix tokens, 1,024 private tokens, 64 follow-up tokens. Two
server-order runs were performed (`disabled -> enabled` and
`enabled -> disabled`), with five measured trials per mode in each run.

Pooled medians across ten trials per mode:

| Phase | Metric | Disabled | Enabled | Delta |
|---|---:|---:|---:|---:|
| First turn | p50 TTFT | 710.11 ms | 714.95 ms | +0.68% |
| First turn | throughput | 11.163 req/s | 11.084 req/s | -0.71% |
| Follow-up | p50 TTFT | 192.73 ms | 189.27 ms | -1.79% |
| Follow-up | throughput | 40.093 req/s | 40.792 req/s | +1.74% |

The local prefix-hit rate was identical in both modes: 66.67% on first turns
and 97.96% on follow-ups. Follow-up latency was visibly bimodal across trials,
so its small apparent improvement should be treated as measurement noise, not
as a performance gain.

The synthetic BlockPool loop performs allocation, cache insertion, release,
cache hit, and release for every block. Median shadow bookkeeping cost was
2.36 microseconds per processed block (about 0.47 microseconds per state
update). The disabled observer checks added about 32 nanoseconds per processed
block relative to the pre-change BlockPool in the same synthetic loop.

`tracemalloc` measured approximately 128.6 bytes of CPU metadata per physical
block. The profiled model exposed 53,728 KV tokens; at a 16-token block size,
the index therefore uses roughly 0.41 MiB.

Reusable profiling entry points:

- `benchmark/scripts/profile_kv_residency_shadow.py`
- `benchmark/scripts/run_kv_residency_shadow_profile.sh`
- `benchmark/scripts/profile_kv_placement_shadow.py`

Raw server artifacts are under
`/root/autodl-tmp/Agentrix/benchmark/results/kv_residency_shadow*`.

## Review hardening

After separating internal pins, scoping sharing metadata to a generation, and
adding operation-aware backup tracking, a fresh callback-counted synthetic
profile measured 2.82 microseconds of shadow bookkeeping per processed block
(0.48 microseconds across 5.872 callbacks per block). Idle metadata measured
128.0 bytes per physical block.

Temporary pins now save their intrusive-list neighbors and restore the idle
position in O(1) when those anchors remain valid. If aging moved either
neighbor during a long pin, unpin refreshes that block as a warm tail entry;
it never inserts an old timestamp behind newer entries or performs an O(N)
ordered search on the scheduler thread.

With 100,000 cached blocks expiring at the same timestamp, one scheduler-step
call moved exactly the configured 256 transitions in 0.097 ms; 99,744 blocks
remained queued for later steps. This replaces the previous unbounded drain
with a deterministic per-step upper bound.

## Placement planner profiling

The first planner implementation constructed a full residency snapshot for
every candidate and took about 0.77 ms to inspect 256 blocks. A minimal
read-only candidate path reduced that workload to 0.19 ms. The default was
then set to 64 candidates per scheduler step.

Across 5,000 fully utilized steps and seven alternating baseline/planner runs,
the final 64-block configuration added a median 49.31 microseconds per step,
or 770 nanoseconds per inspected block. Direct per-step samples measured
49.85 microseconds p50, 50.85 microseconds p95, 52.00 microseconds p99, and
87.89 microseconds maximum. The raw result is
`/root/autodl-tmp/Agentrix/benchmark/results/kv_placement_shadow_final.json`.

Active mode additionally sorts the bounded candidate window, revalidates each
generation, and atomically reprioritizes 64 free-queue nodes. Across 2,000
steps and seven alternating baseline/planner runs, this complete path measured
89.82 microseconds p50, 93.07 microseconds p95, and 135.89 microseconds p99.
The median incremental cost was 89.25 microseconds per step, or 1.39
microseconds per selected block. A CPU-pinned control with cyclic GC disabled
reduced p99 to 94.27 microseconds, but a later 10,000-step run instrumented
with `gc.callbacks` observed no collections. The earlier difference was
therefore scheduler noise rather than sufficient evidence of a GC tail. The
raw baseline results are
`/root/autodl-tmp/Agentrix/benchmark/results/kv_placement_active.json` and
`/root/autodl-tmp/Agentrix/benchmark/results/kv_placement_active_no_gc.json`.

The active allocation path now reuses mutable candidate and queue-validation
buffers. Immutable decisions are materialized lazily only when diagnostics
read the last plan. The queue keeps its atomic validation contract and uses a
one-byte marker per physical block, allocated only on first use, to reject
duplicate IDs without a temporary set. Under the same 8,192-block, 64-victim,
2,000-step profile,
this reduced p50 from 89.34 to 59.20 microseconds and p99 from 93.47 to 62.20
microseconds. Median incremental cost fell from 1.38 to 0.92 microseconds per
selected block. The shadow path remained at 51.58 microseconds p50. Results are
stored in `benchmark/results/kv_placement_active_workspace.json` and
`benchmark/results/kv_placement_shadow_workspace.json` on the test server.

A real RTX 5090 pressure smoke used Qwen3-VL-8B-Instruct and only 128 KV blocks
(2,048 tokens). A 480-token prefix was reused until classified as shared, then
two independent 1,100-token contexts forced recycling. The final shared-prefix
request still hit all 480 reusable tokens. The cumulative hit rate was 26.4%,
exactly 960 cached tokens out of 3,643 prompt tokens.

## Proactive backup profiling

With the default 32-chunk scan budget, 64-block transfer batch, 16-token vLLM
blocks, and 256-token LMCache chunks, 5,000 synthetic coordinator samples
measured:

| Candidate state | p50 | p95 | p99 |
|---|---:|---:|---:|
| Ready, including pin and operation creation | 21.50 us | 22.11 us | 22.63 us |
| Shared, hard protected | 21.04 us | 21.86 us | 22.26 us |
| Already backed up | 66.07 us | 67.36 us | 68.73 us |
| Registration below pressure watermark | 0.21 us | 0.26 us | 0.28 us |
| Pressure drop after registration | 8.64 us | 9.02 us | 9.27 us |

An already-backed chunk must validate that every constituent vLLM block has a
lower-tier copy; unlike the shared path, it cannot safely stop at the first
block. Its 32-chunk worst-case sample therefore performs 512 constant-time
metadata reads while remaining bounded by the scan budget.

The raw result is
`/root/autodl-tmp/Agentrix/benchmark/results/proactive_backup_scheduler_profile.json`;
the reusable entry point is
`benchmark/scripts/profile_proactive_backup.py`.

A real RTX 5090 smoke run with Qwen3-VL-8B-Instruct backed up 1,280 tokens in
three acknowledged batches and exited without a forced kill or leaked
semaphore. Steady transfers reached 10.55--10.85 GiB/s: 512 tokens took
6.48 ms and 256 tokens took 3.33 ms. A one-block startup warmup moved CUDA
initialization off the request path and reduced the first 512-token batch from
about 34 ms to 16.03 ms; the warmup itself took 31.01 ms during server startup.
The smoke artifact is under
`/root/autodl-tmp/Agentrix/benchmark/results/proactive_backup_smoke_warmup`.

The active placement and proactive LMCache paths were also enabled together
with the same 128-block pressure setup. The shared 480-token prompt completed
in 57.03 ms after two independent 1,100-token prompts forced recycling,
compared with 127.96 ms when first populated. LMCache concurrently completed
one 256-token and two 1,024-token proactive stores at 2.35--6.62 GB/s. The
final request used its local prefix (external hit rate remained zero), so the
backup path did not accidentally pin, reclassify, or displace the protected
GPU copy. These latency samples are functional-smoke observations rather than
an end-to-end throughput comparison.

Deferring one synchronization per chunk was also measured separately over 60
alternating 32 MiB trials. It regressed the median from 8.42 ms to 9.19 ms, so
per-chunk deferral remains rejected. The optional event-driven implementation
instead removes the batch-level wait and delays cache publication until one
batch completion event.

Two event-driven runs of the 8-case, 32-branch workload measured 3,550.56 ms
median end-to-end and 1,838.29 ms median branch-wall time. Two synchronous
runs measured 3,571.91 ms and 1,885.73 ms respectively. The differences are
small enough to treat as noise, not as a demonstrated speedup. One-step
sharing revalidation reduced asynchronous backup volume from 7,680 to 5,120
tokens, but the synchronous path stored only 2,560 tokens. Async mode therefore
remains opt-in. Raw artifacts are under
`benchmark/results/proactive_async_grace_async_*` on the test server.

A separate 96-GPU-block pressure test ran prompts in A/B/C/D/A order. The last
A request reloaded 768 tokens from LMCache after GPU recycling and reproduced
the first request's deterministic output exactly. Its server log is under
`benchmark/results/proactive_async_forced_reload`.

Pressure-proportional admission was checked with a 96-GPU-block P/A/B/C/D/A
sequence at the production 0.8 watermark. Requests above the watermark stored
512 rather than all 768 eligible tokens. The final A request reloaded those 512
tokens from LMCache and reproduced its first deterministic output. With the
ordinary 8-case workload, KV occupancy stayed below 0.8 and correctly issued
no backup. A forced-watermark-zero control retained the previous 5,120-token
backup volume and measured 3,545.51 ms end-to-end. Artifacts are under
`benchmark/results/proactive_pressure_admission_*` on the test server.
