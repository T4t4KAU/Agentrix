# Coding-Agent DP=8 Full-System Experiment

## Scope

This document records the completed eight-GPU coding-agent experiment run on
2026-07-18. It is a full-system comparison, not an isolated attention-kernel
benchmark. The comparison is:

- **Baseline:** vLLM, FlashAttention, ordinary internal data parallelism, and
  no application prompt compaction.
- **Agentrix:** ForkAttention, prefix-aware internal data parallelism, and
  application prompt compaction.

Tool-KV trimming and predicted TTL were disabled in both variants. Therefore,
the measured difference combines attention backend, DP placement, and exact
application-level removal of known duplicate tool sections. It must not be
reported as the standalone gain of the ForkAttention operator.

All 18 measured arms completed. The raw result root on the experiment server
is:

```text
/test__02/hwx/Agentrix/benchmark/results/coding_agentrix_dp8_commit24_20260718
```

Each arm contains `run.json`, `memory_summary.json`, raw 0.5-second resource
samples, pre/post Prometheus snapshots, and the vLLM log. Each repository also
contains a final `comparison.json`.

## Environment

| Item | Value |
|---|---|
| Date | 2026-07-18 |
| GPUs | 8 x NVIDIA H20-3e |
| Memory | 143,771 MiB per GPU; 1,150,168 MiB aggregate |
| Driver | 550.144.03 |
| CUDA toolkit | 12.9 (`Build cuda_12.9.r12.9/compiler.36037853_0`) |
| Model | Qwen3-32B at `/test__02/hwx/Qwen3-32B` |
| Precision | float16 |
| Parallelism | internal DP=8, TP=1, one API server |
| Model length | 40,960 tokens |
| GPU KV capacity | 3,852 blocks / 61,632 tokens per rank; 493,056 tokens aggregate |
| Scheduler limits | 16,384 max batched tokens; 64 max sequences |
| GPU memory utilization setting | 0.70 |
| Prefix cache | Enabled in both variants |
| Async scheduling | Disabled in both variants |
| KV offload | Not configured |
| Chat mode | Qwen thinking disabled |

The server used a synchronized working-tree snapshot rather than a clean
release commit. The local base revisions at documentation time were Agentrix
`ce2c877b554dcf6226044fb1c3f9d8563a1eb67e` and vLLM
`268a6ab09b3d7a86d656b2dd4d307fa22f2bd015`, with the experiment changes
applied as uncommitted increments. The raw result directory and the checked-in
manifests are the authoritative provenance for this run.

## Workload Construction

The experiment used Django, SQLite, and FFmpeg separately. Each repository has
24 deterministic read-only commit-analysis cases generated from 24 distinct
ordinary commits. Security fixes, vulnerability work, malformed-input cases,
crash reproduction, memory-safety changes, credentials, authentication, and
patch-generation tasks were excluded.

The source specifications and generated manifests are:

```text
benchmark/configs/{django,sqlite,ffmpeg}_agentrix_commit24_specs.json
benchmark/data/{django,sqlite,ffmpeg}_agentrix/cases_30k_b16_commit24.jsonl
```

Each repository was divided into three fixed batches, using offsets 0, 8, and
16. A batch contains eight cases so that the natural number of long-prefix
cohorts matches DP=8. Repositories were never mixed within one run, and cases
were not selected or reordered based on observed performance.

Each case contains:

- one frozen repository parent of approximately 30K harness-tokenizer tokens;
- 16 specialized coding subagents with distinct private instructions;
- deterministic repository-search and reviewer observations;
- heterogeneous trajectories of one, two, or three model rounds; and
- no request to modify the repository or emit a patch.

One one-token bootstrap request per case establishes its natural DP owner.
After all eight bootstraps complete, the branch requests execute in three
barriered waves of 128, 96, and 32 requests. Each batch therefore contains 128
branches and 256 measured branch requests. Across all nine batches, each
variant processed 72 bootstraps and 2,304 measured branch requests.

The systems A/B mode used deterministic replay: prior model responses were
replaced with fixed trace text before later rounds. Both variants used
temperature zero, a 64-token output limit per branch request, the same cases,
the same barriers, and the same output work. Every measured arm generated
exactly 16,384 branch tokens; each variant generated 147,456 branch tokens in
total.

## Variant Settings

| Setting | Baseline | Agentrix |
|---|---|---|
| Attention backend | `FLASH_ATTN` | `FORK_ATTN` |
| DP policy | Ordinary vLLM internal DP | Prefix-aware internal DP |
| Fanout scheduling | Disabled | Enabled |
| Prefix routing | Disabled | Enabled |
| Prefix Forest CUDA Graphs | Not applicable | Enabled |
| Application prompt compaction | Disabled | Enabled |
| Tool-KV trimming | Disabled | Disabled |
| Predicted TTL | Disabled | Disabled |
| Arrival-wave setting | 10 ms | 10 ms |
| Fanout admission window | 0 | 0 |

Every arm started from a fresh vLLM service. No model, KV cache, request queue,
or CUDA Graph state was carried from the preceding arm. Branch wall time starts
with the first branch request and ends with the final branch response; it
excludes service startup, model loading, CUDA Graph capture, and bootstrap.

## Per-Batch Results

`Speedup` is baseline branch wall time divided by Agentrix branch wall time.
TTFT is the request-level median. KV values are the peak aggregate live-token
estimate derived from per-engine vLLM KV occupancy and the fixed physical
capacity.

| Repository / batch | Wall time, baseline / Agentrix (s) | Speedup | Output, baseline / Agentrix (tok/s) | TTFT p50, baseline / Agentrix (s) | Peak live KV, baseline / Agentrix | KV reduction |
|---|---:|---:|---:|---:|---:|---:|
| Django / 0 | 515.12 / 24.90 | 20.69x | 31.81 / 657.93 | 89.54 / 3.07 | 447,652 / 261,636 | 41.55% |
| Django / 8 | 534.93 / 25.77 | 20.75x | 30.63 / 635.67 | 92.20 / 3.16 | 459,703 / 261,940 | 43.02% |
| Django / 16 | 512.10 / 25.41 | 20.15x | 31.99 / 644.73 | 89.34 / 3.20 | 441,955 / 260,964 | 40.95% |
| SQLite / 0 | 498.17 / 24.41 | 20.41x | 32.89 / 671.14 | 91.13 / 3.06 | 357,373 / 260,996 | 26.97% |
| SQLite / 8 | 531.40 / 24.43 | 21.75x | 30.83 / 670.73 | 92.45 / 2.94 | 478,332 / 259,763 | 45.69% |
| SQLite / 16 | 513.29 / 25.73 | 19.95x | 31.92 / 636.89 | 92.09 / 2.91 | 419,805 / 261,236 | 37.77% |
| FFmpeg / 0 | 536.48 / 24.14 | 22.23x | 30.54 / 678.76 | 93.56 / 2.91 | 418,989 / 261,540 | 37.58% |
| FFmpeg / 8 | 557.73 / 24.24 | 23.01x | 29.38 / 676.00 | 91.66 / 2.98 | 417,676 / 261,140 | 37.48% |
| FFmpeg / 16 | 517.15 / 23.72 | 21.80x | 31.68 / 690.73 | 91.27 / 2.98 | 447,764 / 261,988 | 41.49% |

All nine pairs completed the same 256 branch requests and 16,384 output tokens.
No measured request returned HTTP 5xx and no measured runner failed.

## Aggregate Results

| Scope | Baseline wall time (s) | Agentrix wall time (s) | Mean pair speedup | Mean KV reduction | Mean TTFT reduction |
|---|---:|---:|---:|---:|---:|
| Django | 1,562.16 | 76.09 | 20.53x | 41.84% | 96.52% |
| SQLite | 1,542.86 | 74.56 | 20.70x | 36.81% | 96.77% |
| FFmpeg | 1,611.36 | 72.09 | 22.35x | 38.85% | 96.79% |
| All nine pairs | 4,716.38 | 222.75 | 21.19x | 39.17% | 96.69% |

The geometric mean speedup is 21.17x. Summing the nine measured branch phases
gives 31.26 generated tokens/s for the baseline and 661.98 generated tokens/s
for Agentrix. Median TTFT averages 91.47 seconds for the baseline and 3.02
seconds for Agentrix.

The improvement is not a faster decode loop. Baseline median TPOT ranges from
36.32 to 42.15 ms, while Agentrix median TPOT ranges from 65.94 to 67.60 ms.
The dominant benefit is the collapse of long-prefix prefill and queueing when
case cohorts remain with their prefix owner and duplicate tool sections are
not rematerialized. This is consistent with the approximately 97% reduction in
median TTFT and the reduction in peak live KV pressure.

## Prompt Compaction

The compactor removes only tool sections whose stable segment identity is
already present in the frozen parent. It does not summarize or heuristically
rewrite new evidence. In this workload every declared repeated tool section
was an exact known duplicate.

| Repository | Removed sections | Removed characters | Baseline input tokens | Agentrix input tokens | Input-token reduction |
|---|---:|---:|---:|---:|---:|
| Django | 288 | 1,057,489 | 23,995,714 | 23,688,796 | 1.28% |
| SQLite | 288 | 1,003,179 | 24,026,232 | 23,670,296 | 1.48% |
| FFmpeg | 288 | 861,577 | 24,031,496 | 23,716,384 | 1.31% |
| Total | 864 | 2,922,245 | 72,053,442 | 71,075,476 | 1.36% |

The modest logical input-token reduction does not explain a 21x result by
itself. Most logical prompt tokens are the shared 30K repository parent, which
remains part of every request's API-level token accounting. The full-system
gain depends on physically reusing that parent KV on the owning rank instead
of repeatedly prefiling or evicting mixed roots under ordinary DP.

## Memory Results

Average peak live KV fell from 432,139 to 261,245 tokens, a 39.17% reduction.
Average aggregate KV occupancy fell from approximately 87.65% to 52.98%.
Individual baseline batches ranged from 72.48% to 97.01%; Agentrix stayed in a
narrow 52.68% to 53.14% range.

NVML allocated HBM tells a different and complementary story:

| Metric | Baseline | Agentrix |
|---|---:|---:|
| Peak aggregate allocated HBM | 670,032 MiB | 685,136-685,140 MiB |
| Peak allocated HBM per GPU | 83,754 MiB | approximately 85,642 MiB |
| Peak delta above warm allocation | 34,816 MiB aggregate | 34,816-34,820 MiB aggregate |
| Peak server process-tree RSS | approximately 23.3-24.4 GiB | approximately 30.0-31.0 GiB |
| Peak application process-tree RSS | 125-144 MiB | 122-154 MiB |

Agentrix therefore used about 15.1 GiB more aggregate allocated HBM, or about
1.84 GiB more per GPU, despite retaining substantially fewer live KV tokens at
peak. The extra fixed allocation belongs to the different backend/runtime
footprint; the request-window HBM delta was essentially identical. Claims
about lower KV pressure must use the measured live-KV occupancy, not the NVML
allocation alone. This experiment demonstrates higher throughput and lower KV
pressure, but not lower total allocated GPU memory.

## Failure and Recovery Audit

After SQLite batch 16 baseline completed, the first attempt to start its
Agentrix counterpart failed before measurement. One worker could not bind the
random PyTorch distributed rendezvous port 34939 and raised `EADDRINUSE`; the
parent then exited and the remaining workers reported secondary broken pipes.
There was no OOM and no request was submitted in that attempt.

The failed startup log was preserved as:

```text
sqlite/fork_prefix_aware_compact_dp/batch_16/
  vllm_server.startup_failed_eaddrinuse.log
```

Only the missing SQLite optimized arm was restarted. The 11 completed arms
were not repeated or overwritten. The retry completed normally, generated the
SQLite comparison, and the queue then ran all six FFmpeg arms. The failed
startup is excluded because it contains no measured workload.

vLLM also emitted optional `_qutlass_C` import warnings and Python
`resource_tracker` semaphore/shared-memory cleanup warnings during several
normal service shutdowns. All associated `run.json`, resource samples, and
memory summaries were written successfully.

## Interpretation and Limits

The result is strong and consistent across three repositories and nine fixed
batches: pair speedups range from 19.95x to 23.01x. It demonstrates the value
of combining application compaction with prefix-aware placement and
ForkAttention for a deliberately high-sharing, high-KV-pressure coding-agent
shape.

The result should be scoped carefully:

1. It is a deterministic systems replay, not a coding-quality or resolved-task
   evaluation. Model outputs were generated, but no model-produced patch was
   applied or graded.
2. It compares two complete configurations. A separate ablation is required
   to attribute portions of the gain to ForkAttention, prefix-aware routing,
   and prompt compaction individually.
3. Branch-phase wall time excludes model startup and bootstrap. This matches
   the intended steady agent fanout measurement but is not cold-start latency.
4. The workload intentionally stresses eight simultaneous 30K shared-prefix
   cohorts. Results do not automatically generalize to short prompts, low
   concurrency, or workloads without reusable parents.
5. Agentrix reduced live KV pressure but increased the backend's fixed HBM and
   host-RSS footprint. Both facts must accompany any memory-efficiency claim.

Within those limits, the experiment supports the primary claim: for a
multi-round coding-agent workload with large repository parents, keeping each
cohort on its prefix owner and removing exact repeated tool context converts a
queue-dominated 30-33 tok/s system into a stable 636-691 tok/s system on the
same eight GPUs and output work.
