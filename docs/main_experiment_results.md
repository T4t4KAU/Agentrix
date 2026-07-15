# Main Experiment: Shared-Prefix Performance and Accuracy

## Scope

This document is the canonical record for the completed high-value experiment,
the corrected single-GPU ForkAttention validation performed on 2026-07-14,
and the capacity-aware DP router follow-up completed on 2026-07-15. Current
launch procedures are documented separately in
[`main_experiment_matrix.md`](main_experiment_matrix.md).

The high-value run uses all four bundled datasets: SWE-bench Verified,
AgencyBench, AgentBoard, and AppWorld. It retains the complete Qwen3-1.7B
8K/16K by 8/16-branch grid on AgentBoard and AppWorld, adds cross-dataset
16K/16 stress replications, and validates Llama-3.2-1B on the same stress
shape. The two-rank Qwen3-8B DP result uses paired Warm8K and
pressure-oriented 16K validation of the current capacity-aware routing policy.
Qwen3-14B TP accuracy uses the maximum 16K/32 shape.

The Qwen single-GPU core grid consumes all 9 AgentBoard and 13 AppWorld
records. Its AgencyBench 8K/8 cell uses 32 records; all 16K/16 stress
replications use eight records. The current DP validation uses four
AgencyBench cases per run. TP accuracy uses the first eight records per
dataset. Every comparison within a cell uses identical prompts, and reported
cross-cell means are unweighted.

The experimental KV reload rebalance feature is disabled throughout. The DP
comparison isolates the stable prefix-aware routing path.

## Experiment Groups

### Single GPU

Models:

- Qwen3-1.7B
- Llama-3.2-1B

Variants:

- FlashAttention without offload
- ForkAttention without offload
- FlashAttention with ordinary LRU CPU offload
- ForkAttention with ordinary LRU CPU offload
- ForkAttention with prefix-aware optimized CPU offload

Both offload variants use vLLM's native `OffloadingConnector`, identical CPU
capacity, and LRU eviction. The ordinary variant disables fanout admission,
preemption, GPU hotset protection, and connector planning. The optimized
variant enables that complete prefix-aware policy. LMCache and disk storage
are excluded from this matrix.

The default host matrix uses 8K and 16K prefixes with 8 and 16 branches. Other
settings can be selected through `PREFIX_LENGTHS` and `BRANCH_COUNTS`.

### Data Parallel

Model: Qwen3-8B on two internal DP ranks. The current validation uses four
cases, 16 branches per case, and either an exact warm 8K prefix or a shuffled,
non-warmed 16K pressure workload.

Variants:

- ForkAttention with ordinary DP and no offload
- ForkAttention with the final capacity-aware DP router and no offload

Both variants use the same ForkAttention backend, CUDA Graph settings, and
physical KV capacity. The client does not force a data-parallel rank.

The current capacity-aware router bypasses prefix hashing and arrival waves
for prompts smaller than 20% of one rank's KV capacity. For eligible long
prefixes it selects the deepest prefix match first, keeps a cohort together
under a capacity-derived skew budget, and uses a short arrival wave to expose
the cohort to the router.

### Tensor Parallel Accuracy Guardrail

Model: Qwen3-14B with TP=2 and no offload.

Variants:

- FlashAttention reference
- ForkAttention run 1
- ForkAttention run 2

The two ForkAttention runs measure both backend agreement and repeatability.
Accuracy in this document means normalized exact match, token F1, and text
similarity against greedy FlashAttention output. It is not task success in an
interactive SWE-bench, AgentBoard, or AppWorld environment.

Fork run 2 is also compared directly with Fork run 1. Every run manifest
records the Agentrix and vLLM Git commits, whether either worktree was dirty,
and the sampler backend. The matrix fixes `VLLM_USE_FLASHINFER_SAMPLER=0` so
runtime sampler JIT is not a hidden variable in the attention comparison.

## Recorded Metrics

- Logical KV read volume and reduction relative to branch-local attention
- Sampled GPU KV cache usage and configured GPU KV capacity
- Sampled CPU offload-cache occupancy and peak total physical KV occupancy
- KV offload load/store traffic
- Request, input-token, output-token, and total-token throughput
- TTFT and TPOT mean, P50, P95, and P99
- End-to-end request latency mean, P50, P95, and P99
- GPU compute utilization from NVIDIA `utilization.gpu`
- Memory bandwidth activity proxy from NVIDIA `utilization.memory`
- FlashAttention-to-ForkAttention output agreement for the TP guardrail
- Physical ForkAttention observed/active steps and shared/singleton CTA counts

Logical KV read reduction estimates repeated shared-prefix reads avoided by
ForkAttention; it does not claim that vLLM stores duplicate physical prefix
blocks. Physical GPU KV reduction is reported separately from sampled vLLM
occupancy and is compared with the named baseline for each variant.

Logical KV reduction is not evidence that the ForkAttention kernel executed.
The corrected runner therefore exports cumulative physical execution counters:
an active step contains at least one CTA serving multiple queries, while the
shared CTA ratio compares shared-prefix CTAs with singleton suffix CTAs.

`utilization.memory` measures memory-controller activity. It is not a direct
measurement of HBM bandwidth in GB/s.

## Reproduction

Run from the `benchmark` directory and provide machine-specific model paths
through `MODEL_SPECS` rather than editing tracked scripts:

```bash
MODE=single_gpu MODEL_SPECS='qwen3-1.7b|/path/to/Qwen3-1.7B;llama3.2-1b|/path/to/Llama-3.2-1B' \
  ./scripts/run_main_experiment.sh

MODE=tp_accuracy MODEL_SPECS='qwen3-14b|/path/to/Qwen3-14B' \
  ./scripts/run_main_experiment.sh
```

The current Warm8K and Pressure16K DP commands, including the two-variant
restriction and routing controls, are in
[`dp_experiment_results.md`](dp_experiment_results.md). The generic `MODE=dp`
defaults must not be used as evidence for the current router.

Single-GPU runs now default to `EXPERIMENT_PROFILE=fanout_validated`. This
profile writes to `results/main_experiment_v2`, processes one case per batch,
uses case-major branch admission, warms the exact shared branch context, emits
256 decode tokens, and enables the 16-request fanout admission window for the
no-offload Fork variant. It also enables forest CUDA Graph capture for batches
that contain multiple shared roots. Prefix warm-up is recorded separately and
excluded from measured case latency. These controls measure the intended
ForkAttention regime instead of cache-thrashing mixed-prefix admission.

The previously published 4-case, round-robin, 64-token matrix remains exactly
reproducible with:

```bash
EXPERIMENT_PROFILE=legacy MODE=single_gpu \
  MODEL_SPECS='qwen3-1.7b|/path/to/Qwen3-1.7B' \
  ./scripts/run_main_experiment.sh
```

The runner resumes by skipping completed result files. Generated Markdown and
CSV reports are stored under the selected `benchmark/results/` root.
The default `MAX_DATASET_RECORDS=32` and whether a run is uncapped are included
in its provenance manifest.

For policy comparisons, pin the same physical GPU KV capacity across variants
with `NUM_GPU_BLOCKS_OVERRIDE`. The host run recorded here uses 1700 blocks
(27,200 tokens at the default 16-token block size) after calibration against
the lowest observed backend capacity.

## Executed Coverage

| Group | Hardware | CUDA | Executed coverage |
|---|---|---|---|
| Single GPU | RTX 5070 12 GiB | 13.0 build | 75 runs: Qwen complete grid on AgentBoard/AppWorld; Qwen and Llama 16K/16 stress replications on four datasets |
| Data parallel | 2x RTX 5090 32 GiB | 12.8 | 12 focused Qwen3-8B runs: paired Warm8K and repeated Pressure16K comparisons |
| TP accuracy | 2x RTX 5090 32 GiB | 12.8 | 12 standard Qwen3-14B 16K/32 runs plus two Flash repeatability controls |

Single-GPU comparisons pin 1,700 GPU blocks and an 8 GiB CPU offload cache.
DP comparisons pin 3,852 blocks per rank. The reload-rebalance experiment is
disabled. Earlier diagnostic checkpoints with inherited hotset policy or
backend-dependent capacities are excluded.

The single-GPU and TP result manifests retain their recorded Git provenance.
The current capacity-aware DP result validates the corrected router at vLLM
commit `0cff6d857`.

## Corrected Single-GPU ForkAttention Validation

The original no-offload matrix is a mixed-prefix scheduler/cache stress test,
not a clean operator comparison. It admits four unrelated prefixes in
round-robin order while pinning the GPU cache to 27,200 tokens. Four 8K roots
need roughly 32K tokens and four 16K roots roughly 64K tokens. At 16K the
server reported only 1.24 maximum concurrent requests, usually ran one request
with up to 59 waiting, and recorded near-zero prefix-cache hits. The logical
shared tree still reported more than 90% KV reduction even though sibling
requests rarely reached decode together.

The original raw batches corroborate this diagnosis. The dominant 4-case
Qwen3-1.7B batches were flat or slower, while the final partial batch containing
one case accelerated by 2.35x on AgentBoard 8K/16, 2.03x on AppWorld 8K/16,
and 2.83x on AgentBoard 16K/16.

The corrected validation used the following fixed configuration:

| Setting | Value |
|---|---|
| GPU / build | RTX 5070 12 GiB, CUDA 13.0, native `sm_120` build |
| Model / data | Qwen3-1.7B, AgentBoard record 0 |
| Prefix / branches | 8K / 16 |
| Case admission | one case, case-major, concurrency 16 |
| Shared-prefix setup | exact shared-context warm-up, excluded from measured latency |
| Suffix / decode | lognormal mean 256 / 256 output tokens |
| GPU KV | 1,700 blocks = 27,200 tokens |
| Fork scheduling | enabled, admission window 16 |
| CUDA graph | common-prefix graph; forest mode is not applicable to this one-case run |
| Offload | disabled |

| Backend | Output tok/s | Request latency P50 | TPOT P50 | Physical activation |
|---|---:|---:|---:|---:|
| FlashAttention | 448.19 | 7,823.08 ms | 29.59 ms | - |
| ForkAttention | 923.95 | 3,056.12 ms | 11.30 ms | 255/329 steps (77.51%) |

ForkAttention improves output throughput by 106.15%, cuts median request
latency by 60.9%, and cuts median TPOT by 61.8%. The counters recorded 1,278
shared and 4,069 singleton CTA-plan entries, accumulated once per model step
and not multiplied by layer count. The 23.90% shared-CTA ratio is not an
activation rate because a long shared prefix is represented by a small number
of shared CTAs while every private suffix contributes singleton CTAs.

A second regression kept four cases, disabled explicit prefix warm-up, restored
the original 64-token decode length, and changed only admission semantics to
case-major plus the Fork 16-request scheduling window. On Qwen3-1.7B
AgentBoard 8K/16, ForkAttention reached 363.76 output tok/s versus 293.69 for
FlashAttention, a 23.86% improvement. It physically activated on 241/412
observed steps (58.50%), with 4,109 shared and 8,364 singleton CTA-plan
entries. The server reached 18 concurrent running requests and a 20.6% prefix
cache hit rate; the earlier round-robin run generally had only 2-5 running
requests and 8-13% prefix hits. This four-record regression is not a replacement
for the complete historical dataset aggregate, but it confirms that the
remediation still works in a multi-case workload without artificial warm-up.

The first forest CUDA Graph attempt on that same four-record workload exposed
a real workspace bug: the captured plan reserved 10 splits per sequence while
the runtime forest required 11, and the engine stopped with a
`workspace mismatch` instead of silently falling back. The old default derived
the split capacity mostly from sequence length, but branch points introduce
additional splits independently of sequence length. Reserving the gather
kernel's supported maximum of 32 splits fixes the mismatch. With forest graphs
enabled, the same ForkAttention workload reached 437.15 output tok/s, 20.18%
above eager/dynamic-forest ForkAttention and 48.85% above FlashAttention. It
activated on 242/339 steps (71.39%), with 3,795 shared and 8,861 singleton CTA
plan entries. The measured total latency fell from 11,963.92 ms to 9,955.43 ms.

An independent unprofiled Qwen3-0.6B run with the same 8K/16 fanout shape
measured a 3.57x branch-phase and 2.30x end-to-end speedup. Matched Nsight
Systems captures attributed 1,661.09 ms to all FlashAttention attention
kernels versus 404.85 ms to all attention kernels in the Fork run, a 4.10x
reduction. The dominant Flash branch kernel used 1,321.14 ms, while the
complete specialized Fork split/gather/merge path used 173.65 ms, a 7.61x
reduction. Attention explains 97.65% of the total GPU-kernel time saved.

Nsight Compute on a two-request tail kernel showed 61.86% DRAM throughput,
11.57% compute throughput, 36.86 KiB dynamic shared memory per block, and only
56 one-warp CTAs on 48 SMs. Shared memory limits this shape to two blocks per
SM; 210 registers per thread are not its immediate residency limit. The 2.32%
achieved occupancy and strong SM imbalance identify adaptive CTA splitting and
`sm_120` tile tuning as the highest-value kernel work for shrinking cohorts.
The sample does not represent the full 16-request cohort.

The full breakdown and optimization ranking are recorded in
[`forkattention_operator_profile.md`](forkattention_operator_profile.md).
They prioritize tail-only adaptive prefix splitting, reducing eligible
FlashAttention fallbacks, and architecture-specific tile autotuning. Raw
metadata-copy bandwidth, gather-only fusion, and register reduction alone are
lower-return targets in the current captures. The evidence does not support
the conclusion that the ForkAttention algorithm is ineffective on one GPU.

The accompanying remediation makes the validated profile the single-GPU
default, preserves the old workload behind `EXPERIMENT_PROFILE=legacy`, adds
explicit shared-prefix warm-up provenance, exports physical activation
counters to Prometheus and the generated report, enables forest CUDA Graphs,
reserves enough forest split workspace, and fixes the inconsistent backend
capability check so Ampere/Ada devices accepted by `supports_combination()` are
also accepted by `supports_compute_capability()`.
The corrected validation was intentionally run from the modified development
checkout, so its manifest records both worktrees as dirty.

## Single-GPU Offload

### Corrected Offload Validation

The historical offload matrix combined a round-robin mixed-prefix workload
with a policy change that simultaneously enabled fanout admission, preemption,
GPU hotset protection, and connector planning. A corrected experiment added an
intermediate `fork_scheduled_ordinary_offload` variant so scheduler and
connector effects can be measured separately.

The local validation used an RTX 5070 12 GiB, Qwen3-1.7B, the first four
AgentBoard records, four case-major roots, 16 branches per root, no explicit
prefix warm-up, lognormal 256-token mean suffixes, 256 output tokens, 1,700 GPU
blocks (27,200 tokens), an 8 GiB CPU cache, and forest CUDA Graphs. Every 8K
variant was repeated three times. The 16K scheduler/connector pair was also
repeated three times. All cells generated nonzero CPU-to-GPU reload traffic.

The 8K/16 medians are:

| Variant | Output tok/s | Branch TTFT P50 | Branch TPOT P50 | KV load | Preemptions | Fork active steps |
|---|---:|---:|---:|---:|---:|---:|
| Flash ordinary offload | 474.12 | 10,406.08 ms | 47.57 ms | 4.51 GiB | 113 | - |
| Fork ordinary offload | 791.64 | 4,968.23 ms | 16.91 ms | 6.19 GiB | 41 | 92.28% |
| Fork scheduled ordinary offload | 792.05 | 5,410.28 ms | 20.74 ms | 4.21 GiB | 113 | 90.68% |
| Fork optimized offload | 673.46 | 7,384.82 ms | 25.96 ms | 3.86 GiB | 25 | 90.74% |

Paired medians show that changing only FlashAttention to ForkAttention with the
ordinary connector improves throughput by 66.97% and reduces TPOT by 64.81%.
Adding fanout scheduling to the already case-major workload changes throughput
by only +0.95%. Changing only the connector policy from scheduled ordinary to
optimized reduces throughput by 12.86%, increases TTFT by 37.10%, and reduces
reload traffic by only 2.57%. The complete optimized system remains 44.95%
faster than Flash ordinary offload, but that is a backend/system gain rather
than evidence that the optimized connector is faster.

The 16K/16 pair confirms the connector regression:

| Variant | Output tok/s | Branch TTFT P50 | Branch TPOT P50 | KV load | Fork active steps |
|---|---:|---:|---:|---:|---:|
| Fork scheduled ordinary offload | 511.83 | 10,751.97 ms | 12.41 ms | 11.06 GiB | 85.20% |
| Fork optimized offload | 386.25 | 12,842.16 ms | 27.10 ms | 9.30 GiB | 80.89% |

Across paired runs the optimized connector lowers reload traffic by 17.61% but
lowers throughput by 24.54%, increases TTFT by 17.60%, and more than doubles
TPOT (+118.34%). A `PROFILE_FORK=1` diagnostic run found similar graph-miss
rates (about 2-3%) but a different graph mix: forest-graph hits were 32.4% of
successful graph dispatches with scheduled ordinary offload and 12.6% with
optimized offload. Measured GPU load-copy time was only 0.26-0.33 seconds, so
copy bandwidth does not explain the end-to-end regression. The evidence is
consistent with the optimized restore/residency policy changing cohort shape
and reducing multi-root forest execution; scheduler-level tracing is still
needed to establish the precise causal path.

This pre-remediation result does not reproduce the old claim that the optimized
connector itself provides a 70-150% gain. It established the regression that
the remediation below targets. Raw local results are in
`benchmark/results/investigation_20260714/offload_validated_*`.

### Offload Remediation and Revalidation

Static analysis found three concrete native-offload defects rather than a flaw
in the basic residency objective. First, CPU admission was tracked only by
request and logical block index. Sibling requests with the same `OffloadKey`
therefore planned and prepared the same shared-prefix backup repeatedly.
Second, a CPU LRU eviction did not clear those per-request admission markers,
so an evicted key could remain permanently ineligible for backup and later be
recomputed. Third, hot-prefix backup was delayed until critical pressure even
though `OffloadingConnector` creates a CPU copy without releasing the GPU
block. The GPU hotset reservation is short-lived, so a useful prefix could lose
GPU residency before a recoverable CPU copy existed.

The repaired connector now maintains one process-wide admitted-key set plus a
reverse key-to-request index. All branches reuse one CPU copy, CPU evictions
invalidate every affected request marker, and request completion removes only
the reverse references while retaining knowledge of a still-resident CPU key.
When explicitly enabled, a hot shared prefix is backed up at normal pressure;
the GPU copy remains resident and the backup becomes useful only if later GPU
allocation pressure evicts it.

The same 16K/16 four-root experiment was repeated three times after the fix.
These production measurements used `FANOUT_PROFILE=0`; per-step fanout logging
was used only in separate diagnostic runs because it writes one detailed line
per scheduler step and materially perturbs the optimized variant.

| Variant | Output tok/s | Branch TTFT P50 | Branch TPOT P50 | KV load | Load ops | KV store | Store ops | Fork active steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fork scheduled ordinary offload | 518.86 | 10,702.94 ms | 12.10 ms | 11.88 GiB | 11 | 8.26 GiB | 103 | 73.34% |
| Fork optimized offload | 626.13 | 7,552.10 ms | 12.08 ms | 8.89 GiB | 7 | 8.15 GiB | 135 | 80.68% |

Across paired runs, optimized offload improves throughput by 22.51%, reduces
TTFT by 27.84%, reduces CPU-to-GPU KV load by 25.84%, and leaves TPOT
effectively unchanged (-0.10%). All three throughput pairs are positive
(+7.00%, +22.51%, and +24.88%). The ordinary control remains close to its
pre-fix median, while optimized throughput rises from 386.25 to 626.13 tok/s;
this rules out a generally faster machine run as the explanation. The higher
optimized store-operation count is a remaining small-transfer/job-overhead
target even though total store bytes fall by 1.37%.

The LMCache three-tier path had a separate set of correctness and admission
problems:

- Async disk prefetch held `disk_lock` while CPU allocation could evict and
  demote another object back to disk, creating a lock inversion. Allocation
  failure also leaked the lock/pin.
- `FORK_AWARE` admission compared only against pressure-eligible victims and
  used a strict greater-than comparison. A full cache of protected or
  equal-valued entries could therefore stop making LRU progress, while a more
  valuable incoming entry could fail to replace a less valuable protected one.
- Disk-to-CPU promotion unconditionally polluted L1. Mandatory reads now use
  CPU memory as transient staging when necessary, but remain in CPU only when
  their value justifies admission. Disk admission applies the same comparison
  and falls back to an emergency evictable victim under capacity pressure.

An end-to-end `FORK_AWARE` LMCache smoke used a 0.10 GiB CPU tier, 1 GiB local
disk tier, 2K prefix, four branches, and Qwen3-1.7B. It completed successfully
and demoted three chunks totaling 88,080,384 bytes to disk. The first attempt
also exposed an unrelated initialization bug: LMCache installed an OTel
`LoggingHandler` while OpenTelemetry still exposed only a proxy provider. The
logger now attaches that handler only after a real SDK `LoggerProvider` exists.
The smoke artifact is under
`benchmark/results/investigation_20260714/lmcache_tiered_fix_smoke/`.

### Historical Mixed-Prefix Result

The following table compares the old complete optimized policy against
ordinary ForkAttention LRU offload. Positive throughput is an improvement;
negative TTFT and KV load values are reductions. It remains a historical
system stress result, not an isolated connector comparison.

| Model | Dataset | Prefix | Branches | Throughput | TTFT P50 | KV load |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | AgentBoard | 8K | 8 | +57.2% | -54.8% | -75.5% |
| Qwen3-1.7B | AgentBoard | 8K | 16 | +84.6% | -67.1% | -87.3% |
| Qwen3-1.7B | AgentBoard | 16K | 8 | +88.9% | -62.7% | -82.1% |
| Qwen3-1.7B | AgentBoard | 16K | 16 | +129.4% | -70.8% | -85.1% |
| Qwen3-1.7B | AppWorld | 8K | 8 | +78.9% | -67.8% | -80.2% |
| Qwen3-1.7B | AppWorld | 8K | 16 | +102.3% | -71.3% | -86.6% |
| Qwen3-1.7B | AppWorld | 16K | 8 | +80.3% | -58.0% | -80.6% |
| Qwen3-1.7B | AppWorld | 16K | 16 | +128.6% | -73.5% | -84.9% |
| Qwen3-1.7B | AgencyBench | 8K | 8 | +78.0% | -60.8% | -82.1% |
| Qwen3-1.7B | AgencyBench | 16K | 16 | +150.6% | -70.6% | -86.1% |
| Qwen3-1.7B | SWE-bench | 16K | 16 | +141.7% | -70.5% | -85.5% |
| Llama-3.2-1B | AgentBoard | 16K | 16 | +77.9% | -55.5% | -86.0% |
| Llama-3.2-1B | AppWorld | 16K | 16 | +58.5% | -50.3% | -86.7% |
| Llama-3.2-1B | AgencyBench | 16K | 16 | +73.6% | -56.8% | -85.0% |
| Llama-3.2-1B | SWE-bench | 16K | 16 | +73.4% | -58.5% | -85.2% |

Across the 11 complete Qwen cells, the unweighted mean improvement is 101.9%
for output throughput and 66.2% for TTFT P50, while CPU-to-GPU KV load falls
83.3%. The sampled total physical GPU+CPU KV peak changes by only -0.5% on
average. This is expected: the policy changes residency and reload frequency,
not the amount of useful KV retained. ForkAttention's logical KV read reduction
is reported separately and does not imply duplicate physical vLLM blocks.
Across the four Llama stress cells, throughput rises 70.9%, TTFT P50 falls
55.3%, and KV load falls 85.7%, confirming that the offload result is not
specific to Qwen3.

## Data-Parallel Results

### Current Capacity-Aware Router Follow-up

The current router was retested on two RTX 5090 GPUs with Qwen3-8B FP16, two
internal DP ranks, ForkAttention on both variants, Prefix Forest CUDA Graphs,
3,852 KV blocks per rank, `max_num_seqs=64`, and no offload. Each run contains
four cases and 16 branches per case at concurrency 64, with a deterministic
lognormal mean-256 suffix, 256 output tokens, and seed 2026. The only policy
difference is ordinary DP versus the final capacity-aware prefix router.

| Scenario / variant | Branch tok/s runs | Median | Branch phase | TTFT P50 | TPOT P50 | Peak KV | Max waiting | Preemptions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Warm8K, ordinary DP | 2,142.3 / 2,124.5 / 2,014.0 | 2,124.5 | 7,711.8 ms | 1,002.3 ms | 24.73 ms | 76.40% | 14 | 0 |
| Warm8K, final router | 2,210.3 / 2,116.7 / 2,021.3 | 2,116.7 | 7,740.2 ms | 1,003.4 ms | 23.82 ms | 76.53% | 14 | 0 |
| Pressure16K, ordinary DP | 443.6 / 385.8 / 454.3 | 443.6 | 36,937.9 ms | 17,253.2 ms | 21.66 ms | 88.28% | 56 | 0 |
| Pressure16K, final router | 1,677.2 / 1,689.4 / 1,716.0 | 1,689.4 | 9,698.4 ms | 1,432.5 ms | 31.32 ms | 75.19% | 2 | 0 |

Warm8K uses an exact warm prefix and case-major order. All 72 requests per run
take the native small-prompt bypass, so the final policy is neutral within
variance: -0.37% branch throughput, +0.37% branch-phase time, +0.11% TTFT P50,
and -3.67% TPOT P50 relative to ordinary DP. This is intentional; the current
router does not impose prefix hashing or an arrival wave where a rank has
enough capacity for the short prompt.

Pressure16K uses no explicit warmup and a deterministic four-case shuffle.
Here the final router improves median branch throughput by 280.87% (3.81x),
reduces branch-phase time by 73.74%, reduces TTFT P50 by 91.70%, and lowers
peak KV usage by 13.09 percentage points. Every final run records 64/64
affinity routes and cohort locks, a balanced 34/34 request split, at most two
waiting requests, and zero preemptions.

The final router's higher TPOT P50 does not contradict the throughput gain.
Ordinary DP leaves up to 56 requests waiting before decode and has a
17.3-second TTFT P50, while the routed cohort starts promptly and keeps both
ranks occupied.

The evidence supports a scoped claim: the current router is neutral for
bypassed 8K traffic and high-value under long-prefix memory pressure; it is
not expected to accelerate every DP request. Detailed routing activity,
pressure-run methodology, and validation notes are recorded in
[`dp_experiment_results.md`](dp_experiment_results.md).

## TP Accuracy Guardrail

Each dataset cell compares 264 outputs: 256 branches and eight common-analysis
outputs. Values below are unweighted means over the four 16K/32 cells.

| Comparison | Exact match | Token F1 | Text similarity |
|---|---:|---:|---:|
| Fork run 1 vs Flash | 91.10% | 98.25% | 97.16% |
| Fork run 2 vs Flash | 91.76% | 98.60% | 97.60% |
| Fork run 2 vs Fork run 1 | 94.03% | 98.59% | 97.65% |

AgentBoard Flash repeatability is 92.80% exact match and 98.40% token F1,
which is comparable to Fork repeatability on that cell (91.29% and 98.10%).
SWE-bench is the strict-exact outlier: both Flash and Fork are internally
repeatable (98.48% and 98.86%), but Flash-to-Fork exact match is 85.61% while
token F1 remains 98.45%. The implementation therefore passes a strong
token-level agreement guardrail but is not bitwise or text-exact equivalent.
Because the bundled snapshots do not provide a common executable evaluator,
this experiment does not establish environment-level task accuracy.

## Artifacts

Detailed P50/P95/P99 latency, TPOT, throughput, compute utilization,
memory-controller utilization, physical/logical KV metrics, offload traffic,
and provenance are preserved in the committed report snapshots:

- [Single-GPU report](experiment_results/single_gpu.md) ([CSV](experiment_results/single_gpu.csv))
- [TP accuracy report](experiment_results/tp_accuracy.md) ([CSV](experiment_results/tp_accuracy.csv))

Request-level JSON, telemetry samples, and server logs remain in the ignored
`benchmark/results/main_experiment/` tree on the machines that ran each group.

The current experiment procedures and applicable run counts are in
[`main_experiment_matrix.md`](main_experiment_matrix.md).
