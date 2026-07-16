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
shape. The two-rank Qwen3-8B DP result uses the current adaptive-graph path at
16K/16 and 32K/32. Qwen3-14B TP accuracy uses the maximum 16K/32 shape.

The Qwen single-GPU core grid consumes all 9 AgentBoard and 13 AppWorld
records. Its AgencyBench 8K/8 cell uses 32 records; all 16K/16 stress
replications use eight records. Adaptive16K uses four AgencyBench cases per
run; Pressure32K/32 uses two cases and 32 branches per case. TP accuracy uses
the first eight records per dataset. Every comparison within a cell uses
identical prompts, and reported cross-cell means are unweighted.

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

Model: Qwen3-8B on two internal DP ranks. Adaptive16K uses four shuffled cases
with 16 branches each; Pressure32K/32 uses two shuffled cases with 32 branches
each. Neither workload explicitly warms the shared prefix.

Variants:

- FlashAttention with ordinary DP and no offload, as the primary baseline
- ForkAttention with ordinary DP and no offload
- ForkAttention with the final capacity-aware DP router and no offload

The FlashAttention baseline retains the same workload, CUDA Graph policy, and
KV capacity. The two ForkAttention variants then isolate the backend and
routing contributions. The client does not force a data-parallel rank.

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

The current Adaptive16K and Pressure32K/32 DP setup, routing controls, and
results are recorded in
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
| Data parallel | 2x RTX 5090 32 GiB | 12.8 | 18 focused Qwen3-8B runs: Adaptive16K validation, repeated AgencyBench Pressure32K/32, and three-way checks on three additional datasets |
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

### Cross-Dataset Pure-GPU Revalidation

A larger pure-GPU revalidation used Qwen3-1.7B, 16K prefixes, 16 branches,
256-token outputs, exact shared-prefix warm-up, forest CUDA Graphs, and 1,700
GPU blocks for both backends. It covered 32 SWE-bench records, all 32
AgencyBench records, all 9 AgentBoard records, and all 13 AppWorld records: 86
independent roots and 1,376 measured branch requests per backend. FlashAttention
and ForkAttention had identical per-case input-token counts and zero
preemptions. ForkAttention physically activated on about 76% of observed steps.

| Dataset | Records | Flash tok/s | Fork tok/s | Throughput improvement | TTFT P50 Flash/Fork | TPOT P50 Flash/Fork |
|---|---:|---:|---:|---:|---:|---:|
| SWE-bench | 32 | 257.10 | 667.12 | +159.47% | 470.51/424.94 ms | 52.73/13.56 ms |
| AgencyBench | 32 | 251.91 | 684.74 | +171.82% | 462.62/430.39 ms | 53.71/13.21 ms |
| AgentBoard | 9 | 247.41 | 649.72 | +162.61% | 471.64/440.00 ms | 54.82/14.11 ms |
| AppWorld | 13 | 250.23 | 676.96 | +170.54% | 474.64/436.01 ms | 54.30/13.31 ms |

Across all 86 paired cases, the median Fork/Flash speedup was 2.66x; the
minimum and maximum were 2.44x and 2.91x. This is a controlled, dataset-seeded
serving benchmark: each real record is padded to the same long-prefix shape and
expanded into branches. It establishes cross-dataset systems consistency, not
interactive task success.

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

The same 16K/16 four-root experiment was repeated three times after the fix and
again on the current checkout. These production measurements used
`FANOUT_PROFILE=0`; per-step fanout logging was used only in separate diagnostic
runs because it writes one detailed line per scheduler step and materially
perturbs the optimized variant. The current three-run medians are:

| Variant | Output tok/s | Branch TTFT P50 | Branch TPOT P50 | KV load | Load ops | KV store | Store ops | Fork active steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Flash ordinary offload | 240.88 | 23,747.12 ms | 51.36 ms | 10.71 GiB | 7 | 8.26 GiB | 103 | - |
| Fork ordinary offload | 549.37 | 10,959.79 ms | 12.50 ms | 9.59 GiB | 9 | 8.26 GiB | 103 | 73.03% |
| Fork scheduled ordinary offload | 514.64 | 10,538.35 ms | 12.57 ms | 10.45 GiB | 11 | 8.26 GiB | 103 | 83.41% |
| Fork optimized offload | 616.45 | 7,521.23 ms | 12.56 ms | 8.87 GiB | 5 | 8.15 GiB | 137 | 81.00% |

Across paired current-checkout runs, optimized offload improves throughput by
19.78%, reduces TTFT by 31.74%, reduces CPU-to-GPU KV load by 15.12%, and
leaves TPOT effectively unchanged (-0.48%). All three throughput pairs are
positive (+23.75%, +19.78%, and +13.54%). The complete optimized system is
156.89% faster than Flash ordinary offload, while changing only FlashAttention
to ForkAttention with the ordinary connector improves throughput by 128.93%.
The former is a system-level result; the latter is the cleaner backend
comparison. The optimized and scheduled throughput medians are within 1.6% of
the earlier 626.13 and 518.86 tok/s validation, respectively, although the new
paired reload reduction is smaller than the earlier 25.84% estimate. The higher
optimized store-operation count remains a small-transfer/job-overhead target
even though total store bytes are about 1.3% lower.

This is a dataset-seeded systems benchmark rather than an AgentBoard task-score
evaluation: four real records provide the root content, after which the harness
pads each root to the controlled 16K prefix and creates 16 suffix branches.
It therefore tests the intended long-prefix, multi-branch Agent serving shape,
but does not by itself establish end-to-end Agent task quality.

### Cross-Dataset CPU-Offload Replication

The same cold 16K/16 four-root offload profile was then repeated three times on
SWE-bench and AgencyBench. Both backends used 1,700 GPU blocks and an 8 GiB CPU
tier; every run generated nonzero load/store traffic and no run preempted.

| Dataset | Flash ordinary | Fork ordinary | Paired improvement | Fork optimized | Paired improvement | Optimized KV load |
|---|---:|---:|---:|---:|---:|---:|
| SWE-bench | 218.22 tok/s | 535.39 tok/s | +145.34% | 542.96 tok/s | +143.80% | 8.56 GiB |
| AgencyBench | 233.66 tok/s | 532.37 tok/s | +117.26% | 597.60 tok/s | +155.76% | 8.84 GiB |

The ordinary-connector comparison is the cleaner backend result. Relative to
scheduled ordinary ForkAttention, the optimized connector's paired-median
throughput gain was only 6.13% on SWE-bench and 8.10% on AgencyBench, while KV
reload fell by 11.57% and 16.68%, respectively. Connector performance varied
substantially across individual runs, so the larger Flash-to-Fork system gain
must not be attributed to the connector alone.

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

### FlashAttention-Controlled Disk Reload

A separate small disk experiment compared FlashAttention and ForkAttention
with the same LMCache LRU policy, 2 GiB CPU tier, 8 GiB disk tier, 1,700 GPU
blocks, and forest CUDA Graph policy. Four AgentBoard roots used 4K prefixes,
eight branches, and 64-token outputs. Each server first ran roots 0-3, then
roots 4-7 to evict the target cohort, and finally measured a replay of roots
0-3. The runner records these pre-measurement waves with
`CACHE_PRIME_SAMPLE_INDICES=0,4`.

| Backend | E2E tok/s | Branch tok/s | TTFT P50 | TPOT P50 | Retrieved tokens |
|---|---:|---:|---:|---:|---:|
| FlashAttention + LMCache disk | 522.08 | 819.20 | 422.04 ms | 32.62 ms | 21,248 |
| ForkAttention + LMCache disk | 688.35 | 1,306.04 | 437.60 ms | 16.93 ms | 20,992 |

All three pairs had identical input/output token counts and nonzero disk
retrieval. None reported a disk allocation failure, failed load, or fatal
storage error; the realized disk footprint was about 5.1-5.3 GiB. Fork branch
throughput improved in every pair (+56.12%, +71.29%, and +59.43%), for a
paired median of +59.43%. End-to-end changes were +28.16%, +36.60%, and
-11.89%, for a paired median of +28.16%. The negative third E2E pair shows that
disk/common-stage variability is still material: this result supports stable
branch-phase acceleration, not an assertion that every end-to-end disk run is
faster.

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

### Current Adaptive Forest Results

Adaptive CTA planning exposed a Prefix Forest CUDA Graph capacity bug: a
`forest-64` selection could expand beyond 64 runtime CTAs. The repaired server
path makes bucket selection
adaptive-aware and constrains the final plan to the captured CTA and split
capacities. The default adaptive configuration then completed the three-way
16K comparison and a 32K-prefix, 32-branch-per-case extension without graph
mismatch, preemption, OOM, or request failure.

The following table reports branch-only output throughput.

| Scenario / variant | Branch tok/s runs | Median | Branch phase | TTFT P50 | TPOT P50 | Peak KV |
|---|---|---:|---:|---:|---:|---:|
| Adaptive16K, Flash ordinary DP | 496.3 | 496.3 | 33,009.4 ms | 16,357.7 ms | 23.45 ms | 89.04% |
| Adaptive16K, Fork ordinary DP | 424.1 | 424.1 | 38,634.7 ms | 18,212.5 ms | 22.77 ms | 87.60% |
| Adaptive16K, Fork prefix-aware DP | 1,529.1 | 1,529.1 | 10,714.6 ms | 1,413.3 ms | 35.05 ms | 75.10% |
| Pressure32K/32, Flash ordinary DP | 394.7 / 212.6 | 303.7 | 59,284.8 ms | 32,462.9 ms | 25.47 ms | 59.01% |
| Pressure32K/32, Fork ordinary DP | 264.8 / 377.9 | 321.3 | 52,619.6 ms | 29,645.3 ms | 21.88 ms | 59.74% |
| Pressure32K/32, Fork prefix-aware DP | 1,570.1 / 1,611.9 | 1,591.0 | 10,299.8 ms | 2,153.2 ms | 26.82 ms | 72.62% |

Adaptive16K prefix-aware ForkAttention reaches 3.08x Flash ordinary DP and
3.61x Fork ordinary DP branch throughput in the repair-validation run. For
Pressure32K/32, the two prefix-aware runs vary by only 2.67% and finish with an
exact 33/33 rank allocation; both 32K prefix owners are balanced and all 64
post-bootstrap requests preserve prefix affinity. Ordinary ForkAttention and
FlashAttention show much larger run-to-run variation, so the 32K evidence is
best read as stable prefix-aware throughput above 1,570 branch tok/s rather
than as a single maximum speedup ratio. Against the primary Flash ordinary-DP
baseline, the paired gains are 3.98x and 7.58x; against the Fork ordinary-DP
ablation, they are 5.93x and 4.27x.

The same Pressure32K/32 shape was run once on each remaining bundled dataset.
Flash ordinary DP is the primary baseline; Fork ordinary DP isolates the
routing contribution. Each variant starts a fresh service and receives the
same two fixed records, deterministic 32K prompt construction, suffix budgets,
shuffled 64-request fanout, and output limits. The client does not select DP
ranks. A required 64-token common-analysis request bootstraps each case before
fanout, but no extra exact-prefix warmup request is sent. Branch throughput
measures only the fanout phase, not service startup or common analysis.

| Dataset | Flash ordinary branch tok/s | Fork ordinary branch tok/s | Fork prefix-aware branch tok/s | Gain vs Flash | Gain vs Fork | Prefix-aware TTFT P50 |
|---|---:|---:|---:|---:|---:|---:|
| AgentBoard | 231.4 | 233.7 | 1,007.1 | 4.35x | 4.31x | 2,135.6 ms |
| AppWorld | 173.8 | 163.1 | 1,033.4 | 5.95x | 6.33x | 2,131.5 ms |
| SWE-bench Verified | 185.8 | 174.4 | 1,032.7 | 5.56x | 5.92x | 1,964.3 ms |

All nine cross-dataset runs completed 66/66 requests without preemption. Each
prefix-aware run recorded 64 affinity routes and cohort locks and ended with a
33/33 rank split. The cross-dataset result is one run per variant and dataset,
so it establishes consistency for this controlled pressure shape rather than
a universal average speedup.

The measured shared prefix is 32,850 tokens after chat formatting. The service
uses a conservative 42,560-token configured limit because the harness adds a
25% tokenizer margin, while measured request lengths remain below the model's
declared 40,960-token position limit. The adaptive capacity repair is currently
an uncommitted server-side vLLM patch; it must be synchronized before the new
rows are reproducible from a clean checkout.

The evidence supports a scoped claim: the current router is high-value for the
tested long-prefix memory-pressure shape across all four bundled dataset
sources, but it is not expected to accelerate every DP request. Detailed
routing activity, pressure-run methodology, and validation notes are recorded in
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

## HotpotQA Agentrix Positive Control

The standalone configuration, live-pipeline scope, results, limitations, and
reproduction commands are recorded in
[`hotpot_agentrix_experiment.md`](hotpot_agentrix_experiment.md).

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
