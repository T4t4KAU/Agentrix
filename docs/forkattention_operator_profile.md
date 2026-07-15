# ForkAttention CUDA Operator Profile

## Scope and Result

This note isolates the CUDA operator behavior behind the corrected single-GPU
ForkAttention result. It uses an unprofiled API benchmark for end-to-end
latency and throughput, a matched Nsight Systems capture for kernel-time
attribution, and Nsight Compute for one low-batch tail kernel. The profile is
for the no-offload path; CPU or disk KV movement is outside this operator
analysis.

The main result is that ForkAttention is already effective for the intended
8K-prefix, 16-branch workload. In the matched Nsight Systems captures, total
attention kernel time falls from 1,661.09 ms to 404.85 ms, a 4.10x reduction.
Attention accounts for 97.65% of the total GPU-kernel time saved. The most
valuable remaining work is therefore not a wholesale kernel rewrite. The
first high-return item, tail-adaptive prefix splitting, has now been
implemented for both eager and CUDA Graph execution. The remaining work is:

1. validate the policy on wider cohorts and mixed-case concurrency;
2. classify fallback reasons before extending backend coverage;
3. autotune the existing tile choices for `sm_120` and similar devices.

Gather fusion, raw metadata-copy bandwidth, and register-count reduction on
their own are lower-return projects according to the current traces.

## Profile Configuration

| Setting | Value |
|---|---|
| GPU | NVIDIA RTX 5070, 48 SMs, 12 GiB |
| Build | CUDA build with native `sm_120` ForkAttention kernels |
| Model | Qwen3-0.6B, FP16 |
| Workload | one case, target 8K shared prefix, 16 branches |
| Decode | 128 output tokens per branch |
| vLLM | prefix caching enabled, CUDA Graphs enabled |
| Capacity | `max_model_len=10240`, `max_num_seqs=32`, GPU utilization 0.70 |
| Attention backends | `FLASH_ATTN` and `FORK_ATTN`, otherwise matched |
| Profilers | Nsight Systems 2025.5.2 and Nsight Compute 2025.4.1 |

The unprofiled result is stored in
`benchmark/results/investigation_20260714/ideal_c1_p8192_b16_o128/`. The
profiler reports are stored in the sibling `nsys_flash_ideal/`,
`nsys_fork_ideal/`, and `ncu_fork_ideal/` directories. These are repository
relative paths; the model can be supplied as either a model registry name or a
local `<model-dir>/Qwen3-0.6B` path.

## End-to-End Result Without a Profiler

| Metric | FlashAttention | ForkAttention | Fork speedup |
|---|---:|---:|---:|
| Case wall time | 4,296.66 ms | 1,864.91 ms | 2.304x |
| End-to-end output throughput | 476.65 tok/s | 1,098.18 tok/s | 2.304x |
| Branch-phase wall time | 3,383.56 ms | 947.66 ms | 3.570x |
| Branch output throughput | 605.28 tok/s | 2,161.11 tok/s | 3.570x |
| Common-phase latency | 758.51 ms | 765.06 ms | 0.991x |

The common phase is effectively unchanged, while the branch phase improves by
3.57x. This localizes the gain to the shared-prefix decode regime rather than
model loading, prefill, or unrelated model kernels.

## Nsight Systems Kernel Breakdown

The table sums GPU kernel durations over the complete matched captures. Kernel
time is useful for attribution but is not wall time: kernels on different
streams can overlap, and profiler instrumentation changes end-to-end timing.

| GPU work over capture | FlashAttention | ForkAttention | Change |
|---|---:|---:|---:|
| Fork split kernels | - | 169.46 ms | specialized path |
| Fork gather and attention-state merge | - | 4.19 ms | specialized path |
| Flash split/combine kernels | 1,661.09 ms | 231.20 ms | -86.08% |
| All attention kernels | 1,661.09 ms | 404.85 ms | -75.63%, 4.103x |
| Non-attention kernels | 362.96 ms | 332.68 ms | -8.34% |
| All GPU kernels | 2,024.05 ms | 737.53 ms | -63.56%, 2.744x |
| All kernel launches | 52,098 | 51,114 | -1.89% |

The dominant baseline branch kernel alone takes 1,321.14 ms. The complete
Fork split/gather/merge path takes 173.65 ms, which is 7.61x smaller. The
broader 4.10x attention result is the safer full-capture number because it also
counts the 231.20 ms of FlashAttention kernels that remain in the Fork run.

Three conclusions follow from this breakdown:

- The speedup is caused by eliminating repeated attention work and KV reads,
  not by reducing the total launch count. Total launches fall by only 1.89%.
- The remaining Flash split/combine kernels are 31.35% of all Fork-run GPU
  time and 57.11% of its attention time, but launch-count analysis indicates
  that most are not eligible shared decode work. They must not be treated as a
  31.35% optimization opportunity without per-reason counters.
- Once attention is reduced, non-attention model work becomes 45.11% of the
  Fork capture. This is an Amdahl-law floor for operator-only optimization.

### CUDA API and Copy Attribution

Nsight Systems attributes 537.69 ms of Fork-run CPU API duration to
`cudaMemcpyAsync`, but this is not 537.69 ms of GPU copy work. The corresponding
GPU copy engines process only 10.24 MB in 1.89 ms; FlashAttention processes
8.75 MB in 1.69 ms. The long CPU API duration is synchronization/dependency
wait charged to the call site during autoregressive decode. In the Flash run,
similar waiting is mainly charged to `cudaEventSynchronize`.

Consequently, removing a few metadata copies is useful for eager-path CPU
overhead and graph misses, but raw H2D/D2H copy bandwidth is not a high-yield
GPU optimization in this capture. Production graph-hit paths already use
persistent device workspaces; any metadata project should first measure graph
hit rate, host planning/allocation time, and step latency rather than treating
CPU API duration as transfer time.

### Reclassification of Remaining FlashAttention Work

The earlier profile grouped every FlashAttention kernel in the Fork run as a
possible backend miss. The per-specialization launch counts allow a narrower
interpretation:

- The largest remaining Flash split kernel has 1,792 launches, exactly
  `64 decode steps x 28 model layers`. The benchmark has a 64-token singleton
  common-generation phase, where there is no sibling request with which to
  share KV reads. Its matching combine kernel also has 1,792 launches.
- The other Flash split kernels have 280 and 112 launches, or 10 and 4
  complete 28-layer passes. Their template shapes and placement are consistent
  with prefill/chunked-prefill and setup work rather than a shared decode
  cohort.

This is an inference from the launch arithmetic and benchmark trace, not a
per-request kernel label. The new `PROFILE_FORK=1` reason counters should be
used in the next capture to confirm it. Until then, extending ForkAttention to
more fallback paths is P1 validation work rather than an assumed P0 speedup.

## Nsight Compute Tail-Kernel Analysis

The sampled kernel is
`fork_fwd_splitkv_kernel<half, float, 16, 64, 128, 1, 2>`. It represents a
two-request tail step, not the main 16-request cohort, so its occupancy result
must not be generalized to every ForkAttention launch.

| Metric | Measured value | Interpretation |
|---|---:|---|
| Duration | 87.04 us | one sampled tail kernel |
| Grid / block | 56 blocks / 32 threads | 1.17 blocks per SM on 48 SMs |
| Waves per SM | 0.58 | insufficient parallel work for a long-lived tail kernel |
| DRAM throughput | 61.86% | memory pressure dominates |
| Compute throughput | 11.57% | not compute-bound |
| Dynamic shared memory | 36.86 KiB/block | limits residency to two blocks/SM |
| Registers | 210/thread | high, but allows eight blocks/SM in this shape |
| Theoretical occupancy | 4.17% | two active one-warp blocks per SM |
| Achieved occupancy | 2.32% | 1.11 active warps per SM |

For this shape, shared memory is the immediate residency limit: Nsight reports
block limits of 2 from shared memory, 8 from registers, and 48 from warps. The
kernel traits explain the 36 KiB footprint: the `M=16`, head-dimension 128 query
tile consumes about 4 KiB and the `N=64` K/V tiles consume about 32 KiB.
Reducing registers alone therefore cannot raise current residency.

The stronger issue is the small and imbalanced grid. Nsight reports that the
most active SM has 35.26% more active cycles than average and the least active
SM has 83.96% fewer. Its workload-distribution rule estimates a possible
22.46% global speedup if that imbalance were removed. This percentage is an
Nsight heuristic, not a measured projection, but it gives a concrete reason to
prioritize adaptive splitting for tail steps.

## Bottleneck Summary

The operator has two distinct regimes:

- In a full shared cohort, ForkAttention's algorithmic KV reuse wins. The
  7.61x specialized-path and 4.10x all-attention reductions show that memory
  traffic avoided at the algorithm level is more important than a small
  per-kernel inefficiency.
- As requests finish, cohorts become small. A one-warp CTA with a static 2,048
  token prefix chunk can leave the GPU with roughly one block per SM, while
  synchronization inside the K/V pipeline prevents that one warp from hiding
  latency. This is the clearest remaining kernel bottleneck.

The implementation contains the mechanisms needed to address much of this:
split outputs and a gather kernel, `M/N/warp` specializations including
`N=32`, CUDA Graph workspace buckets, and forest plans. Shape-aware prefix
splitting is now implemented; measured tile selection remains.

## Implemented Tail-Adaptive Split Policy

The implementation now targets about two CTA waves only for long-prefix,
small-cohort decode. The default policy is:

- retain the 2,048-token base chunk for short prefixes and already-wide
  cohorts;
- enable adaptive splitting at 4,096 prefix tokens;
- estimate physical CTA demand from active requests, Q/KV head ratio, KV-head
  count, and the device SM count;
- cap common-prefix CUDA Graph plans to captured `4,8,12` chunk buckets and
  forest plans to the gather kernel's 32 splits per request;
- set `VLLM_FORK_ATTN_TARGET_CTA_WAVES=0` to reproduce the static policy.

CUDA Graph state changes the planning mechanism, so both paths were changed:

| Execution mode | Planning path | Adaptive behavior |
|---|---|---|
| CUDA Graph enabled | Scheduler selects a captured common/forest capacity, then the metadata builder fills a persistent device workspace. | Scheduler and runtime use the same request/head geometry. An 8K, two-request common prefix selects `common:12` instead of being capped at `common:4`. |
| CUDA Graph disabled | The metadata builder constructs a prefix trie and dynamic forest boxes for the current batch. | Long shared segments are re-chunked before packing; the same two-request shape produces 10 shared-prefix boxes plus two suffix boxes. |

This separation matters: changing only the runtime chunk size is ineffective
if the CUDA Graph scheduler has already selected a four-chunk workspace.

### CUDA Graph On/Off A/B

The matched service benchmark uses four 8K-prefix AgentBoard cases, two equal
branches per case, concurrency two, 128 output tokens per branch, Qwen3-0.6B
FP16, `max_model_len=10240`, and `max_num_seqs=32`. The static control sets
target waves to zero; the candidate uses the default target of two. Tile
override is disabled in both.

| Execution | Metric | Static chunks | Adaptive chunks | Change |
|---|---|---:|---:|---:|
| CUDA Graph on | Branch-phase wall time | 3,063.60 ms | 2,777.67 ms | -9.33% |
| CUDA Graph on | Branch output throughput | 334.25 tok/s | 368.65 tok/s | +10.29% |
| CUDA Graph on | End-to-end output throughput | 192.44 tok/s | 201.77 tok/s | +4.85% |
| CUDA Graph on | TPOT P50 | 5.68 ms | 5.11 ms | -9.89% |
| CUDA Graph on | TTFT P50 | 41.97 ms | 42.37 ms | +0.97% |
| CUDA Graph off | Branch-phase wall time | 3,415.47 ms | 3,341.52 ms | -2.17% |
| CUDA Graph off | Branch output throughput | 299.81 tok/s | 306.45 tok/s | +2.21% |
| CUDA Graph off | End-to-end output throughput | 175.33 tok/s | 177.83 tok/s | +1.43% |
| CUDA Graph off | TPOT P50 | 6.35 ms | 6.19 ms | -2.57% |
| CUDA Graph off | TTFT P50 | 48.43 ms | 49.25 ms | +1.71% |

Two independent one-case repeats provide a consistency check. With CUDA Graphs
enabled, their mean branch throughput improves by 8.80% and mean TPOT falls by
8.63%; both repeats move in the same direction. With graphs disabled, the
corresponding means move by only +1.12% and -1.41%, with one adaptive repeat
faster and one slower. The four-case eager result is positive, but materially
less robust than the graph result.

CUDA Graph capture completed for both FULL and PIECEWISE graphs. Plan telemetry
confirms that the four-case run raises shared CTAs from 2,579 to 4,154 while
singleton CTAs remain effectively unchanged. The eager dynamic forest raises
shared CTAs from 2,579 to 4,146. A profiling-only eager run measures about
0.65-0.67 ms of forest metadata construction for both policies, so the smaller
eager gain is not explained by extra adaptive planner time. The likely causes
are dilution by uncaptured model/launch overhead and the different persistent
common-plan versus dynamic-forest kernel topologies.

Before the two-request branch phase, the reason log records 63
`single_request` steps, 9 `non_decode` steps, and 1 `no_shared_kv` step; Fork
telemetry subsequently records 126 active two-request steps. This independently
supports the launch-count inference that the largest remaining FlashAttention
group is not eligible shared decode.

### Wide-Cohort Guardrail

The 8K-prefix, 16-branch guardrail uses the same model and server settings,
with concurrency and branch-group size both set to 16. The four-case aggregate
was run with CUDA Graphs both enabled and disabled.

| Execution | Metric | Static chunks | Adaptive policy | Change |
|---|---|---:|---:|---:|
| CUDA Graph on | Branch-phase wall time | 3,586.12 ms | 3,592.19 ms | +0.17% |
| CUDA Graph on | Branch output throughput | 2,284.36 tok/s | 2,280.50 tok/s | -0.17% |
| CUDA Graph on | End-to-end output throughput | 1,373.14 tok/s | 1,369.96 tok/s | -0.23% |
| CUDA Graph on | TPOT P50 | 6.07 ms | 6.08 ms | +0.22% |
| CUDA Graph on | TTFT P50 | 112.80 ms | 115.95 ms | +2.79% |
| CUDA Graph off | Branch-phase wall time | 5,320.30 ms | 5,288.08 ms | -0.61% |
| CUDA Graph off | Branch output throughput | 1,539.76 tok/s | 1,549.14 tok/s | +0.61% |
| CUDA Graph off | End-to-end output throughput | 1,046.93 tok/s | 1,049.73 tok/s | +0.27% |
| CUDA Graph off | TPOT P50 | 9.50 ms | 9.46 ms | -0.40% |
| CUDA Graph off | TTFT P50 | 123.06 ms | 110.38 ms | -10.31% |

Two independent one-case repeats show a mean branch-throughput decrease of
0.55% and TPOT increase of 0.75%. The longer four-case result and identical CTA
counts show no meaningful wide-cohort regression: with 16 active requests the
base four-chunk plan already supplies the target CTA count, so adaptive
splitting does not activate. Graph-on telemetry is exactly identical at 2,569
shared CTAs, 9,298 singleton CTAs, and 561 active Fork steps. Eager telemetry
has 564 versus 561 active steps, which explains its small total-CTA difference;
the per-step topology remains unchanged. The TTFT movements in both directions
are outside the unchanged decode plan and are treated as run-to-run noise.

The artifacts are under
`benchmark/results/investigation_20260715/{graph,eager}_c4_b2_adaptive*/` and
the one-case repeats are in the nearby `graph_b2_*` and `eager_b2_*`
directories. Wide-cohort artifacts use `graph_b16_*` and `graph_c4_b16_*`.

### Tail Tile A/B

For the same four-case eager workload with adaptive splitting enabled, forcing
`N=32` instead of the automatic `N=64` tail tile reduces branch throughput by
1.69% and increases TPOT by 1.60%. Smaller shared-memory residency does not
offset the extra loop work after adaptive splitting has supplied enough CTAs.
The override remains available for other shapes, but `N=32` is not selected as
the `sm_120` default from this result.

## Optimization Priority

| Priority | Optimization | Evidence and expected scope | Main risk |
|---|---|---|---|
| Implemented; monitor | Adaptive prefix split-K / CTA count | The new two-wave policy improves two-request branch throughput by 10.29% with CUDA Graphs and 2.21% in eager mode. The 16-request guardrail changes throughput by -0.17% with graphs and +0.61% in eager mode, with no topology change. | Mixed request counts and irregular completion order still need longer validation. |
| P1 validation | Classify and convert eligible Flash fallbacks | Flash kernels remain 231.20 ms, but launch arithmetic assigns the largest group to 64 singleton decode steps. Use reason counters before extending coverage. | Prefill, no-sharing, and unsupported shapes should continue to fall back for correctness. |
| P1 | `sm_120` tile autotuning | `N=32` uses less shared memory than `N=64`, but regresses branch throughput by 1.69% in the measured two-request eager shape. Keep automatic `N=64` and sweep other cohort/KV lengths before adding an architecture table. | `N=32` doubles loop iterations and can regress once adaptive splitting supplies enough CTAs. |
| P1 | K/V load-path and cache-efficiency tuning | 61.86% DRAM versus 11.57% compute confirms memory pressure. Inspect sector utilization/replay in a full-cohort NCU sample before changing vectorization or page traversal. | The current algorithm already removes most repeated bytes; micro-tuning has a lower ceiling than CTA/fallback work. |
| P1 | FP8 KV specialization | A supported FP8 path could approximately halve KV bytes for memory-bound kernels. | New kernels, numerical validation, scale handling, and broader compatibility work; current Fork dispatch intentionally accepts FP16/BF16 KV only. |
| P1/P2 | Persistent GPU forest planning on graph misses | Eager forest construction still creates host plan objects and tensors, while graph-hit paths reuse persistent workspaces. Optimize only if reason counters show material eager/graph-miss time. | GPU copy time itself is only 1.89 ms, so optimizing the wrong layer will not move throughput. |
| P2 | Prefix/suffix/gather fusion | Fork gather and merge consume only 4.19 ms, 0.57% of total Fork GPU time. Fusion is worthwhile only if it also removes partial-state traffic or enables better adaptive splitting. | Considerable kernel complexity for little direct launch-time saving. |
| P2 | Register reduction alone | 210 registers/thread looks high, but shared memory currently limits blocks per SM first. | May increase instructions or spills without changing occupancy. |

### Recommended Implementation Order

1. Run longer mixed 2/4/8/16-request traces with irregular completion order;
   retain the policy only if the transition between cohort sizes stays stable.
2. Correlate the new `PROFILE_FORK=1` reason counters with an Nsys capture to
   attach kernel time, not only step counts, to each fallback class.
3. Sweep the already compiled `N=32` and `N=64` one-warp kernels on `sm_120`,
   then use a small architecture-specific dispatch table.
4. Re-run the unprofiled 8K/16 benchmark and Nsys capture. Accept a change only
   if it improves tail latency without reducing full-cohort branch throughput.
5. Capture NCU once for a full 16-request cohort and once for the 1-2 request
   tail before attempting lower-level K/V or register work.

This order targets the largest measured residuals first and keeps the
algorithm's demonstrated full-cohort benefit intact.

## Reproducing the Profile

Run from the repository root and supply a model registry name or a portable
model path:

```bash
cd benchmark

MODEL_PATH=<model-dir>/Qwen3-0.6B \
PREFIX_TOKENS=8192 \
BRANCHES=16 \
OUTPUT_TOKENS=128 \
MAX_MODEL_LEN=10240 \
MAX_NUM_SEQS=32 \
OUTPUT_DIR=results/fork_attention_nsys_8k16 \
./scripts/run_fork_attention_nsight.sh
```

Summarize stored reports without embedding machine-specific locations:

```bash
nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  --format csv \
  benchmark/results/investigation_20260714/nsys_fork_ideal/fork_ideal.nsys-rep

ncu --import \
  benchmark/results/investigation_20260714/ncu_fork_ideal/fork_kernel.ncu-rep \
  --page details --csv
```

Use the same request seed and server settings for the FlashAttention control.
Do not compare a profiled backend with an unprofiled backend, and do not infer
end-to-end speedup from the sum of kernel durations.
