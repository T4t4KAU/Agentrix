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
valuable remaining work is therefore not a wholesale kernel rewrite, but:

1. create more useful CTAs for long-KV, small-cohort tail steps;
2. reduce the decode attention work that still falls back to FlashAttention;
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
  time and 57.11% of its attention time. Some are required prefill or singleton
  fallbacks, but eligible decode graph misses and incomplete cohorts are a
  large target.
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

The current implementation already contains the mechanisms needed to address
much of this: split outputs and a gather kernel, `M/N/warp` specializations
including `N=32`, CUDA Graph workspace buckets, and forest plans. The missing
piece is measured shape-aware selection rather than a single static prefix
chunk and the current coarse cohort/KV-length thresholds.

## Optimization Priority

| Priority | Optimization | Evidence and expected scope | Main risk |
|---|---|---|---|
| P0 | Adaptive prefix split-K / CTA count | Tail sample launches only 56 CTAs on 48 SMs and is strongly imbalanced. Reduce the 2,048-token chunk only when a long-KV launch would provide fewer than roughly two CTA waves. High expected return for 1-4 request tail steps; end-to-end return depends on their time share. | Extra partial-output traffic and gather work can erase the gain if applied to wide cohorts. |
| P0 | Convert eligible Flash fallbacks | Flash kernels remain 231.20 ms, 31.35% of total Fork GPU time. Track fallback reasons separately and extend forest/graph bucket coverage for shared decode steps. High system-level return when fallback is caused by scheduling or graph coverage. | Prefill, no-sharing, and unsupported shapes should continue to fall back for correctness. |
| P0/P1 | `sm_120` tile autotuning | The existing `M=16,N=32,warps=1` specialization needs about 20 KiB shared memory versus about 36 KiB for `N=64`, potentially allowing more resident blocks once adaptive splitting supplies them. Benchmark `N=32` and `N=64` by cohort size and KV length, then dispatch from a measured table. | `N=32` doubles loop iterations and can regress well-populated grids. |
| P1 | K/V load-path and cache-efficiency tuning | 61.86% DRAM versus 11.57% compute confirms memory pressure. Inspect sector utilization/replay in a full-cohort NCU sample before changing vectorization or page traversal. | The current algorithm already removes most repeated bytes; micro-tuning has a lower ceiling than CTA/fallback work. |
| P1 | FP8 KV specialization | A supported FP8 path could approximately halve KV bytes for memory-bound kernels. | New kernels, numerical validation, scale handling, and broader compatibility work; current Fork dispatch intentionally accepts FP16/BF16 KV only. |
| P1/P2 | Persistent GPU forest planning on graph misses | Eager forest construction still creates host plan objects and tensors, while graph-hit paths reuse persistent workspaces. Optimize only if reason counters show material eager/graph-miss time. | GPU copy time itself is only 1.89 ms, so optimizing the wrong layer will not move throughput. |
| P2 | Prefix/suffix/gather fusion | Fork gather and merge consume only 4.19 ms, 0.57% of total Fork GPU time. Fusion is worthwhile only if it also removes partial-state traffic or enables better adaptive splitting. | Considerable kernel complexity for little direct launch-time saving. |
| P2 | Register reduction alone | 210 registers/thread looks high, but shared memory currently limits blocks per SM first. | May increase instructions or spills without changing occupancy. |

### Recommended Implementation Order

1. Export per-reason fallback counters and distributions of active requests,
   CTAs, KV length, selected tile, and prefix split count.
2. Add a tail-only adaptive chunk-size policy with a minimum target of about
   two CTA waves, capped by the existing 32-split gather workspace.
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
