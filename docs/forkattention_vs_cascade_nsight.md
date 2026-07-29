# ForkAttention versus Cascade Attention with Nsight

This workflow compares ForkAttention with vLLM's native two-level Cascade
Attention on identical decode tensors and physically shared paged-KV blocks.
It preserves the native Nsight reports so they can be opened later in the
Nsight Compute and Nsight Systems graphical interfaces.

## Pure-operator Nsight Compute capture

The operator harness first validates ordinary FlashAttention, Cascade
Attention, and ForkAttention outputs against one another. Correctness setup
and warm-up run outside the profiler range. Each profiled process captures
exactly one complete operator invocation.

Run one matched cell from the repository root:

```bash
VLLM_PYTHON=/path/to/vllm/python \
PREFIX_TOKENS=4096 \
PRIVATE_SUFFIX_TOKENS=128 \
BRANCHES=8 \
OUTPUT_DIR="$PWD/benchmark/results/fork_cascade_ncu_p4096_b8" \
./benchmark/scripts/run_fork_cascade_ncu.sh
```

The output directory contains five report groups for each of the three arms:

```text
flash_attn_dram.ncu-rep
flash_attn_l2_total.ncu-rep
flash_attn_l2_hit.ncu-rep
flash_attn_l2_miss.ncu-rep
flash_attn_instructions.ncu-rep
cascade_attn_dram.ncu-rep
cascade_attn_l2_total.ncu-rep
cascade_attn_l2_hit.ncu-rep
cascade_attn_l2_miss.ncu-rep
cascade_attn_instructions.ncu-rep
fork_attn_dram.ncu-rep
fork_attn_l2_total.ncu-rep
fork_attn_l2_hit.ncu-rep
fork_attn_l2_miss.ncu-rep
fork_attn_instructions.ncu-rep
```

Open any `.ncu-rep` directly in Nsight Compute. The matching raw CSV, log, and
per-backend aggregate JSON files are retained beside the UI reports.

### Rich reports for interactive UI analysis

The five split reports per backend intentionally capture minimal counter
groups. They are suitable for controlled aggregation but do not populate most
of Nsight Compute's analysis pages. Generate one full-section report per arm
when interactive inspection is required:

```bash
VLLM_PYTHON=/path/to/vllm/python \
PREFIX_TOKENS=4096 \
PRIVATE_SUFFIX_TOKENS=128 \
BRANCHES=8 \
OUTPUT_DIR="$PWD/benchmark/results/fork_cascade_ncu_ui_p4096_b8" \
./benchmark/scripts/run_fork_cascade_ncu_ui.sh
```

For the local 4K/8 capture, the three reports are already available at:

```text
benchmark/results/fork_cascade_ncu_smoke_p4096_b8/ui_full/
  flash_attention_full.ncu-rep
  cascade_attention_full.ncu-rep
  fork_attention_full.ncu-rep
```

Open all three in `ncu-ui`. In each report, double-click the dominant kernel
row and select `Details`. The full reports populate these useful sections:

- `GPU Speed Of Light Throughput`: SM and memory throughput balance;
- `Launch Statistics`: grid, block, registers, shared memory, and waves per SM;
- `Occupancy`: theoretical and achieved occupancy constraints;
- `Memory Workload Analysis`: DRAM/L1/L2 traffic and hit behavior;
- `Warp State Statistics`: issue efficiency and warp stall reasons;
- `Instruction Statistics`: executed instruction mix.

Inspect these dominant rows first:

| Arm | Dominant kernel |
|---|---|
| Ordinary Flash | `flash_fwd_splitkv_kernel` |
| Cascade | the 150.944-us `flash_fwd_splitkv_kernel`, grid `(1, 1, 16)` |
| Fork | `fork_fwd_splitkv_kernel`, grid `(12, 8, 1)` |

Use `Launch Statistics` to validate the 16-versus-96 CTA difference and
`Occupancy`/`Warp State Statistics` to determine how that launch geometry
affects realized utilization. Use `Memory Workload Analysis` and `Instruction
Statistics` to explain where work is removed.

The full section set requires approximately 40 kernel-replay passes. Its
per-kernel duration and cache state can therefore differ from the controlled
single-pass counter captures. Continue to use the split `*_dram.ncu-rep` and
aggregate JSON files for the published cross-backend time/traffic table; use
the full reports for diagnosis and screenshots.

Cascade consists of shared-prefix FlashAttention, per-query suffix
FlashAttention, and `merge_attn_states_kernel`. Fork consists of `fork_fwd`
and, when required, `gather_kernel`. Compare the sum of every kernel belonging
to an invocation rather than only its longest kernel.

## End-to-end Nsight Systems capture

Run matched fresh-server captures for native Cascade and Fork:

```bash
cd benchmark
MODEL_PATH=/path/to/model \
PREFIX_TOKENS=8192 \
BRANCHES=16 \
OUTPUT_TOKENS=64 \
MAX_MODEL_LEN=32768 \
MAX_NUM_SEQS=16 \
OUTPUT_DIR=results/fork_cascade_nsys_p8192_b16_o64 \
./scripts/run_fork_cascade_nsight.sh
```

The UI reports are written separately as:

```text
results/fork_cascade_nsys_p8192_b16_o64/flash_attention/flash_attention.nsys-rep
results/fork_cascade_nsys_p8192_b16_o64/cascade_attention/cascade_attention.nsys-rep
results/fork_cascade_nsys_p8192_b16_o64/fork_attention/fork_attention.nsys-rep
```

The ordinary `FLASH_ATTN` arm explicitly disables Cascade. The Cascade arm uses
the same backend but explicitly enables native Cascade Attention. The
`FORK_ATTN` arm enables the same shared-prefix metadata and uses it to build
the Fork plan. Confirm all three paths from server logs and GPU kernel names:

- Cascade: two `flash_fwd` phases followed by `merge_attn_states_kernel`;
- Fork: `fork_fwd` plus optional `gather_kernel`;
- a trace containing only ordinary per-request `flash_fwd` is not a valid
  Cascade comparison;
- a Fork server log reporting fallback is not a valid Fork measurement.

Nsight Systems is the system-level view and includes model layers, scheduling,
CUDA Graph replay, and API execution. Nsight Compute is the source for detailed
operator counters. Do not mix their kernel durations into one speedup number.

## H20 4K/8 operator validation

The matched operator harness was also validated on an NVIDIA H20-3e (SM90)
with Torch 2.11.0+cu129. All three Flash, Cascade, and Fork outputs passed the
same `atol=2e-2`, `rtol=2e-2` comparison. The clean Nsight Systems range
captures use `cudaProfilerApi`, so correctness setup and warm-up kernels are
excluded.

| Arm | Summed target GPU kernel time |
|---|---:|
| Native Cascade | 37.312 us |
| ForkAttention | 21.472 us |

At this 4K-prefix, eight-query, 128-token-private-suffix point, Fork is 1.738x
faster than native Cascade at the operator boundary. On H20, FA3 main kernels
appear in Nsight Systems as templated CUTLASS `device_kernel` rows rather than
names beginning with `flash_fwd`. The Cascade total includes both FA3 main
kernels, two scheduler/prepare launches, the FA3 combine kernel, and
`merge_attn_states_kernel`. The Fork total includes `fork_fwd_splitkv_kernel`
and `gather_kernel`.

UI-openable reports and exported summaries are under:

```text
benchmark/results/h20_fork_cascade_nsys_operator_p4096_b8/
  cascade_attn.nsys-rep
  fork_attn.nsys-rep
  cascade_attn_kernels.csv
  fork_attn_kernels.csv
```

The host used for this capture runs NVIDIA driver 550.144.03. Nsight Compute
can attach with version 2024.3.2, but hardware-counter collection is blocked
by the host driver's `NVreg_RestrictProfilingToAdminUsers` setting and reports
`ERR_NVGPUCTRPERM` even for root inside the container. The `.ncu-rep` counter
matrix must therefore be rerun after an administrator enables performance
counters on the host. Installing Nsight inside the container cannot change
that driver-level permission.

## Local 4K/8 NCU result

The first matched operator capture was run locally on an NVIDIA GeForce RTX
5070 (48 SMs) with PyTorch 2.11.0+cu130. Its shape was:

| Property | Value |
|---|---:|
| Shared prefix | 4,096 tokens |
| Simultaneous decode queries | 8 |
| Private suffix per query | 128 tokens |
| Query heads / KV heads | 16 / 8 |
| Head dimension | 128 |
| KV block size | 16 tokens |
| Dtype | FP16 |

All three outputs passed comparison against ordinary FlashAttention with
`atol=2e-2` and `rtol=2e-2` before profiling. NCU captured one complete
operator invocation after warm-up.

| Metric | Ordinary Flash | Native Cascade | ForkAttention |
|---|---:|---:|---:|
| Total attention-kernel time | 219.328 us | 171.488 us | 56.416 us |
| Speedup versus ordinary Flash | 1.000x | 1.279x | 3.888x |
| Speedup versus Cascade | 0.782x | 1.000x | 3.040x |
| Kernel launches | 2 | 3 | 2 |
| DRAM read bytes | 21,345,536 | 21,228,288 | 21,392,128 |
| L2 read bytes | 139,260,896 | 38,821,920 | 21,772,384 |
| Tensor-pipe warp instructions | 2,162,688 | 589,824 | 81,920 |
| FMA-pipe warp instructions | 1,992,672 | 550,816 | 186,464 |

The reports and their raw exports are stored under:

```text
benchmark/results/fork_cascade_ncu_smoke_p4096_b8/
```

The directory contains fifteen UI-openable `.ncu-rep` files: five independent
counter captures for each of ordinary Flash, Cascade, and Fork. The matching
aggregate files are `flash_attn_summary.json`, `cascade_attn_summary.json`, and
`fork_attn_summary.json`.

### Why Cascade is slower even though it shares the prefix

Cascade does share the prefix. The nearly identical cold-cache DRAM traffic is
evidence of that: both implementations read approximately the unique 4K
prefix plus the eight private suffixes once. The result must therefore not be
described as Fork avoiding shared-prefix DRAM reads that Cascade repeats.

The difference at this point is how the shared work is tiled and scheduled.
The Cascade invocation decomposes into:

| Cascade phase | NCU kernel time | Grid | DRAM reads |
|---|---:|---:|---:|
| Shared-prefix `flash_fwd_splitkv_kernel` | 150.944 us | `(1, 1, 16)` | 16,869,120 bytes |
| Per-query suffix `flash_fwd_splitkv_kernel` | 17.760 us | `(1, 8, 8)` | 4,283,904 bytes |
| `merge_attn_states_kernel` | 2.784 us | `(16, 1, 1)` | 75,264 bytes |

The shared-prefix phase alone accounts for 88.0% of Cascade's measured time.
Its grid exposes only 16 CTAs for the prefix, so it cannot fill the 48 SMs on
this GPU. It is also a general FlashAttention kernel with a 128-query tile,
while this decode cohort contains only eight query rows. Prefix sharing avoids
eight independent KV traversals, but it does not make this tile shape or its
occupancy efficient for a small decode cohort.

Fork uses a decode-specialized multi-query tile and splits the prefix into four
chunks selected by the production adaptive policy. Its main kernel grid is
`(12, 8, 1)`, or 96 CTAs, which supplies two CTA waves across 48 SMs. The
measured decomposition is:

| Fork phase | NCU kernel time | Grid | DRAM reads |
|---|---:|---:|---:|
| `fork_fwd_splitkv_kernel` | 52.544 us | `(12, 8, 1)` | 21,054,464 bytes |
| `gather_kernel` | 3.872 us | `(8, 16, 1)` | 337,664 bytes |

Fork consequently obtains more parallelism while using a query/KV tile aimed
at synchronized decode fanout. NCU also records 43.92% fewer L2 reads, 86.11%
fewer Tensor-pipe instructions, and 66.15% fewer FMA-pipe instructions. These
are the direct explanations for the 3.04x kernel-time result; the external
DRAM byte count is not.

This is one operator point on one GPU. The advantage can change with prefix
length, cohort width, SM count, FlashAttention version, and tile selection. A
prefix/branch matrix is required before making a general Fork-versus-Cascade
claim, particularly around low-query occupancy boundaries.

## Comparative analysis

### Cascade sharing is working

The three arms represent different levels of shared-prefix exploitation:

| Arm | Shared-prefix execution |
|---|---|
| Ordinary Flash | Each query attends over its full prefix and suffix |
| Native Cascade | All queries attend over the common prefix together, then process private suffixes and merge states |
| ForkAttention | A decode-specialized plan assigns shared prefix chunks and private suffixes to cooperative multi-query CTAs, then gathers split states |

At 4K/8, ordinary Flash requested 139.26 MB at the L2 read interface, while
Cascade requested 38.82 MB. This 72.1% reduction directly confirms that
Cascade removed most repeated shared-prefix reads. Its DRAM traffic was only
21.23 MB because many ordinary Flash duplicate reads were already served by
L2. Cascade is therefore not slow because prefix sharing failed.

### Current native Cascade limitation on this shape

The measured limitation is the mapping of the reduced work to the GPU. Native
Cascade reuses the general FlashAttention implementation. For this shape its
shared-prefix launch uses a 128-query tile for only eight decode query rows and
exposes a `(1, 1, 16)` grid. The resulting 16 CTAs cannot occupy all 48 SMs.
Each CTA then advances through a long 4K KV range, leaving insufficient
independent work to hide latency across the device.

The source heuristic chooses Cascade by comparing it with ordinary
FlashDecoding. It correctly predicts that sharing is useful, as demonstrated
by Cascade's 1.279x gain over ordinary Flash. It does not compare against a
decode-specific alternative that adds adaptive prefix splitting and
multi-query KV reuse. Thus the accurate conclusion is:

> vLLM native Cascade successfully optimizes logical shared-prefix work, but
> its current general FlashAttention tiling and scheduling are not fully
> optimized for this small-query, long-prefix decode shape.

This is an implementation and shape-specialization limitation, not a
fundamental limitation of the Cascade algorithm. More aggressive prefix
split-K, decode-specific query tiles, a CTA-wave-aware heuristic, or fusion of
suffix and merge processing could narrow the difference.

### What ForkAttention adds beyond Cascade

ForkAttention's advantage is not an additional claim that the prefix is
shared. Both optimized paths share it. Fork adds a GPU execution plan tailored
to synchronized decode fanout:

1. **Multi-query KV reuse inside a CTA.** A loaded shared KV tile is consumed
   by several branch queries before it is discarded, reducing on-chip traffic
   and control work.
2. **Adaptive parallelism along the prefix.** The production policy splits the
   4K prefix into four chunks. Together with private suffix work, the main
   `(12, 8, 1)` grid exposes 96 CTAs, approximately two CTA waves over 48 SMs.
3. **Decode-specific tiles.** Fork does not use a generic 128-query tile for an
   eight-row decode cohort; it selects a narrower multi-query configuration
   based on query count, GQA ratio, KV length, block size, and GPU capacity.
4. **A unified split plan.** Shared prefix chunks and branch-private suffixes
   produce split states for one gather, instead of separate prefix attention,
   suffix attention, and merge phases.

The counter differences support this explanation. Relative to Cascade, Fork
reduced L2 reads by 43.92%, Tensor-pipe warp instructions by 86.11%, and
FMA-pipe warp instructions by 66.15%, while DRAM reads stayed within 0.77%.
The 3.040x time reduction therefore comes from better tiling, less on-chip
work, and higher exposed parallelism—not from reading substantially fewer
bytes from external memory.

### Interpretation boundary

The result does not establish a universal 3x Fork advantage. Cascade may
select a different FlashAttention specialization or obtain better tile
utilization at larger query counts. Fork also has a narrower applicability
boundary: it requires compatible head geometry, block size and KV dtype,
physically shared resident pages, decode-only queries, and a sufficiently
aligned cohort. With too few queries, a short prefix, a large private suffix,
staggered arrivals, or weak physical sharing, Fork overhead can approach or
exceed its saved work and the runtime should fall back.

The next defensible experiment is a three-arm matrix over 4K-64K prefixes and
2-32 simultaneous queries, retaining the same correctness checks and all NCU
reports. That matrix can identify where Cascade changes specialization, where
Fork reaches adequate CTA waves, and where either shared-prefix method crosses
its break-even boundary.
