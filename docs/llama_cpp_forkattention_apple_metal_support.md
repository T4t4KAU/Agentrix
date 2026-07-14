# llama.cpp ForkAttention Apple Metal Support

## Scope

Agentrix now carries an Apple Metal implementation of the Qwen3 decode-only ForkAttention path in its `llama.cpp` submodule. The implementation is intended for Apple Silicon systems that run the llama.cpp Metal backend.

The Metal path complements the existing CUDA and MUSA ForkAttention implementations. It is selected automatically for supported `FLASH_ATTN_EXT` nodes when all ForkAttention constraints are satisfied. Unsupported attention shapes continue to use the regular Metal FlashAttention implementation.

## Source Version

The Agentrix `llama.cpp` submodule is updated to the ForkAttention-enabled source revision:

```text
3287e73c1 ggml-metal : add ForkAttention support
```

The Apple support touches only the Metal backend:

```text
ggml/src/ggml-metal/ggml-metal-device.cpp
ggml/src/ggml-metal/ggml-metal-device.h
ggml/src/ggml-metal/ggml-metal-impl.h
ggml/src/ggml-metal/ggml-metal-ops.cpp
ggml/src/ggml-metal/ggml-metal-ops.h
ggml/src/ggml-metal/ggml-metal.metal
```

## Supported Configuration

The Metal ForkAttention kernel is a specialized Qwen3 decode path. It is intentionally narrower than generic FlashAttention.

| Property | Metal ForkAttention support |
|---|---|
| Platform | Apple Silicon with llama.cpp Metal backend |
| Model family | Qwen3-style ForkAttention plan |
| Operation | Decode-only `FLASH_ATTN_EXT` with a ForkAttention plan |
| Query type | `F32` |
| Output type | `F32` |
| KV type | `F16` or `BF16` |
| Head dimension | 64 or 128 |
| Query batch | 2 to 8 one-token decode queries |
| GQA | Supported when query heads are divisible by KV heads |
| Sinks | Not supported on the fork kernel |
| ALiBi max bias | Not supported on the fork kernel |
| Logit softcap | Not supported on the fork kernel |

The fallback rule is important: if any condition is not met, the graph remains correct because llama.cpp keeps using the normal Metal FlashAttention path.

## Apple Platform Differences

Apple Silicon is not a discrete-GPU platform in the same sense as typical CUDA or MUSA deployments. The ForkAttention Metal path was designed around the following Apple-specific properties.

### Unified memory

Apple Silicon uses a unified memory architecture: CPU and GPU share the same physical memory pool. In llama.cpp this usually appears as `has unified memory = true` in the Metal backend logs.

Practical implications:

- There is no PCIe host-to-device copy path like on a discrete NVIDIA GPU.
- Model weights, KV cache buffers, and intermediate tensors are allocated through Metal buffer types that can map naturally onto unified memory.
- The main performance cost is not PCIe transfer. It is memory bandwidth, cache locality, kernel launch overhead, and how much repeated KV work can be skipped.
- ForkAttention is still useful because shared-prefix decode can avoid recomputing attention over the same common KV cells across multiple branch queries.

Unified memory does not mean memory is free. Large Qwen3 contexts still compete for the same system memory pool used by the CPU and GPU, so `--ctx-size`, `--parallel`, and KV cache type should still be sized carefully.

### Metal instead of CUDA-style device code

The Apple path is implemented in Metal Shading Language rather than CUDA C++:

- Kernels are declared in `ggml-metal.metal`.
- Host dispatch is wired through the llama.cpp Metal encoder helpers.
- Pipeline selection uses Metal function names such as `kernel_fork_attn_ext_f16_d128`.
- The kernel is compiled through Apple's Metal compiler, either ahead of time into a metallib or from source when `GGML_METAL_EMBED_LIBRARY=OFF`.

This means CUDA concepts such as warps, WMMA, CUDA shared memory syntax, CUDA streams, and compute capability do not directly apply. The implementation uses Metal threadgroups, threadgroup memory, and Metal pipeline objects instead.

### Threadgroup execution model

The fork kernel maps one `(query, head)` row to one Metal threadgroup. The threadgroup has one thread per head element, so the current specialized kernels use 64 or 128 threads.

The kernel uses threadgroup memory for:

- dot-product reduction across the head dimension;
- online softmax state;
- per-row normalization state.

This maps cleanly to the small decode batches targeted by ForkAttention. It is not intended to replace the broader Metal FlashAttention kernels used for prefill, wide batches, uncommon head sizes, or quantized KV types.

### BF16 availability is device-dependent

The source contains BF16 Metal kernel variants, but BF16 execution depends on the Apple GPU family and Metal feature support detected by llama.cpp. On validated hardware, the Metal log reported:

```text
has bfloat = true
```

When BF16 is not available, the backend should avoid selecting BF16 Metal kernels and fall back according to the normal llama.cpp backend support checks. FP16 KV remains the most portable Apple Metal path.

### Pipeline compilation and deployment

During development, `GGML_METAL_EMBED_LIBRARY=OFF` is useful because llama.cpp can load `ggml-metal.metal` from the build output. This makes it easy to iterate on the new ForkAttention kernel.

For packaged deployments, embedding or shipping a precompiled Metal library can reduce startup compilation overhead. If the source Metal library is used, the first execution of a new pipeline may log compilation of names such as:

```text
kernel_fork_attn_ext_f16_d128
kernel_fork_attn_ext_bf16_d64
```

This first-use compilation cost is separate from steady-state inference speed.

### Comparison with CUDA and MUSA

| Area | CUDA/MUSA ForkAttention | Apple Metal ForkAttention |
|---|---|---|
| Memory model | Usually discrete GPU memory | Unified CPU/GPU memory on Apple Silicon |
| Kernel language | CUDA-like device code | Metal Shading Language |
| Execution unit terminology | Warps / blocks | SIMD groups / threadgroups |
| Specialized matrix path | CUDA implementations may use NVIDIA-specific primitives | Current Metal path uses scalar/threadgroup reductions |
| Data movement bottleneck | PCIe transfer can matter on discrete GPUs | Unified-memory bandwidth and locality dominate |
| Pipeline setup | CUDA/MUSA binary kernels from the build | Metal pipeline objects, optionally compiled from source/metallib |
| Best workload | Shared-prefix Qwen3 decode | Shared-prefix Qwen3 decode |

The Metal implementation is therefore a correctness-first Apple backend path. It brings the ForkAttention feature to Apple Silicon while preserving llama.cpp's existing Metal FlashAttention fallback for general attention shapes.

## Build

From the Agentrix repository root:

```bash
cmake -S llama.cpp -B llama.cpp/build-metal \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=OFF \
  -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build llama.cpp/build-metal \
  --target llama-parallel test-backend-ops \
  -j"$(sysctl -n hw.ncpu)"
```

`GGML_METAL_EMBED_LIBRARY=OFF` is convenient during development because llama.cpp loads `ggml-metal.metal` from the build output and recompiles changed kernels without embedding a new metallib into the binary.

## Runtime

ForkAttention is exposed on multi-sequence examples such as `llama-parallel` and server-style workloads. A single `llama-cli` prompt is useful as a backend smoke test, but it normally does not create the shared-prefix multi-branch decode pattern that ForkAttention optimizes.

Example using a Qwen3 GGUF model:

```bash
./llama.cpp/build-metal/bin/llama-parallel \
  -m /path/to/Qwen3-0.6B-Q8_0.gguf \
  -ngl 99 \
  -fa on \
  --fork-attn \
  -c 2048 \
  -b 512 \
  -ub 8 \
  -np 4 \
  -ns 4 \
  -pps \
  -n 16 \
  --seed 42
```

The key flags are:

- `-fa on`: enables FlashAttention.
- `--fork-attn`: enables the experimental Qwen3 ForkAttention decode plan.
- `-pps`: shares the prompt prefix across parallel sequences.
- `-ub 8`: keeps the physical decode micro-batch inside the specialized kernel range.
- `-np 4 -ns 4`: creates parallel decode sequences that can reuse the shared prefix.

## Implementation Notes

The Metal implementation adds a dedicated templated kernel:

```text
kernel_fork_attn_ext_f16_d64
kernel_fork_attn_ext_f16_d128
kernel_fork_attn_ext_bf16_d64
kernel_fork_attn_ext_bf16_d128
```

At dispatch time, llama.cpp chooses the kernel name from the KV type and head dimension. The host-side predicate verifies tensor types, head sizes, query batch size, strides, GQA compatibility, plan layout, and the absence of sinks, ALiBi bias, and logit softcap.

The kernel reads the ForkAttention plan from `src[5]`. The plan layout is:

```text
header[8]
common_cells[n_kv]
private_lengths[n_queries]
private_cells[n_queries * n_kv]
```

Each Metal threadgroup computes one `(query, head)` output row. The implementation uses one thread per head element and a threadgroup reduction for the query-key dot product. Softmax is evaluated online across the common cells followed by the query-private cells, so the kernel does not materialize the full attention score matrix.

## Validation

The Apple path was validated on an Apple M2 Max Metal backend.

Shader compile checks:

```bash
xcrun -sdk macosx metal -O3 \
  -I llama.cpp/ggml/src \
  -I llama.cpp/ggml/src/ggml-metal \
  -c llama.cpp/ggml/src/ggml-metal/ggml-metal.metal \
  -o /tmp/ggml-metal-test.air

xcrun -sdk macosx metal -O3 -DGGML_METAL_HAS_BF16=1 \
  -I llama.cpp/ggml/src \
  -I llama.cpp/ggml/src/ggml-metal \
  -c llama.cpp/ggml/src/ggml-metal/ggml-metal.metal \
  -o /tmp/ggml-metal-test-bf16.air
```

Backend op test:

```bash
./llama.cpp/build-metal/bin/test-backend-ops \
  test -o FLASH_ATTN_EXT -b MTL0 -j 4
```

Expected result:

```text
4756/4756 tests passed
Backend MTL0: OK
```

The test log should show the dedicated fork kernels being compiled, for example:

```text
kernel_fork_attn_ext_f16_d128
kernel_fork_attn_ext_bf16_d64
```

End-to-end smoke test:

```bash
./llama.cpp/build-metal/bin/llama-parallel \
  -m /path/to/Qwen3-0.6B-Q8_0.gguf \
  -ngl 99 -fa on --fork-attn \
  -c 2048 -b 512 -ub 8 \
  -np 4 -ns 4 -pps \
  -n 16 --seed 42
```

The validated run completed four shared-prefix parallel requests and generated 64 total tokens without runtime errors.

## Operational Guidance

Use this path for Apple Silicon deployments that serve multiple Qwen3 requests with a shared prefix. It is most useful for server-style workloads, batched agents, and evaluation jobs where several continuations branch from the same context.

For unrelated model families, single-sequence decoding, prefill-heavy workloads, unsupported head dimensions, quantized KV types, or attention variants with sinks/bias/softcap, leave `--fork-attn` disabled or rely on the automatic fallback to the regular Metal FlashAttention path.
