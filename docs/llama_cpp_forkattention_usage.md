# llama.cpp ForkAttention Build and Runtime Guide

## Scope

Agentrix includes its ForkAttention-enabled `llama.cpp` fork as the `llama.cpp` Git submodule. The submodule tracks the `fork-attn` branch of:

```text
https://github.com/T4t4KAU/llama.cpp.git
```

The recorded Git submodule commit is the reproducible source version. The `branch = fork-attn` entry in `.gitmodules` is used only when explicitly requesting a remote submodule update.

This guide covers:

- checking out the submodule;
- building and starting the NVIDIA CUDA version;
- building and starting the Moore Threads MUSA version;
- enabling Qwen3 ForkAttention;
- verifying that the specialized path is selected;
- understanding the MUSA-specific adaptation and fallback rules.

For Huawei Cloud EulerOS and CentOS package/toolchain differences, see [llama_cpp_forkattention_hce_centos_adaptation.md](llama_cpp_forkattention_hce_centos_adaptation.md).

## Initialize the Submodule

For a new Agentrix clone:

```bash
git submodule update --init --recursive
```

Confirm the recorded source revision:

```bash
git submodule status llama.cpp
git -C llama.cpp log -1 --oneline
```

Do not use `git submodule update --remote` in normal deployment scripts. That command follows the latest remote `fork-attn` branch and changes the reproducible Git link recorded by Agentrix.

## Supported ForkAttention Configuration

ForkAttention is an opt-in decode path. Unsupported workloads remain correct by falling back to the regular llama.cpp FlashAttention implementation.

| Property | CUDA | MUSA |
|---|---|---|
| Valid GPU generation | NVIDIA Turing or newer | Moore Threads QY2 or newer |
| Validated example | Tesla T4 (`sm_75`) | MTT S4000 (`mp_22`) |
| KV type | FP16 on Turing; BF16 on Ampere or newer | FP16 |
| Model | Qwen3 | Qwen3 |
| Decode width | 2 to 8 one-token sequences | 2 to 8 one-token sequences |
| Head dimensions | 64 or 128 | 64 or 128 |
| Attention | Causal decode | Causal decode |
| Shared-prefix requirement | A sufficiently valuable shared prefix | A sufficiently valuable shared prefix |

Prefill, single-sequence decoding, unsupported KV layouts, unsupported model families, and small shared prefixes use the native path.

## NVIDIA CUDA Build

### Environment

The following example targets a Tesla T4 with CUDA installed under `/usr/local/cuda`:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

On CentOS 7, initialize the newer compiler and user-installed CMake first:

```bash
export LC_ALL=C
export LANG=C
source /opt/rh/devtoolset-10/enable
export PATH=/root/.local/bin:/usr/local/cuda/bin:$PATH
```

### Configure and build

Run from the Agentrix repository root:

```bash
cmake -S llama.cpp -B llama.cpp/build-cuda-t4 -G Ninja \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DGGML_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build llama.cpp/build-cuda-t4 \
  --target llama-server llama-cli llama-parallel test-backend-ops \
  -j"$(nproc)"
```

Use the compute capability of the target NVIDIA GPU when it is not a T4. The ForkAttention FP16 implementation requires Turing or newer. BF16 requires Ampere or newer.

## Moore Threads MUSA Build

### Prerequisites

Install a compatible MUSA SDK and GPU driver before building. The CMake backend searches for the SDK in this order:

1. `$MUSA_PATH`, when set;
2. `/opt/musa`;
3. `/usr/local/musa`.

When the SDK is installed under `/usr/local/musa`, initialize it with:

```bash
export MUSA_PATH=/usr/local/musa
export PATH="$MUSA_PATH/bin:$PATH"
```

If the dynamic loader does not already know the MUSA SDK library directory, add the directory used by the installed SDK, for example:

```bash
export LD_LIBRARY_PATH="$MUSA_PATH/lib:$MUSA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Check that the toolkit and GPU are visible before configuring llama.cpp:

```bash
"$MUSA_PATH/bin/clang" --version
mthreads-gmi
```

### Configure and build for MTT S4000

The MTT S4000 uses the QY2 target, represented by MUSA architecture 22:

```bash
cmake -S llama.cpp -B llama.cpp/build-musa -G Ninja \
  -DGGML_MUSA=ON \
  -DGGML_CUDA_FA=ON \
  -DMUSA_ARCHITECTURES=22 \
  -DGGML_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build llama.cpp/build-musa \
  --target llama-server llama-cli llama-parallel test-backend-ops \
  -j"$(nproc)"
```

Specifying only architecture 22 avoids compiling the default multi-architecture set (`21;22;31`) and reduces build time on an S4000 deployment host.

### What was adapted for MUSA

The MUSA backend reuses the ggml CUDA backend structure through llama.cpp's MUSA compatibility layer, but ForkAttention has a distinct MUSA partial-attention kernel. The implementation accounts for the following platform differences:

- MUSA compilation uses the SDK `clang` and `clang++` with `-x musa -mtgpu`.
- Architecture 22 is emitted as the `mp_22` GPU target.
- The QY2 path uses the MUSA execution and FP16 primitives instead of NVIDIA WMMA instructions.
- MUSA runtime APIs are mapped through `ggml-cuda/vendors/musa.h`.
- The specialized MUSA ForkAttention path is restricted to QY2 or newer and FP16 KV data.
- BF16 ForkAttention is not enabled on the current MUSA path.
- Unsupported batches are sent to the existing MUSA FlashAttention/native path instead of failing inference.

The command-line interface is intentionally identical on CUDA and MUSA. No application-level backend flag is required after building the correct backend.

## Start llama-server

ForkAttention is most useful in `llama-server`, where concurrent requests can share a long prompt prefix. Use a Qwen3 GGUF model and enable both FlashAttention and ForkAttention.

### CUDA

```bash
./llama.cpp/build-cuda-t4/bin/llama-server \
  --model /path/to/qwen3.gguf \
  --n-gpu-layers 99 \
  --flash-attn on \
  --fork-attn \
  --alias qwen3 \
  --parallel 4 \
  --ctx-size 8192 \
  --host 0.0.0.0 \
  --port 8080
```

### MUSA

```bash
./llama.cpp/build-musa/bin/llama-server \
  --model /path/to/qwen3.gguf \
  --n-gpu-layers 99 \
  --flash-attn on \
  --fork-attn \
  --alias qwen3 \
  --parallel 4 \
  --ctx-size 8192 \
  --host 0.0.0.0 \
  --port 8080
```

The server exposes an OpenAI-compatible endpoint. A basic request is:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Explain prefix sharing briefly."}],
    "temperature": 0
  }'
```

A single request does not exercise the multi-branch kernel. Use concurrent requests with the same long system/user prefix and different suffixes to trigger ForkAttention.

## Start llama-cli

`llama-cli` is useful for a basic model and backend smoke test:

### CUDA

```bash
./llama.cpp/build-cuda-t4/bin/llama-cli \
  --model /path/to/qwen3.gguf \
  --n-gpu-layers 99 \
  --flash-attn on \
  --prompt 'Write one sentence about shared KV caches.' \
  --n-predict 32
```

### MUSA

```bash
./llama.cpp/build-musa/bin/llama-cli \
  --model /path/to/qwen3.gguf \
  --n-gpu-layers 99 \
  --flash-attn on \
  --prompt 'Write one sentence about shared KV caches.' \
  --n-predict 32
```

`llama-cli` normally decodes one sequence, so it validates model loading and GPU execution but does not demonstrate the ForkAttention multi-sequence path.

## Shared-prefix Functional Test

`llama-parallel` provides a deterministic way to exercise four branches that share a prompt prefix. This is a functional smoke test; increasing token counts turns it into a benchmark and is not required for deployment validation.

### CUDA

```bash
./llama.cpp/build-cuda-t4/bin/llama-parallel \
  --model /path/to/qwen3.gguf \
  --n-gpu-layers 99 \
  --flash-attn on \
  --fork-attn \
  --parallel 4 \
  --sequences 4 \
  -pps \
  --n-predict 4 \
  --temp 0 \
  --seed 123 \
  --ctx-size 4096 \
  --verbose
```

### MUSA

```bash
./llama.cpp/build-musa/bin/llama-parallel \
  --model /path/to/qwen3.gguf \
  --n-gpu-layers 99 \
  --flash-attn on \
  --fork-attn \
  --parallel 4 \
  --sequences 4 \
  -pps \
  --n-predict 4 \
  --temp 0 \
  --seed 123 \
  --ctx-size 4096 \
  --verbose
```

The short aliases used by llama.cpp are `-np 4 -ns 4 -pps -n 4 -s 123 -c 4096`.

## Backend Correctness Test

Run the focused backend operation test after a new compiler, SDK, driver, or GPU architecture change.

### CUDA backend name

```bash
./llama.cpp/build-cuda-t4/bin/test-backend-ops test \
  -b CUDA0 \
  -o FLASH_ATTN_EXT \
  -p 'fork=1' \
  -j 4
```

### MUSA backend name

```bash
./llama.cpp/build-musa/bin/test-backend-ops test \
  -b MUSA0 \
  -o FLASH_ATTN_EXT \
  -p 'fork=1' \
  -j 4
```

The test compares the GPU result with the CPU FlashAttention reference while using an exact physical KV plan.

## Runtime Verification

Run with `--verbose` and check for both configuration and planner messages:

```text
fork_attn = true
fork-attn: plans=..., queries=..., common=..., private_max=..., saved_kv_reads=...
```

Interpretation:

- `fork_attn = true` confirms that the option reached the llama context.
- A `fork-attn: plans=` line confirms that at least one decode batch used the shared-prefix plan.
- No planner line can be normal for prefill, one active sequence, a short shared prefix, or another unsupported layout; those cases intentionally fall back.
- `cache miss 0` in `llama-parallel` confirms that the short smoke test did not lose sequence cache state.

On MUSA, also verify that the startup log reports a MUSA device rather than CPU-only execution. On CUDA, verify that it reports the intended NVIDIA device and architecture.

## Updating the llama.cpp Submodule

Update only when intentionally adopting a newer tested `fork-attn` commit:

```bash
git -C llama.cpp fetch origin fork-attn
git -C llama.cpp checkout <reviewed-commit>
git add llama.cpp
```

Review and validate the new submodule commit on each required backend before committing the Agentrix Git-link change. A submodule checkout is normally detached; this is expected for reproducible parent-repository builds.
