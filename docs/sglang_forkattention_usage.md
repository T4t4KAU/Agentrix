# SGLang ForkAttention Build and Runtime Guide

## Scope

Agentrix includes SGLang as the `sglang` Git submodule. The submodule fork is
[oyyyyy61/sglang.git](https://github.com/oyyyyy61/sglang.git). The
ForkAttention integration lives in the submodule's `fork-attn` branch and adds
an opt-in SGLang attention backend named:

```text
fork_attn
```

This guide covers:

- checking out the SGLang submodule branch;
- installing the SGLang Python environment with `uv`;
- building `sgl-kernel` for the local NVIDIA RTX 4090 only;
- enabling ForkAttention at SGLang runtime;
- running Agentrix benchmarks against the OpenAI-compatible SGLang server;
- verifying that the ForkAttention decode path is selected.

The commands below are written for the Agentrix repository root. Set these
variables first and adjust them for the target deployment host:

```bash
export AGENTRIX_ROOT=/path/to/Agentrix
export MODEL_PATH=/path/to/Qwen3-8B
export SGLANG_PYTHON="${AGENTRIX_ROOT}/sglang/.venv/bin/python"
```

## Initialize the Submodule

For a new Agentrix clone:

```bash
git submodule update --init --recursive sglang
```

Check the submodule state:

```bash
git -C sglang status --short
git -C sglang branch --show-current
git -C sglang log -1 --oneline
```

The ForkAttention development branch should be:

```text
fork-attn
```

If the submodule is detached or on another branch, switch it explicitly:

```bash
git -C sglang fetch origin fork-attn
git -C sglang switch fork-attn
```

Do not use `git submodule update --remote` in normal deployment scripts. That
command follows the latest remote branch and changes the reproducible Git link
recorded by Agentrix.

## Supported ForkAttention Configuration

ForkAttention is an opt-in decode path. Unsupported SGLang requests remain
correct by falling back to the existing Triton attention path.

| Property | Current SGLang ForkAttention path |
|---|---|
| Runtime flag | `--attention-backend fork_attn` |
| GPU backend | NVIDIA CUDA |
| Validated GPU | RTX 4090, compute capability 8.9, `sm_89` |
| Validated model | Qwen3-8B |
| KV cache layout | SGLang page-major KV layout |
| Required SGLang flags | `--page-size 16 --enable-page-major-kv-layout` |
| Dtype | FP16 or BF16 tensors accepted by the backend guard |
| Attention type | Decoder self-attention |
| Head dimensions | 64 or 128 |
| Unsupported cases | MLA, DCP, cross attention, sliding window, attention sinks, logit soft cap, quantized KV descale, incompatible KV cache layout |

The current 4090 build intentionally emits only `sm_89` code to reduce build
time and host memory pressure. On a different GPU, use the target architecture
for that GPU and validate the resulting extension before deployment.

## Environment

The tested SGLang environment uses a Python virtual environment under:

```text
sglang/.venv
```

Install or refresh the editable SGLang environment with `uv` according to the
SGLang project requirements. When rebuilding only `sgl-kernel`, the important
part is that the target Python is:

```text
${AGENTRIX_ROOT}/sglang/.venv/bin/python
```

The local Qwen3-8B model used for validation is:

```text
${MODEL_PATH}
```

## Build `sgl-kernel` for RTX 4090

Build the SGLang kernel extension from the Agentrix repository root:

```bash
cd "${AGENTRIX_ROOT}"

CMAKE_BUILD_PARALLEL_LEVEL=1 uv pip install \
  --python "${SGLANG_PYTHON}" \
  --no-build-isolation \
  --force-reinstall \
  --config-settings=cmake.define.SGL_KERNEL_COMPILE_THREADS=1 \
  --config-settings=cmake.define.SGL_KERNEL_CUDA_ARCH_LIST=89 \
  -e "${AGENTRIX_ROOT}/sglang/sgl-kernel"
```

The expected CMake configuration message is:

```text
Building only the SM89 common_ops variant for Ada/RTX 4090
```

The installed extension should be:

```text
sglang/.venv/lib/python3.11/site-packages/sgl_kernel/sm89/common_ops.abi3.so
```

If a previous build was interrupted or used a broader architecture set, remove
temporary build directories before rebuilding:

```bash
rm -rf /tmp/<sgl-kernel-build-temp-dir>
rm -rf "${UV_CACHE_DIR:-${HOME}/.cache/uv}/builds-v0"/.tmp*
```

Only remove known temporary build directories. Do not delete the SGLang source
tree or the virtual environment unless intentionally recreating the environment.

## Kernel Registration Check

Run this check after every `sgl-kernel` rebuild:

```bash
cd "${AGENTRIX_ROOT}"

PYTHONPATH="${AGENTRIX_ROOT}/sglang/python" \
"${SGLANG_PYTHON}" - <<'PY'
import torch
import sgl_kernel
from sglang.srt.layers.attention.fork_attn_backend import _fork_attention_op_available

print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("capability:", torch.cuda.get_device_capability())
print("common_ops:", getattr(sgl_kernel.common_ops, "__file__", None))
print("wrapper_export:", hasattr(sgl_kernel, "fork_attention"))
print("op_available:", _fork_attention_op_available())
print("op:", torch.ops.sgl_kernel.fork_attention)
PY
```

Validated output on the RTX 4090 host:

```text
cuda_available: True
capability: (8, 9)
common_ops: .../site-packages/sgl_kernel/sm89/common_ops.abi3.so
wrapper_export: True
op_available: True
op: sgl_kernel.fork_attention
```

If `op_available` is `False`, the SGLang server will log a fallback reason and
use Triton decode attention.

## Start SGLang with ForkAttention

A direct server launch looks like:

```bash
cd "${AGENTRIX_ROOT}"

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="${AGENTRIX_ROOT}/sglang/python" \
"${SGLANG_PYTHON}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name qwen3-8b-fork-attn \
  --host 127.0.0.1 \
  --port 9000 \
  --dtype float16 \
  --tp-size 1 \
  --context-length 4096 \
  --mem-fraction-static 0.80 \
  --max-running-requests 4 \
  --attention-backend fork_attn \
  --page-size 16 \
  --enable-page-major-kv-layout \
  --disable-cuda-graph
```

`--disable-cuda-graph` is useful during bring-up because it keeps the execution
path easier to inspect. SGLang currently reports it as deprecated; the newer
equivalent is to disable CUDA graph through the decode and prefill CUDA graph
backend flags.

The server exposes an OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sglang-local' \
  -d '{
    "model": "qwen3-8b-fork-attn",
    "messages": [{"role": "user", "content": "Explain shared KV caches briefly."}],
    "temperature": 0,
    "max_tokens": 32
  }'
```

## Agentrix Benchmark Smoke Test

Use `benchmark/scripts/run_sglang_benchmark.sh` to start SGLang, run the
OpenAI-compatible Agentrix benchmark, collect logs, and stop the server.

Minimal one-case smoke:

```bash
cd "${AGENTRIX_ROOT}"

MODEL_PATH="${MODEL_PATH}" \
SERVED_MODEL_NAME=qwen3-8b-fork-attn-smoke \
DATASET=swebench \
CASE_COUNT=1 \
SAMPLE_COUNT=1 \
PREFIX_TOKENS=128 \
BRANCHES=1 \
OUTPUT_TOKENS=8 \
COMMON_ANALYSIS_TOKENS=8 \
CONTEXT_LENGTH=4096 \
MAX_RUNNING_REQUESTS=4 \
MEM_FRACTION_STATIC=0.80 \
OUTPUT_DIR=results/sglang_fork_attn_smoke_qwen3_8b \
SGLANG_EXTRA_ARGS='--attention-backend fork_attn --page-size 16 --enable-page-major-kv-layout --disable-cuda-graph' \
ENABLE_TELEMETRY=0 \
ENABLE_METRICS=0 \
STARTUP_TIMEOUT=300 \
bash benchmark/scripts/run_sglang_benchmark.sh
```

Multi-case shared-prefix benchmark smoke:

```bash
cd "${AGENTRIX_ROOT}"

MODEL_PATH="${MODEL_PATH}" \
SERVED_MODEL_NAME=qwen3-8b-fork-attn-doccheck \
DATASET=swebench \
CASE_COUNT=2 \
SAMPLE_COUNT=2 \
PREFIX_TOKENS=256 \
BRANCHES=2 \
OUTPUT_TOKENS=16 \
COMMON_ANALYSIS_TOKENS=16 \
CONTEXT_LENGTH=4096 \
MAX_RUNNING_REQUESTS=4 \
MEM_FRACTION_STATIC=0.80 \
OUTPUT_DIR=results/sglang_fork_attn_doccheck_c2_b2_p256_o16 \
SGLANG_EXTRA_ARGS='--attention-backend fork_attn --page-size 16 --enable-page-major-kv-layout --disable-cuda-graph' \
ENABLE_TELEMETRY=0 \
ENABLE_METRICS=0 \
STARTUP_TIMEOUT=300 \
bash benchmark/scripts/run_sglang_benchmark.sh
```

Validated output directory:

```text
benchmark/results/sglang_fork_attn_doccheck_c2_b2_p256_o16/sglang
```

The validated CSV contained one benchmark case with four branch requests:

```text
case_id=api_forest_c2_p291_b2_g1_lognormal_s0
branch_count=4
branch_total_output_tokens=64
branch_total_input_tokens=1786
kv_tokens_saved=873
kv_reduction_percent=48.880179171332585
```

## Runtime Verification

Check the SGLang server log after a run:

```bash
rg -n \
  "ForkAttention decode path is enabled|ForkAttention decode falls back|Traceback|CUDA error|RuntimeError|ERROR|Exception" \
  benchmark/results/sglang_fork_attn_doccheck_c2_b2_p256_o16/sglang/sglang_server.log
```

Expected success signal:

```text
ForkAttention decode path is enabled.
```

No `ForkAttention decode falls back` line should appear for the validated
Qwen3-8B 4090 configuration. A fallback line means the request still completed,
but the decode path used Triton attention. The fallback reason printed in the
log identifies the unmet condition.

Common fallback reasons include:

- `torch.ops.sgl_kernel.fork_attention is not registered`;
- `MLA attention is not supported`;
- `DCP attention is not supported`;
- `sliding window is not supported`;
- `logit soft cap is not supported`;
- `quantized KV descale is not supported`;
- `KV cache is not a 4D paged layout`;
- `KV cache page dimension does not match page_size`;
- `head dim ... is not supported`.

Also verify the launch arguments in the server log:

```text
attention_backend='fork_attn'
enable_page_major_kv_layout=True
page_size=16
```

## Triton Baseline Comparison

For performance comparisons, run the same workload with `triton` and
`fork_attn`.

Triton baseline:

```bash
MODEL_PATH="${MODEL_PATH}" \
SERVED_MODEL_NAME=qwen3-8b-triton-baseline \
DATASET=swebench \
CASE_COUNT=4 \
SAMPLE_COUNT=4 \
PREFIX_TOKENS=512 \
BRANCHES=2 \
OUTPUT_TOKENS=32 \
COMMON_ANALYSIS_TOKENS=32 \
CONTEXT_LENGTH=8192 \
MAX_RUNNING_REQUESTS=8 \
MEM_FRACTION_STATIC=0.80 \
OUTPUT_DIR=results/sglang_triton_c4_b2_p512_o32 \
SGLANG_EXTRA_ARGS='--attention-backend triton --page-size 16 --enable-page-major-kv-layout --disable-cuda-graph' \
ENABLE_TELEMETRY=0 \
ENABLE_METRICS=0 \
STARTUP_TIMEOUT=300 \
bash benchmark/scripts/run_sglang_benchmark.sh
```

ForkAttention:

```bash
MODEL_PATH="${MODEL_PATH}" \
SERVED_MODEL_NAME=qwen3-8b-fork-attn \
DATASET=swebench \
CASE_COUNT=4 \
SAMPLE_COUNT=4 \
PREFIX_TOKENS=512 \
BRANCHES=2 \
OUTPUT_TOKENS=32 \
COMMON_ANALYSIS_TOKENS=32 \
CONTEXT_LENGTH=8192 \
MAX_RUNNING_REQUESTS=8 \
MEM_FRACTION_STATIC=0.80 \
OUTPUT_DIR=results/sglang_fork_attn_c4_b2_p512_o32 \
SGLANG_EXTRA_ARGS='--attention-backend fork_attn --page-size 16 --enable-page-major-kv-layout --disable-cuda-graph' \
ENABLE_TELEMETRY=0 \
ENABLE_METRICS=0 \
STARTUP_TIMEOUT=300 \
bash benchmark/scripts/run_sglang_benchmark.sh
```

Compare:

- `benchmark_results.csv`;
- `summary.md`;
- server log throughput lines;
- fallback lines in `sglang_server.log`;
- telemetry output if `ENABLE_TELEMETRY=1`.

Use deterministic decoding settings when comparing output equality rather than
throughput. For throughput, keep the same model, dtype, context length, case
count, branch count, output tokens, and CUDA graph settings across both runs.

## Current Validation Record

The following validation was run on 2026-07-15 on the local RTX 4090 host:

```text
Model: ${MODEL_PATH}
GPU: RTX 4090, compute capability (8, 9)
sgl-kernel extension: sgl_kernel/sm89/common_ops.abi3.so
SGLang attention backend: fork_attn
Page size: 16
Page-major KV layout: enabled
CUDA graph: disabled for bring-up validation
```

Validation commands completed successfully:

- kernel registration check;
- one-case smoke benchmark;
- two-case, two-branch benchmark smoke.

The multi-branch run logged:

```text
ForkAttention decode path is enabled.
```

No `ForkAttention decode falls back`, `Traceback`, `CUDA error`, or
`RuntimeError` line was found in the validated server log.

## Updating the SGLang Submodule

Update only when intentionally adopting a newer tested `fork-attn` commit:

```bash
git -C sglang fetch origin fork-attn
git -C sglang checkout <reviewed-commit>
git add sglang
```

After updating the submodule, rebuild `sgl-kernel`, rerun the registration
check, and rerun at least the one-case and multi-branch smoke tests before
committing the Agentrix Git-link change.
