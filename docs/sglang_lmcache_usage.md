# SGLang LMCache Runtime Guide

## Scope

Agentrix can run its OpenAI-compatible benchmark against SGLang with LMCache
enabled in SGLang's multi-process mode. In this mode a standalone `lmcache`
daemon owns the KV cache store, and SGLang connects to it through `mp_host` and
`mp_port` from a YAML config.

This guide covers:

- installing the LMCache runtime pieces in the SGLang environment;
- launching the reusable Agentrix SGLang+LMCache benchmark wrapper;
- checking that SGLang registered KV cache with the LMCache daemon;
- current limitations when combining LMCache with the ForkAttention backend.

The commands below are written for the Agentrix repository root:

```bash
export AGENTRIX_ROOT=/path/to/Agentrix
export MODEL_PATH=/path/to/Qwen3-8B
export SGLANG_PYTHON="${AGENTRIX_ROOT}/sglang/.venv/bin/python"
```

## Install Runtime Dependencies

Install LMCache into the same Python environment that runs SGLang:

```bash
cd "${AGENTRIX_ROOT}"

NO_GPU_EXT=1 MAX_JOBS=2 uv pip install \
  --python "${SGLANG_PYTHON}" \
  --no-build-isolation \
  -e "${AGENTRIX_ROOT}/LMCache"
```

The SGLang LMCache MP path also imports CuPy when registering the KV cache:

```bash
uv pip install \
  --python "${SGLANG_PYTHON}" \
  cupy-cuda13x
```

Some Conda-based Python environments may load an older `libstdc++` before the
system library. The benchmark wrapper automatically preloads
`/usr/lib/x86_64-linux-gnu/libstdc++.so.6` when it exists. Override that behavior
with:

```bash
export LMCACHE_LD_PRELOAD=/path/to/libstdc++.so.6
```

## Run the Benchmark Wrapper

Use the wrapper script so that LMCache startup, YAML generation, SGLang launch,
benchmark execution, log checks, and cleanup happen in one command:

```bash
cd "${AGENTRIX_ROOT}"

MODEL_PATH="${MODEL_PATH}" \
SERVED_MODEL_NAME=qwen3-8b-sglang-lmcache \
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
OUTPUT_DIR=results/sglang_lmcache_smoke \
ENABLE_TELEMETRY=0 \
ENABLE_METRICS=0 \
bash benchmark/scripts/run_sglang_lmcache_benchmark.sh
```

The wrapper writes:

```text
benchmark/results/sglang_lmcache_smoke/lmcache_mp.yaml
benchmark/results/sglang_lmcache_smoke/lmcache_server.log
benchmark/results/sglang_lmcache_smoke/sglang/
```

Useful knobs:

| Variable | Default | Meaning |
|---|---:|---|
| `LMCACHE_HOST` | `127.0.0.1` | LMCache MP daemon host |
| `LMCACHE_PORT` | `5556` | LMCache MP daemon port |
| `LMCACHE_L1_SIZE_GB` | `1` | LMCache L1 cache size |
| `LMCACHE_EVICTION_POLICY` | `LRU` | LMCache eviction policy |
| `SGLANG_LMCACHE_ATTENTION_BACKEND` | `triton` | SGLang attention backend used for the LMCache run |
| `LMCACHE_CONFIG_FILE` | under `OUTPUT_DIR` | Generated MP YAML path |
| `LMCACHE_LOG_FILE` | under `OUTPUT_DIR` | LMCache daemon log path |

## Verification

The wrapper fails if either log contains a traceback, module import failure, or
KV registration error. For manual inspection:

```bash
rg -n "enable_lmcache=True|lmcache_config_file|Registered KV cache|Stored [0-9]+ tokens|Traceback|ModuleNotFoundError" \
  benchmark/results/sglang_lmcache_smoke/lmcache_server.log \
  benchmark/results/sglang_lmcache_smoke/sglang/sglang_server.log
```

A valid smoke run should show:

```text
enable_lmcache=True
Registered KV cache for GPU ID ... with 36 layers
Stored 256 tokens in ... seconds
```

The benchmark CSV is written to:

```text
benchmark/results/sglang_lmcache_smoke/sglang/benchmark_results.csv
```

## ForkAttention Combination Status

The validated LMCache smoke path uses SGLang's default KV layout and
`SGLANG_LMCACHE_ATTENTION_BACKEND=triton`. The current ForkAttention backend
requires:

```bash
--attention-backend fork_attn
--page-size 16
--enable-page-major-kv-layout
```

SGLang's current LMCache integration registers page-size-1 KV blocks in the
tested path. Combining LMCache and ForkAttention therefore needs a separate
layout compatibility pass before treating the two features as validated
together.
