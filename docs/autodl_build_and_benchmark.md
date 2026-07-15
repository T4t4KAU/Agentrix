# Building and Benchmarking Agentrix on AutoDL

This guide reproduces the CUDA 12.8 build used on an AutoDL host with two
NVIDIA GeForce RTX 5090 GPUs. It keeps all environments, caches, source
dependencies, build outputs, models, and benchmark results under
`/root/autodl-tmp`.

## Validated Host

- Ubuntu 22.04
- NVIDIA driver 595.71.05
- CUDA toolkit 12.8.93 at `/usr/local/cuda-12.8`
- 2 x RTX 5090, 32 GiB each, compute capability 12.0
- 208 logical CPUs and 754 GiB RAM
- Python 3.12 from `/root/miniconda3`
- Qwen3-8B at `/root/autodl-tmp/models/Qwen3-8B`

The host can access PyPI mirrors but cannot reliably access GitHub. Source
repositories and CMake `FetchContent` dependencies therefore need to be
transferred from a development machine.

## Synchronize the Repository Without GitHub

On the development machine, create and transfer a Git bundle:

```bash
cd /path/to/Agentrix
git bundle create /tmp/agentrix-main.bundle main
scp -P <autodl-ssh-port> /tmp/agentrix-main.bundle \
  root@<autodl-ssh-host>:/root/autodl-tmp/
```

On the AutoDL host:

```bash
cd /root/autodl-tmp/Agentrix
git config --global --add safe.directory "$PWD"
git config --global --add safe.directory "$PWD/vllm"
git config --global --add safe.directory "$PWD/LMCache"
git fetch /root/autodl-tmp/agentrix-main.bundle \
  main:refs/remotes/dev/main
git checkout -B main refs/remotes/dev/main
git submodule update --init --recursive
git submodule status
```

The validated revisions were Agentrix `b98cf22`, vLLM `173016d22`, and
LMCache `b84945ca`.

## Configure uv and the Mirror

Install uv with the base Python and place its cache on the data disk. The
PyTorch CUDA wheels still come from the PyTorch CUDA 12.8 index; all regular
Python packages use the Tsinghua mirror.

```bash
export PATH=/root/miniconda3/bin:$PATH
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple uv

export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export UV_HTTP_TIMEOUT=300
mkdir -p "$UV_CACHE_DIR"
```

Do not leave the uv cache under `/root/.cache` on this image. That path is on
the container overlay and extracting large PyTorch and Triton wheels there is
much slower than using `/root/autodl-tmp`.

## Install the CUDA 12.8 Python Environment

```bash
cd /root/autodl-tmp/Agentrix/vllm
uv venv --python /root/miniconda3/bin/python .venv
uv pip install --python .venv/bin/python --torch-backend=cu128 \
  -r requirements/build/cuda.txt \
  -r requirements/cuda.txt
```

Verify that the environment did not resolve a CUDA 13 wheel:

```bash
.venv/bin/python - <<'PY'
import torch

print(torch.__version__, torch.version.cuda)
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index),
          torch.cuda.get_device_capability(index))
PY
```

The validated output reports `torch 2.11.0+cu128`, CUDA `12.8`, and capability
`(12, 0)` for both GPUs.

## Transfer CMake Source Dependencies

vLLM fetches several projects from GitHub during CMake configuration. Reuse
the sources from an existing development build and transfer them without Git
history. The source directories used by the validated build were:

```text
cutlass v4.4.2
DeepGEMM
FlashMLA
MSA/fmha_sm100
QUTLASS
vLLM FlashAttention
Triton kernels v3.5.1
```

For example, transfer a prepared dependency directory with:

```bash
tar -C /path/to/prepared-vllm-deps --exclude='*/.git' -cf - . \
  | pigz -p 16 -3 \
  | ssh -p <autodl-ssh-port> root@<autodl-ssh-host> \
      'mkdir -p /root/autodl-tmp/deps/vllm && \
       gzip -dc | tar -C /root/autodl-tmp/deps/vllm -xf -'
```

The local Triton override must point to the Python package directory, not the
Triton repository root:

```text
/root/autodl-tmp/deps/vllm/triton_kernels-src/python/triton_kernels/triton_kernels
```

Pointing it at the repository root incorrectly starts an LLVM/MLIR compiler
build.

## Configure and Build vLLM

Set the CUDA toolkit explicitly because `nvcc` is not in the default `PATH`:

```bash
cd /root/autodl-tmp/Agentrix/vllm
export PATH="$PWD/.venv/bin:/usr/local/cuda-12.8/bin:/root/miniconda3/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=12.0
export VLLM_TARGET_DEVICE=cuda
export NVCC_THREADS=4

export VLLM_CUTLASS_SRC_DIR=/root/autodl-tmp/deps/cutlass-v4.4.2
export DEEPGEMM_SRC_DIR=/root/autodl-tmp/deps/vllm/deepgemm-src
export FLASH_MLA_SRC_DIR=/root/autodl-tmp/deps/vllm/flashmla-src
export FMHA_SM100_SRC_DIR=/root/autodl-tmp/deps/vllm/fmha_sm100-src
export QUTLASS_SRC_DIR=/root/autodl-tmp/deps/vllm/qutlass-src
export VLLM_FLASH_ATTN_SRC_DIR=/root/autodl-tmp/deps/vllm/vllm-flash-attn-src
export TRITON_KERNELS_SRC_DIR=/root/autodl-tmp/deps/vllm/triton_kernels-src/python/triton_kernels/triton_kernels

.venv/bin/python tools/generate_cmake_presets.py --force-overwrite
sed -i 's#cmake-build-release#cmake-build-cu128#g' CMakeUserPresets.json
.venv/bin/cmake --preset release -DNVCC_THREADS=4
.venv/bin/cmake --build --preset release --target install --parallel 48
```

The persistent incremental build directory is `vllm/cmake-build-cu128`.
Successful configuration must include both of these lines:

```text
CUDA target architectures: 12.0
Building experimental ForkAttention for archs: 12.0
```

Generate Python package metadata without rebuilding the CUDA extensions:

```bash
.venv/bin/python setup.py egg_info
```

Create the CLI wrapper used by the benchmark scripts:

```bash
cat >.venv/bin/vllm <<'EOF'
#!/usr/bin/env bash
export PYTHONPATH=/root/autodl-tmp/Agentrix/vllm${PYTHONPATH:+:$PYTHONPATH}
exec /root/autodl-tmp/Agentrix/vllm/.venv/bin/python \
  -m vllm.entrypoints.cli.main "$@"
EOF
chmod +x .venv/bin/vllm
.venv/bin/vllm --version
```

Create a reusable environment file outside the Git checkout:

```bash
cat >/root/autodl-tmp/agentrix-cu128-env.sh <<'EOF'
export AGENTRIX_ROOT=/root/autodl-tmp/Agentrix
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$AGENTRIX_ROOT/vllm/.venv/bin:$CUDA_HOME/bin:/root/miniconda3/bin:$PATH
export PYTHONPATH=$AGENTRIX_ROOT/vllm:$AGENTRIX_ROOT/benchmark/src${PYTHONPATH:+:$PYTHONPATH}
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export TORCH_CUDA_ARCH_LIST=12.0
EOF
```

## Install and Validate the Benchmark Environment

```bash
cd /root/autodl-tmp/Agentrix/benchmark
uv venv --python /root/miniconda3/bin/python .venv
uv pip install --python .venv/bin/python -e ".[data,test]"
.venv/bin/python -m pytest tests -q
```

The validated benchmark suite has 23 passing tests. For focused GPU coverage,
install `pytest` and `tblib` in the vLLM environment and run:

```bash
cd /root/autodl-tmp/Agentrix/vllm
uv pip install --python .venv/bin/python pytest tblib
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/kernels/test_fork_attention.py -q \
  -k 'masks_partial_suffix_block or interleaved_kv_cache_suffix_page_stride'
```

The checked-in `requirements/test/cuda.txt` was generated for cu130. Do not
install that lock file unchanged into this cu128 environment. Install the
focused test dependencies as above or regenerate the test lock for cu128.

## Run the Three-Way DP Benchmark

The following command runs FlashAttention with ordinary internal DP,
ForkAttention with ordinary internal DP, and ForkAttention with prefix-aware
internal DP. It does not enable KV offloading.

```bash
source /root/autodl-tmp/agentrix-cu128-env.sh
export VLLM_USE_FLASHINFER_SAMPLER=0
export STARTUP_TIMEOUT=900
cd /root/autodl-tmp/Agentrix/benchmark

MODEL_PATH=/root/autodl-tmp/models/Qwen3-8B \
SERVED_MODEL_NAME=qwen3-8b-dp \
VLLM_BIN=/root/autodl-tmp/Agentrix/vllm/.venv/bin/vllm \
DATASET=agencybench \
DATA_PATH=/root/autodl-tmp/Agentrix/benchmark/data/agencybench_v2.jsonl \
OUTPUT_ROOT=results/dp_agencybench_qwen3_8b_cu128_r1 \
DP_REPLICAS=2 \
GPU_IDS=0,1 \
CASE_COUNT=8 \
BRANCHES=8 \
PREFIX_TOKENS=8192 \
CONCURRENCY=64 \
SUFFIX_MEAN=128 \
OUTPUT_TOKENS=64 \
MAX_NUM_SEQS=32 \
MAX_MODEL_LEN=16384 \
GPU_MEMORY_UTILIZATION=0.80 \
RUN_PRESSURE=0 \
PROFILE_FORK=1 \
./scripts/run_vllm_dp_full_dataset.sh
```

Use a distinct `OUTPUT_ROOT` for each repetition. The runner restarts the
server between variants, so model state and GPU KV cache contents are not
shared across configurations.
