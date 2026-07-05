# Agentrix

Agentrix keeps the shared-prefix attention implementation for vLLM and its
end-to-end benchmark suite in one repository:

- `vllm/` is a Git submodule pinned to the experimental implementation.
- `benchmark/` contains simulation, API, and local vLLM benchmarks.

## System Requirements

- Linux x86_64 with an NVIDIA GPU.
- A working NVIDIA driver and CUDA Toolkit, with `nvcc` available in `PATH`.
- GCC/G++ 11.3 or newer.
- Git, curl, and standard C/C++ build tools. `ccache` is recommended.

Install the basic tools on Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential git curl ccache
nvcc --version
nvidia-smi
```

## Clone the Repository

Initialize the vLLM submodule when cloning:

```bash
git clone --recurse-submodules <agentrix-repository-url> agentrix
cd agentrix
git submodule update --init --recursive
```

For an existing clone:

```bash
git pull
git submodule sync --recursive
git submodule update --init --recursive
```

The `vllm/` submodule should be pinned to commit `e00e85d09` on the
`fork-attn` branch. Before publishing the Agentrix repository, verify that this
commit is available from the vLLM
remote specified in `.gitmodules`.

## Install uv

This project uses `uv` for all Python environment management:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

## Build and Install vLLM

Run these commands from the Agentrix repository root. The editable install must
be executed inside the `vllm/` submodule, not at the Agentrix root directory.

The following commands perform a full C++/CUDA source build. Low build
parallelism avoids exhausting system memory. `TORCH_CUDA_ARCH_LIST=12.0`
targets SM120 / Blackwell GPUs; change it if you build on another GPU
architecture.

```bash
cd vllm
uv venv --python 3.12 --seed
source .venv/bin/activate
export PATH="$PWD/.venv/bin:$PATH"

uv pip install -r requirements/lint.txt
pre-commit install

export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
export PATH="$CUDA_HOME/bin:$PWD/.venv/bin:$PATH"
export MAX_JOBS=2
export NVCC_THREADS=1
export TORCH_CUDA_ARCH_LIST=12.0

VERBOSE=1 uv pip install -v -e . --torch-backend=auto
```

`uv pip install` can still buffer some output because it drives the build
through Python packaging. `-v` asks `uv` to show build output, and `VERBOSE=1`
passes `CMAKE_VERBOSE_MAKEFILE=ON` into vLLM's CMake setup. If you want the
clearest per-target Ninja/CMake progress, use the incremental CMake workflow
below after the editable install has created the environment.

Verify the ForkAttention backend and CUDA Graph dispatch helpers:

```bash
uv pip install -r requirements/test/cuda.txt
.venv/bin/python -m pytest -q \
  tests/v1/worker/test_gpu_block_table.py \
  tests/v1/worker/test_fork_cudagraph_dispatch.py \
  tests/kernels/test_fork_attention.py \
  tests/kernels/test_fork_attention_backend.py
cd ..
```

After modifying files under `vllm/csrc/`, use the incremental build workflow:

```bash
cd vllm
export PATH="$PWD/.venv/bin:$PATH"
export MAX_JOBS=2
export NVCC_THREADS=1
export TORCH_CUDA_ARCH_LIST=12.0

.venv/bin/python tools/generate_cmake_presets.py --force-overwrite
.venv/bin/cmake --preset release -DNVCC_THREADS=1
.venv/bin/cmake --build --preset release --target install --parallel 2 --verbose
cd ..
```

## Install the Benchmark Suite

The benchmark suite uses a separate environment so that it does not alter the
vLLM build dependencies:

```bash
cd benchmark
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -e ".[data,test]"
.venv/bin/python -m pytest
.venv/bin/agentrix-bench inspect-data
cd ..
```

## Run Benchmarks

The scripts start vLLM, wait for its health check, run the benchmark, and then
stop the server. `MODEL_PATH` may be a Hugging Face model ID or a local model
directory.

Run the default ForkAttention benchmark:

```bash
cd benchmark

MODEL_PATH=Qwen/Qwen3-0.6B \
PREFIX_TOKENS=8192 \
BRANCHES=16 \
OUTPUT_TOKENS=64 \
MAX_MODEL_LEN=32768 \
MAX_NUM_SEQS=16 \
GPU_MEMORY_UTILIZATION=0.70 \
OUTPUT_DIR=results/fork_attention_p8192_b16_o64 \
./scripts/run_vllm_benchmark.sh
```

Run the FlashAttention baseline with the same workload:

```bash
cd benchmark

MODEL_PATH=Qwen/Qwen3-0.6B \
PREFIX_TOKENS=8192 \
BRANCHES=16 \
OUTPUT_TOKENS=64 \
MAX_MODEL_LEN=32768 \
MAX_NUM_SEQS=16 \
GPU_MEMORY_UTILIZATION=0.70 \
OUTPUT_DIR=results/flash_attention_p8192_b16_o64 \
./scripts/run_flash_attention_benchmark.sh
```

To run both backends and write a comparison summary in one command:

```bash
cd benchmark

BACKENDS="FLASH_ATTN FORK_ATTN" \
MODEL_PATH=Qwen/Qwen3-0.6B \
PREFIX_TOKENS=8192 \
BRANCHES=16 \
OUTPUT_TOKENS=64 \
MAX_MODEL_LEN=32768 \
MAX_NUM_SEQS=16 \
GPU_MEMORY_UTILIZATION=0.70 \
OUTPUT_DIR=results/fork_vs_flash_p8192_b16_o64 \
./scripts/run_vllm_benchmark.sh
```

The comparison summary is written to
`results/fork_vs_flash_p8192_b16_o64/backend_comparison.md`. Per-backend CSV
files are written under the corresponding `flash_attn/` and `fork_attn/`
subdirectories.

Profile ForkAttention with Nsight Systems:

```bash
cd benchmark

MODEL_PATH=Qwen/Qwen3-0.6B \
PREFIX_TOKENS=8192 \
BRANCHES=16 \
OUTPUT_TOKENS=64 \
MAX_MODEL_LEN=32768 \
MAX_NUM_SEQS=16 \
GPU_MEMORY_UTILIZATION=0.70 \
OUTPUT_DIR=results/fork_attention_nsys_p8192_b16_o64 \
./scripts/run_fork_attention_nsight.sh
```

The Nsight report is written under
`results/fork_attention_nsys_p8192_b16_o64/fork_attn/` as
`fork_attention.nsys-rep` plus the regular benchmark CSV/Markdown outputs.

ForkAttention is intended to run with CUDA Graph capture, so the script leaves
`ENFORCE_EAGER=0` by default. Common overrides include `PORT`,
`MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `GPU_MEMORY_UTILIZATION`, `DTYPE`,
`STARTUP_TIMEOUT`, `KEEP_SERVER`, and `VLLM_SERVER_EXTRA_ARGS`.
Results and server logs are written to the selected `OUTPUT_DIR`.

Use a smaller workload for a quick smoke test:

```bash
cd benchmark
PREFIX_TOKENS=2048 \
BRANCHES=2 \
OUTPUT_TOKENS=32 \
OUTPUT_DIR=results/smoke_fork_attention \
  ./scripts/run_vllm_benchmark.sh
```

## Docker

The provided image builds both the ForkAttention-enabled vLLM submodule and the
benchmark environment on CUDA 13.1:

```bash
docker build \
  --build-arg MAX_JOBS=2 \
  --build-arg NVCC_THREADS=1 \
  --build-arg TORCH_CUDA_ARCH_LIST=12.0 \
  -t agentrix:cu131 .
```

Run a smoke benchmark with access to the NVIDIA GPU and the host Hugging Face
cache:

```bash
docker run --rm \
  --gpus all \
  --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  agentrix:cu131 \
  bash -lc 'cd benchmark && \
    PREFIX_TOKENS=2048 BRANCHES=2 OUTPUT_TOKENS=32 \
    ./scripts/run_vllm_benchmark.sh'
```

For another GPU architecture, override `TORCH_CUDA_ARCH_LIST` at build time.
The default value `12.0` targets NVIDIA Blackwell consumer GPUs.
