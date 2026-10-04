# Agentrix

Agentrix keeps the shared-prefix attention implementation for vLLM, its
LMCache integration, and the end-to-end benchmark suite in one repository:

- `vllm/` is a Git submodule pinned to the experimental implementation.
- `vllm-ascend/` is pinned to `v0.22.1rc1` for Ascend development with vLLM 0.22.1.
- `LMCache/` is a Git submodule pinned to the tiered KV storage implementation.
- `llama.cpp/` is a Git submodule pinned to the CUDA/MUSA ForkAttention implementation.
- `benchmark/` contains simulation, API, and local vLLM benchmarks.

## Documentation

Start with the [technical report and documentation index](docs/README.md).
The [cross-platform report](docs/agentrix_cross_platform_optimizations.md)
explains the design, measured benefits, resource costs, and applicable workloads.
Core references: [Ascend architecture and execution](docs/agentx_ascend.md),
[DP routing](docs/dp_routing.md), [KV memory](docs/kv_memory_optimization_status.md),
[profiling](docs/fork_attention/forkattention_operator_profile.md).

DP deployment now uses the official `vllm-router==0.1.15` package. Custom
prefix/session routing and the unproven KV admission bypass have been removed;
previous routing results remain historical.

The generic build and experimental recipes below include older runtime paths.
Check the relevant backend guide and pinned versions before enabling an
experimental integration.

## System Requirements

The following requirements apply to the vLLM and LMCache workflow. The
`llama.cpp` submodule can instead use a supported Moore Threads GPU and MUSA
SDK as described in its build and runtime guide.

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

Initialize the submodules required by the selected runtime when cloning:

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

Submodule URLs and branch hints are recorded in `.gitmodules`; they are not all
on the same branch. Agentrix records exact commit IDs. Verify those commits are
available from the remotes before publishing the parent repository.

### Ascend development baseline

The Ascend baseline pairs `vllm-ascend v0.22.1rc1` with upstream vLLM 0.22.1
in a separate environment. The `vllm/` submodule contains the CUDA implementation
based on 0.28.0; installing that checkout with this Ascend plugin is not supported.

The validated single-card stack uses openEuler 24.03, Python 3.11,
CANN 9.0.0 with the matching 910B Ops package, PyTorch/torch_npu 2.10.0,
and Triton-Ascend 3.2.1. On the Ascend 910B1 instance, Qwen3-0.6B inference,
ACLGraph decode, shared-prefix caching and eviction, custom Ascend C/Triton
kernels, and pinned CPU/NPU transfers passed the environment checks.
On the two-card Ascend 910B2 instance, Qwen3-8B (128K YaRN) and Qwen3.5-9B
(native 256K) also passed the official AgentX scenario with TP1/DP2 and
session-sticky routing. See [AgentX on Ascend](docs/agentx_ascend.md) for the
reproducible launchers and key results from routing, scheduling, hybrid cache
retention, and decode graph experiments. Full experiment records stay on the
server at the paths documented there. NPU ForkAttention now has an opt-in
Qwen3.5 decode integration with physical-page admission, multiple shared groups,
reusable plans and workspace, and native FIA fallback within ACLGraph. Its
operator measurements remain separate from official AgentX scores; see the
validation and workload limits in the same document.
The FIA-based implementation is the selected development baseline for NPU
ForkAttention and remains disabled by default. The measured official AgentX
run did not select its shared path and showed no end-to-end gain. Tool-wait
offload and resume prefetch driven by Agent Hints remain unimplemented in the
measured configuration; the existing connectors are only the transport basis.

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

When validating more than one CUDA toolkit, keep every Python environment,
CMake build directory, and install prefix separate. In particular, reserve
`vllm/cmake-build-cu130` for the host CUDA 13.0 build and use
`vllm/cmake-build-cu128` only from the CUDA 12.8 container. Never reconfigure
one directory with the other toolkit or PyTorch wheel. A container-private
CUDA 12.8 setup can use `${CUDA_VENV_DIR}` and
`${CUDA_INSTALL_DIR}`; the source checkout may be bind-mounted, but its
environment and compiled output must not be shared with the host build.
On SM120 with CUDA 12.8, set `VLLM_USE_FLASHINFER_SAMPLER=0` so vLLM uses its
native sampler; the current FlashInfer JIT requires CUDA 12.9 or newer for
SM120. This does not disable the ForkAttention backend.

## Build and Enable LMCache

The recipes in this section describe an older integration. The current LMCache
policy registry has no `FORK_AWARE`, and the current CUDA vLLM checkout lacks the
Agentrix residency/placement hooks and native fanout offload options used below.
These comparisons cannot establish an active optimization on the current
checkout. See the [current KV memory audit](docs/kv_memory_optimization_status.md)
before using these historical recipes.

Install LMCache into the same environment as vLLM so the connector and CUDA
extension use the same Python, PyTorch, and CUDA ABI:

```bash
git submodule update --init --recursive
cd LMCache
source ../vllm/.venv/bin/activate
export MAX_JOBS=2
export NVCC_THREADS=1
uv pip install --no-build-isolation -e .
cd ..
```

Create an LMCache configuration. With `disk_cache_mode: eviction`, new KV
chunks enter CPU memory first and are asynchronously demoted to disk only when
the CPU cache evicts them. Omitting this setting preserves LMCache's original
write-through behavior.

```yaml
# /tmp/agentrix-lmcache.yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 8
local_disk: /path/to/cache  # Replace with a private server cache directory.
max_local_disk_size: 128
cache_policy: FORK_AWARE
extra_config:
  disk_cache_mode: eviction
  disk_io_threads: 4
```

Enable the vLLM connector with the configuration file:

```bash
export LMCACHE_CONFIG_FILE=/tmp/agentrix-lmcache.yaml
vllm serve /path/to/model \
  --enable-prefix-caching \
  --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

`FORK_AWARE` keeps high-fanout shared-prefix chunks in CPU ahead of low-value
suffix chunks, uses HOT/COOLING/COLD lifecycle hysteresis, and only promotes a
disk chunk ahead of a lower-value CPU victim when capacity permits. Foreground
loads can still use emergency eviction to preserve vLLM's lookup/load contract.
Use `LRU` to reproduce the unmodified LMCache three-tier baseline.

Run the reproducible three-tier smoke test. It intentionally limits the CPU
cache to a few Qwen3-1.7B KV chunks and lowers vLLM's GPU allocation so
CPU-to-disk demotion is exercised quickly:

```bash
cd benchmark
MODEL_PATH=/path/to/Qwen3-1.7B \
VLLM_BIN=../vllm/.venv/bin/vllm \
./scripts/run_lmcache_tiered_smoke.sh
cd ..
```

Set `LMCACHE_CACHE_POLICY=LRU` or `FORK_AWARE` to compare policies with the same
smoke workload. The command succeeds only if LMCache writes KV chunk files to
the disk tier.
Its configuration, server log, benchmark output, and `smoke_summary.txt` are
written under `benchmark/results/lmcache_tiered_smoke/` by default.

For a fair policy comparison, run the paired benchmark. It always runs
LMCache's default `LRU` policy first, followed by `FORK_AWARE`, with identical
model, tier capacities, dataset, and concurrency:

```bash
cd benchmark
MODEL_PATH=/path/to/Qwen3-1.7B \
VLLM_BIN=../vllm/.venv/bin/vllm \
./scripts/run_lmcache_policy_comparison.sh
cd ..
```

The paired report is written to
`benchmark/results/lmcache_policy_comparison/policy_comparison.md`. Its primary
policy metric is total KV reload demand reduced relative to default LMCache,
reported in tokens, GiB, and percent. Actual retrieval, storage, and disk-load
allocation failures are shown alongside it. The separate logical footprint
table estimates fully duplicated branch-local storage versus shared storage.
It does not measure physical memory saved over vLLM APC, which also shares
prefix pages with FlashAttention.

To compare CPU-only offload, tiered LMCache, and vLLM's native connector with
both attention backends, run:

```bash
cd benchmark
MODEL_PATH=/path/to/Qwen3-1.7B \
VLLM_BIN=../vllm/.venv/bin/vllm \
CPU_SIZE_GB=0.5 \
DISK_SIZE_GB=2 \
PREFIX_TOKENS=4096 \
BRANCHES=8 \
CASE_COUNT=4 \
CONCURRENCY=32 \
./scripts/run_offload_backend_comparison.sh
cd ..
```

The script runs seven configurations with the same workload and capacity:
ForkAttention without offload, ForkAttention native CPU offload, default
LMCache LRU CPU offload, fork-aware LMCache CPU offload, fork-aware LMCache
CPU plus disk, FlashAttention without offload, and FlashAttention native LRU
CPU offload. The script passes historical fanout admission and hot-prefix
protection options for ForkAttention; the current native connector does not
implement them. On a runtime that supports these options, changing both the
attention backend and cache policy would confound their individual effects.

The report is written to
`benchmark/results/offload_backend_comparison/offload_comparison.md`. It shows
end-to-end and branch throughput, pairwise offload impact within each backend,
KV load/store traffic, disk footprint, load failures, and a hypothetical logical
KV footprint comparison between duplicated and shared storage. This last metric
is not a measured FlashAttention-versus-ForkAttention memory saving.
Repeat the command into distinct `OUTPUT_DIR` values
and use paired medians when collecting performance results.

## Install the Benchmark Suite

The benchmark suite uses a separate environment so that it does not alter the
vLLM build dependencies:

```bash
cd benchmark
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -e ../application
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

ForkAttention can capture a sparse set of CUDA Graph plan capacities instead
of the full batch-size and plan-bucket product. Profile a representative run,
then pass the hot capacities to the matrix runner:

```bash
VLLM_FORK_ATTN_CUDAGRAPH_CAPTURE_BUCKETS="common:4,8;forest:256,512,1024" \
./scripts/run_vllm_fanout_matrix.sh
```

`server_profile.json` records CUDA Graph hit/miss counters and average
ForkAttention metadata construction time for selecting these capacities.

ForkAttention is intended for synchronized decode fanout, not arbitrary prompt
batching: several active requests must consume the same resident physical KV
prefix in the same decode step. On the evaluated GPU, use it primarily for
4K-8K or longer shared prefixes with high shared-context ratio and at least two
active siblings; 8-16 aligned branches have the strongest evidence. Single
continuations, prefill, short prefixes, or unrelated round-robin cases should
remain on FlashAttention. The exact boundary and KV-work model are documented
in [the operator profile](docs/fork_attention/forkattention_operator_profile.md#applicability-boundary).

Long-prefix tail batches use shape-aware adaptive splitting by default. The
planner targets about two CTA waves only when the shared prefix is at least 4K
tokens; wide cohorts retain the base 2K chunk. CUDA Graph runs select a matching
captured bucket, while `ENFORCE_EAGER=1` runs build the equivalent dynamic
forest plan. Use these controls for matched A/B experiments:

```bash
# Static-chunk control, valid with CUDA Graphs enabled or ENFORCE_EAGER=1.
VLLM_FORK_ATTN_TARGET_CTA_WAVES=0 ./scripts/run_vllm_benchmark.sh

# Adaptive default; 4/8/12 are captured common-prefix chunk capacities.
VLLM_FORK_ATTN_TARGET_CTA_WAVES=2 \
VLLM_FORK_ATTN_ADAPTIVE_SPLIT_MIN_TOKENS=4096 \
VLLM_FORK_ATTN_PREFIX_CHUNK_BUCKETS=4,8,12 \
./scripts/run_vllm_benchmark.sh
```

`VLLM_FORK_ATTN_TAIL_TILE_N=32` is an experimental tail-kernel A/B override;
zero keeps automatic tile selection. Do not make an architecture-specific tile
the default without measuring both small-tail and full-cohort cases.

AgentBoard and AppWorld prompt snapshots are included under `benchmark/data/`:

```bash
DATASET=agentboard ./scripts/run_vllm_benchmark.sh
DATASET=appworld ./scripts/run_vllm_benchmark.sh
```

Regenerate both snapshots from complete dataset checkouts with:

```bash
benchmark/scripts/import_agent_datasets.py \
  --agentboard-source /path/to/AgentBoard \
  --appworld-source /path/to/appworld
```

### Dataset and DP experiments

Use `benchmark/scripts/run_vllm_dataset_matrix.sh` for the bundled AgentBoard
and AppWorld attention-backend matrix. Set `MODEL_PATH`, `BACKENDS`, and a
server-side `OUTPUT_ROOT` explicitly. This is a systems workload; task quality
requires the corresponding dataset evaluator.

For current DP routing, use the official router and the entry points in the
[DP guide](docs/dp_routing.md). The retired private-router matrix launchers have
been removed; archived recipes require their matching historical runtime.

On a two-GPU machine, run two single-GPU vLLM replicas and compare DP routing
policies. `round_robin` is the load-balancing baseline. `prefix_forest` keeps
each branch group on one replica while greedily balancing group weights across
replicas:

```bash
cd benchmark

BACKENDS="FORK_ATTN" \
DP_REPLICAS=2 \
DP_ROUTING=prefix_forest \
GPU_IDS="0,1" \
MODEL_PATH=Qwen/Qwen3-0.6B \
PREFIX_TOKENS=8192 \
SAMPLE_COUNT=4 \
CASE_COUNT=4 \
BRANCHES=32 \
BRANCH_GROUP_SIZE=8 \
CONCURRENCY=64 \
OUTPUT_TOKENS=64 \
MAX_MODEL_LEN=32768 \
MAX_NUM_SEQS=32 \
GPU_MEMORY_UTILIZATION=0.70 \
OUTPUT_DIR=results/fork_dp2_prefix_forest \
./scripts/run_vllm_benchmark.sh
```

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
`ENFORCE_EAGER=0` by default. Set `TP_SIZE` and provide
`DP_REPLICAS * TP_SIZE` entries in `GPU_IDS` for tensor-parallel model tests.
Common overrides include `PORT`,
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

CUDA 12.8 can compile the same SM120 kernels. Pin `TORCH_BACKEND=cu128` so a
newer host driver does not make uv select a CUDA 13 PyTorch wheel. A single
build job keeps the peak memory of large CUTLASS translation units bounded:

```bash
docker build \
  --build-arg CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu22.04 \
  --build-arg TORCH_BACKEND=cu128 \
  --build-arg MAX_JOBS=1 \
  --build-arg NVCC_THREADS=1 \
  --build-arg TORCH_CUDA_ARCH_LIST=12.0 \
  -t agentrix:cu128 .
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
The default value `12.0` targets NVIDIA Blackwell consumer GPUs. Keep
`CUDA_IMAGE` and `TORCH_BACKEND` on the same CUDA release family.
