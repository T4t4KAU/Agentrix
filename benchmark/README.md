# Agentrix Bench

This directory contains the Agentrix shared-prefix simulator, API benchmarks,
and local vLLM end-to-end benchmarks. See
[`../README.md`](../README.md) for the complete installation, build, and
reproduction workflow.

## Common Commands

```bash
.venv/bin/agentrix-bench inspect-data
.venv/bin/agentrix-bench simulate
.venv/bin/python -m pytest

BACKENDS="FLASH_ATTN FORK_ATTN" \
PREFIX_TOKENS=8192 \
BRANCHES=16 \
OUTPUT_TOKENS=64 \
./scripts/run_vllm_benchmark.sh

MODEL_PATH=/path/to/Qwen3-0.6B \
PREFIX_TOKENS=2048 \
BRANCHES=2 \
OUTPUT_TOKENS=64 \
./scripts/run_sglang_benchmark.sh

DRY_RUN=1 ./scripts/run_vllm_fanout_matrix.sh
```

## LangGraph RAG Agent Benchmark

Install the optional Agent dependency and run a live graph against an
OpenAI-compatible Agentrix vLLM server:

```bash
uv pip install -e ".[agent,test]"

.venv/bin/agentrix-langgraph live \
  --base-url http://127.0.0.1:9000/v1 \
  --model qwen3-0.6b \
  --task-file configs/langgraph_agent_tasks.jsonl \
  --cases 1 \
  --branches 16 \
  --rag-root ../docs \
  --bootstrap-chunks 24 \
  --bootstrap-max-chars 30000 \
  --concurrency 16 \
  --output results/langgraph/live.json
```

The graph performs an initial retrieval over real local files, plans over the
retrieved evidence, fans out parallel research branches, requires every branch
to issue an OpenAI function call to `rag_search`, feeds each tool result back
to the model, and reduces the branch answers. The live result records every
exact LLM request and tool result.

Use dependency-ordered replay for fair backend comparisons. It preserves the
captured request bodies and runs `planner -> parallel tool calls -> parallel
reflections -> reducer` without carrying live orchestration idle time into the
backend measurement:

```bash
.venv/bin/agentrix-langgraph replay \
  --base-url http://127.0.0.1:9000/v1 \
  --model qwen3-0.6b \
  --trace results/langgraph/live.json \
  --concurrency 16 \
  --output results/langgraph/replay.json
```

Set `--timing captured` only when reproducing the original absolute arrival
timeline. It is not the default throughput comparison mode.

The default model is `Qwen/Qwen3-0.6B`. Set `MODEL_PATH` to use another
Hugging Face model or a local model directory. All output is written under
the Git-ignored `results/` directory. The vLLM script writes one subdirectory
per backend plus `backend_comparison.csv` and `backend_comparison.md` with the
end-to-end latency and throughput deltas.

## WebLINX Multimodal 8-DP Benchmark

Build a deterministic eight-case subset from the WebLINX validation split:

```bash
.venv/bin/python -m weblinx_data \
  --output-dir results/weblinx_subset \
  --split validation \
  --case-count 8 \
  --branch-count 8 \
  --seed 2026
```

The builder selects distinct demonstrations with good screenshots and eight
ranked candidates, downloads only the selected replay files and PNGs, resizes
the screenshots to 1280x720, and writes a reproducible `manifest.json`. The
downloaded data remains under the Git-ignored `results/` directory.

Run the Pressure32K/32-shaped workload on eight DP replicas:

```bash
MODEL_PATH=/path/to/Qwen3.6-27B \
GPU_IDS=0,1,2,3,4,5,6,7 \
MANIFEST="$PWD/results/weblinx_subset/manifest.json" \
NUM_GPU_BLOCKS_OVERRIDE=84 \
./scripts/run_weblinx_8dp.sh
```

Each of the eight WebLINX states first issues one natural multimodal bootstrap
request. Its eight candidates are then expanded into four independent rollout
strategies each, giving 32 branches per state and 256 globally shuffled branch
requests. The client sends no DP-rank header: placement is entirely controlled
by the internal-DP server.

The default matrix contains the three ablations needed to reproduce the text
Pressure32K/32 design: `flash_ordinary`, `fork_ordinary`, and
`fork_prefix_aware`. Only the last arm enables prefix-aware DP routing and
fanout scheduling. All arms use the same 256-token output limit, seeded
lognormal 256-token suffix distribution, KV capacity, and request order.

`TEXT_PREFIX_TOKENS` defaults to 28,000, leaving room for the image tokens,
64-token common analysis, candidate/rollout suffix, and 256 generated tokens
within a 32K context. `MAX_NUM_SEQS` defaults to 64 and Forest CUDA Graphs are
enabled. Calibrate `NUM_GPU_BLOCKS_OVERRIDE` so one root cohort fits per rank
but two independent roots do not; `84` produces 57,344 KV tokens per rank for
the validated Qwen3.6/H20 build. At that deliberately tight boundary, the
validated optimized run still recorded 27 preemptions because Qwen3.6 uses
coarse 784-token hybrid cache pages. The result directory contains per-variant
CSV/JSON summaries, server logs, Prometheus metrics, and `comparison.md`. See
`docs/fork_attention/qwen35_qwen36_forkattention_design.md` for the validated result and its
limitations.

## SGLang Local Benchmark

After adding and installing the `sglang` submodule, run the same Agentrix
OpenAI-compatible workload against SGLang:

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
SGLANG_PYTHON=/path/to/python \
PREFIX_TOKENS=2048 \
BRANCHES=2 \
OUTPUT_TOKENS=64 \
./scripts/run_sglang_benchmark.sh
```

The script launches one SGLang server per `DP_REPLICAS`, routes Agentrix
branches through the existing `agentrix-bench run-api` client, and writes
results under `benchmark/results/sglang_*`. It defaults to `--no-stream` for
the benchmark client because SGLang deployments vary in streaming usage
reporting support. Set `BENCHMARK_EXTRA_ARGS=""` to request streaming mode.

Use `run_vllm_fanout_matrix.sh` for stronger shared-prefix cases. It compares
`FLASH_ATTN` and `FORK_ATTN` over five long-prefix, high-branch-count workloads
and writes an aggregate `matrix_summary.md`.

Use `run_offload_backend_comparison.sh` for the seven-way offload comparison:

```bash
MODEL_PATH=/path/to/Qwen3-1.7B \
CPU_SIZE_GB=0.5 \
DISK_SIZE_GB=2 \
./scripts/run_offload_backend_comparison.sh
```

It covers ForkAttention with no offload, native CPU offload, default and
fork-aware LMCache CPU offload, and fork-aware LMCache CPU plus disk. It also
covers FlashAttention with no offload and ordinary native LRU CPU offload. The
generated `offload_comparison.md` includes pairwise throughput deltas, logical
KV footprint reduction, KV movement, disk footprint, and load failures.

For DP routing experiments, set `DP_REPLICAS=2` and choose `DP_ROUTING`.
`round_robin` is the load-balancing baseline; `prefix_forest` keeps branch
groups together while balancing group weights across replicas.

Set `DP_DEPLOYMENT=internal` to launch one vLLM frontend with multiple internal
DP engines. This exercises vLLM's request router instead of the benchmark-side
router. Current prefix-aware routing is opt-in and works with FlashAttention:

```bash
CUDA_VISIBLE_DEVICES=0,1 VLLM_AGENTRIX_DP_ROUTING_POLICY=prefix_aware \
../vllm/.venv/bin/vllm serve /path/to/Qwen3-8B \
  --data-parallel-size 2 --data-parallel-size-local 2 --api-server-count 1 \
  --attention-config '{"backend":"FLASH_ATTN"}' --enable-prefix-caching
```

The current frontend uses load slack 4, work slack 8,192 token units, decode
weight 16, and a 300-second TTL for completed-prefix hints. Equal-depth hits
prefer less remaining work, rather than more historical visits. Generated
tokens update the remaining-work estimate; preemption restores a conservative
recomputation budget. These hints do not prove GPU cache residency.

Use `scripts/benchmark_prefix_aware_dp.py` for controlled revisit, replicated
prefix, and cold-request comparisons, including per-rank completion counters.
See [DP routing](../docs/dp_routing.md) for tested commands, results, and limits.
The historical `VLLM_FORK_ATTN_DP_PREFIX_*`, Graph-bucket, and arrival-wave
controls described by older recipes are absent from the current frontend.

The following full-dataset recipe targets those older routing controls and is
retained for historical reproduction, not validation of the current router:

```bash
MODEL_PATH=/path/to/Qwen3-8B \
GPU_IDS=0,1 \
./scripts/run_vllm_dp_full_dataset.sh
```

This produces separate `flash_dp`, `fork_dp`, and `fork_optimized_dp` results,
plus an optional pressure-aware offload run. Existing result CSV files are
treated as checkpoints, so an interrupted full-dataset run can be resumed with
the same `OUTPUT_ROOT`. The full-dataset optimized variant uses strict Graph
bucket placement and a 10 ms arrival wave; both ordinary-DP baselines keep the
optimized router and its telemetry path disabled.
