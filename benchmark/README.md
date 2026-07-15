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
router. ForkAttention prefix-aware routing is opt-in:

```bash
DP_DEPLOYMENT=internal \
DP_REPLICAS=2 \
GPU_IDS=0,1 \
VLLM_FORK_ATTN_DP_PREFIX_ROUTING=1 \
./scripts/run_vllm_benchmark.sh
```

`VLLM_FORK_ATTN_DP_PREFIX_LOAD_SLACK` bounds how far an affinity-selected rank
may exceed the least-loaded rank in vLLM's `waiting * 4 + running` score. The
default `32` permits an eight-request shared-prefix cohort to remain together.
Finished prefixes remain soft routing hints for 30 seconds by default; control
this with `VLLM_FORK_ATTN_DP_PREFIX_WARM_TTL`.

The optimized internal router also consumes physical ForkAttention telemetry
published through vLLM's existing DP stats channel. It uses the configured
forest CTA buckets as discrete Graph costs, applies token-weighted load bounds,
and groups requests arriving in the same short wave by their deepest shared
logical subtree. The main controls are:

```text
VLLM_FORK_ATTN_DP_GRAPH_SLACK_BUCKETS=1
VLLM_FORK_ATTN_DP_WORK_SLACK_TOKENS=8192
VLLM_FORK_ATTN_DP_DECODE_TOKEN_WEIGHT=16
VLLM_FORK_ATTN_DP_ARRIVAL_WAVE_MS=1
```

Run the complete dataset comparison with two internal DP ranks using:

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
