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

DRY_RUN=1 ./scripts/run_vllm_fanout_matrix.sh
```

The default model is `Qwen/Qwen3-0.6B`. Set `MODEL_PATH` to use another
Hugging Face model or a local model directory. All output is written under
the Git-ignored `results/` directory. The vLLM script writes one subdirectory
per backend plus `backend_comparison.csv` and `backend_comparison.md` with the
end-to-end latency and throughput deltas.

Use `run_vllm_fanout_matrix.sh` for stronger shared-prefix cases. It compares
`FLASH_ATTN` and `FORK_ATTN` over five long-prefix, high-branch-count workloads
and writes an aggregate `matrix_summary.md`.

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
