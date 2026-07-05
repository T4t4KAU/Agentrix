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
