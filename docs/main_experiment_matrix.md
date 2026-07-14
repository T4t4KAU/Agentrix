# Complete Main Experiment Matrix

## Purpose

This guide launches the complete Agentrix shared-prefix experiment matrix. It
keeps offload measurements on a physical single-GPU host and runs only
no-offload DP/TP measurements on the multi-GPU server.

The complete shape matrix contains 304 runs:

| Group | Models | Datasets | Prefixes | Branches | Variants | Runs |
|---|---:|---:|---:|---:|---:|---:|
| Single GPU | 2 | 4 | 2 | 2 | 5 | 160 |
| Data parallel | 1 | 4 | 2 | 3 | 3 | 72 |
| TP accuracy | 1 | 4 | 2 | 3 | 3 | 72 |

The default record cap is 32 records per dataset. Set
`MAX_DATASET_RECORDS=0` to consume every source record. Use a new
`OUTPUT_ROOT` whenever the record cap or another workload parameter changes;
completed CSV files are intentionally skipped on resume.

## Shared Controls

Run commands from the `benchmark` directory. Supply machine-specific model
paths through `MODEL_SPECS`; do not edit tracked scripts.

The legacy runner profile fixes the following comparison controls:

- Prefix lengths: 8K and 16K tokens.
- Deterministic greedy generation with 64 output tokens.
- FlashInfer sampler disabled for both attention backends.
- Experimental KV reload rebalance disabled.
- Identical GPU block capacity within each performance comparison.
- LMCache and disk storage excluded from the main matrix.
- Remote DP and TP experiments use no KV offload.

Each run records Git provenance, the record cap, GPU KV occupancy, logical KV
read reduction, physical ForkAttention active steps and CTA counts,
throughput, TTFT, TPOT, P50/P95/P99 latency, GPU utilization,
memory-controller utilization, and offload traffic where applicable.

## Single GPU

This group compares:

1. FlashAttention without offload.
2. ForkAttention without offload.
3. FlashAttention with ordinary LRU CPU offload.
4. ForkAttention with ordinary LRU CPU offload.
5. ForkAttention with prefix-aware optimized CPU offload.

For a valid no-offload ForkAttention operator/system comparison, use the
default `fanout_validated` profile. It uses one case per batch, case-major
admission, exact shared-prefix warm-up, 256 output tokens, and enables Fork
fanout admission and forest CUDA Graphs. The forest graph workspace reserves
the full 32-split range supported by the gather kernel, so branch points cannot
overflow a sequence-length-derived capture size. Results go to
`results/main_experiment_v2` so they cannot be silently mixed with the
historical matrix:

```bash
cd benchmark

MODE=single_gpu \
EXPERIMENT_PROFILE=fanout_validated \
MODEL_SPECS='qwen3-1.7b|/path/to/Qwen3-1.7B' \
DATASETS='agentboard,appworld,agencybench,swebench' \
PREFIX_LENGTHS='8192,16384' \
BRANCH_COUNTS='8,16' \
MAX_DATASET_RECORDS=32 \
NUM_GPU_BLOCKS_OVERRIDE=1700 \
VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH=1 \
./scripts/run_main_experiment.sh
```

The command below reproduces the historical mixed-prefix scheduler/offload
stress matrix. Keep `EXPERIMENT_PROFILE=legacy` explicit when publishing those
numbers; they must not be described as a pure ForkAttention operator result.

Calibrate `NUM_GPU_BLOCKS_OVERRIDE` to the lowest capacity supported by all
five variants on the target GPU. The value below is only an example.

```bash
cd benchmark

MODE=single_gpu \
EXPERIMENT_PROFILE=legacy \
MODEL_SPECS='qwen3-1.7b|/path/to/Qwen3-1.7B;llama3.2-1b|/path/to/Llama-3.2-1B-Instruct' \
DATASETS='agentboard,appworld,agencybench,swebench' \
PREFIX_LENGTHS='8192,16384' \
BRANCH_COUNTS='8,16' \
CASE_COUNT=4 \
MAX_DATASET_RECORDS=32 \
OUTPUT_TOKENS=64 \
COMMON_ANALYSIS_TOKENS=64 \
NUM_GPU_BLOCKS_OVERRIDE=1700 \
OFFLOAD_CPU_GIB=8 \
OUTPUT_ROOT=results/main_experiment_full32 \
./scripts/run_main_experiment.sh
```

## Data Parallel

This group compares FlashAttention ordinary DP, ForkAttention ordinary DP,
and ForkAttention prefix-aware DP on two internal DP ranks. All variants run
without offload. `NUM_GPU_BLOCKS_OVERRIDE` is per rank for this deployment.

```bash
cd benchmark

MODE=dp \
MODEL_SPECS='qwen3-8b|/path/to/Qwen3-8B' \
DATASETS='agentboard,appworld,agencybench,swebench' \
PREFIX_LENGTHS='8192,16384' \
BRANCH_COUNTS='8,16,32' \
CASE_COUNT=4 \
MAX_DATASET_RECORDS=32 \
OUTPUT_TOKENS=64 \
COMMON_ANALYSIS_TOKENS=64 \
GPU_IDS='0,1' \
DP_REPLICAS=2 \
TP_SIZE=1 \
NUM_GPU_BLOCKS_OVERRIDE=3852 \
OUTPUT_ROOT=results/main_experiment_full32 \
./scripts/run_main_experiment.sh
```

## TP Accuracy

This group runs one FlashAttention reference and two ForkAttention repeats
with TP=2 and no offload. The generated report includes Flash-to-Fork output
agreement and Fork run-to-run repeatability. It is a deterministic backend
accuracy guardrail, not environment-level task success.

For performance claims, calibrate and pin `NUM_GPU_BLOCKS_OVERRIDE` across
backends. It may be omitted when the run is used only for output agreement.

```bash
cd benchmark

MODE=tp_accuracy \
MODEL_SPECS='qwen3-14b|/path/to/Qwen3-14B' \
DATASETS='agentboard,appworld,agencybench,swebench' \
PREFIX_LENGTHS='8192,16384' \
BRANCH_COUNTS='8,16,32' \
CASE_COUNT=4 \
MAX_DATASET_RECORDS=32 \
OUTPUT_TOKENS=64 \
COMMON_ANALYSIS_TOKENS=64 \
GPU_IDS='0,1' \
DP_REPLICAS=1 \
TP_SIZE=2 \
OUTPUT_ROOT=results/main_experiment_full32 \
./scripts/run_main_experiment.sh
```

## Detached Runs

Long matrices should be detached from the SSH or IDE session:

```bash
cd benchmark

nohup env MODE=dp \
  MODEL_SPECS='qwen3-8b|/path/to/Qwen3-8B' \
  DATASETS='agentboard,appworld,agencybench,swebench' \
  PREFIX_LENGTHS='8192,16384' \
  BRANCH_COUNTS='8,16,32' \
  CASE_COUNT=4 \
  MAX_DATASET_RECORDS=32 \
  OUTPUT_TOKENS=64 \
  COMMON_ANALYSIS_TOKENS=64 \
  GPU_IDS='0,1' \
  DP_REPLICAS=2 \
  TP_SIZE=1 \
  NUM_GPU_BLOCKS_OVERRIDE=3852 \
  OUTPUT_ROOT=results/main_experiment_full32 \
  ./scripts/run_main_experiment.sh \
  >main_experiment_dp.log 2>&1 < /dev/null &
```

Re-running the same command resumes the matrix. Inspect progress by counting
non-empty result files:

```bash
find results/main_experiment_full32/single_gpu \
  -name benchmark_results.csv -size +0c | wc -l  # expected: 160
find results/main_experiment_full32/dp \
  -name benchmark_results.csv -size +0c | wc -l  # expected: 72
find results/main_experiment_full32/tp_accuracy \
  -name benchmark_results.csv -size +0c | wc -l  # expected: 72
```

## Reports

Each completed mode writes:

- `main_experiment_report.md`: human-readable metrics and provenance.
- `main_experiment_report.csv`: all raw report columns.
- `raw_api_results.json`: request-level outputs and timing.
- `telemetry.json`: sampled GPU and KV cache measurements.
- `output_agreement.json`: TP output agreement or repeatability.

Regenerate a report after copying results between machines:

```bash
.venv/bin/python src/main_experiment_report.py \
  results/main_experiment_full32/dp \
  --output results/main_experiment_full32/dp/main_experiment_report.md
```

Before publishing results, verify that every compared cell contains all of its
variants, manifests report clean Agentrix and vLLM worktrees, and no run used
the experimental reload-rebalance switch.
