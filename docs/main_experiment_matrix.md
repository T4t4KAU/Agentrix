# Main Experiment Procedures

## Purpose

This guide launches the current Agentrix shared-prefix experiments. It keeps
offload measurements on a physical single-GPU host and runs no-offload DP/TP
measurements on the multi-GPU server.

The documented groups are:

| Group | Models | Datasets | Prefixes | Branches | Variants | Runs |
|---|---:|---:|---:|---:|---:|---:|
| Single GPU | 2 | 4 | 2 | 2 | 5 | 160 |
| Data parallel | 1 | 1 | 2 | 1 | 2 | 12 |
| TP accuracy | 1 | 4 | 2 | 3 | 3 | 72 |

The default record cap is 32 records per dataset. Set
`MAX_DATASET_RECORDS=0` to consume every source record. Use a new
`OUTPUT_ROOT` whenever the record cap or another workload parameter changes;
completed CSV files are intentionally skipped on resume.

## Shared Controls

Run commands from the `benchmark` directory. Supply machine-specific model
paths through `MODEL_SPECS`; do not edit tracked scripts.

The single-GPU and TP matrix controls include:

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
historical matrix. The matched Nsight operator breakdown and optimization
ranking are documented in
[`forkattention_operator_profile.md`](forkattention_operator_profile.md):

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

Use `offload_validated` for an attributable offload comparison. It runs four
case-major roots without explicit warm-up and separates ordinary offload,
ordinary offload with fanout scheduling, and the optimized connector policy.
The workload must report nonzero load and store traffic; otherwise lower
`NUM_GPU_BLOCKS_OVERRIDE` or add cases before interpreting it as an offload
experiment.

```bash
cd benchmark

VLLM_BIN="$PWD/.venv/bin/vllm" \
MODE=single_gpu \
EXPERIMENT_PROFILE=offload_validated \
MODEL_SPECS='qwen3-1.7b|/path/to/Qwen3-1.7B' \
DATASETS=agentboard \
PREFIX_LENGTHS='8192,16384' \
BRANCH_COUNTS=16 \
MAX_DATASET_RECORDS=4 \
NUM_GPU_BLOCKS_OVERRIDE=1700 \
OFFLOAD_CPU_GIB=8 \
./scripts/run_main_experiment.sh
```

Keep `FANOUT_PROFILE=0` for performance measurements. Set
`FANOUT_PROFILE=1` only for scheduler diagnostics; it emits one detailed
fanout-admission record per scheduling step and can measurably reduce the
optimized variant's throughput. A minimal post-change LMCache CPU/disk
correctness smoke is:

```bash
cd /path/to/Agentrix

OUTPUT_DIR=results/lmcache_tiered_smoke \
LMCACHE_CACHE_POLICY=FORK_AWARE \
MODEL_PATH=/path/to/Qwen3-1.7B \
VLLM_BIN="$PWD/benchmark/.venv/bin/vllm" \
bash benchmark/scripts/run_lmcache_tiered_smoke.sh
```

This smoke verifies that `disk_cache_mode=eviction` initializes, the request
completes without storage errors, and CPU evictions create disk chunks. It is a
correctness check, not a throughput comparison; LMCache remains outside the
published main matrix.

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

The current group compares FlashAttention ordinary DP, ForkAttention ordinary
DP, and ForkAttention with the final capacity-aware prefix router on two
internal DP ranks.

The validation contains one three-way Adaptive16K run and two repetitions per
variant for Pressure32K/32, for nine variant runs in total. All variants pin
3,852 KV blocks per rank, use the same workload and physical KV capacity, and
disable offload and reload rebalance. ForkAttention runs enable Prefix Forest
CUDA Graphs. The exact setup and current results are documented in
[`dp_experiment_results.md`](dp_experiment_results.md).

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

Long runs should be detached from the SSH or IDE session by wrapping the exact
group command in `nohup env ... >experiment.log 2>&1 < /dev/null &`. Use a new
output root for every DP repetition so paired runs cannot overwrite one
another. The single-GPU and TP matrices resume by skipping completed result
files. Inspect their progress with:

```bash
find results/main_experiment_full32/single_gpu \
  -name benchmark_results.csv -size +0c | wc -l  # expected: 160
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
  results/dp_capacity_aware_pressure16k/dp \
  --output results/dp_capacity_aware_pressure16k/dp/main_experiment_report.md
```

Before publishing results, verify that every compared cell contains all of its
variants, manifests report clean Agentrix and vLLM worktrees, and no run used
the experimental reload-rebalance switch.
