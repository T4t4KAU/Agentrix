# Executable Coding-Agent Quality A/B

## Purpose

This experiment closes the task-quality loop for the full Agentrix system.
Unlike a deterministic systems replay, every task starts from a pinned source
snapshot with an injected regression. The model must inspect the repository,
edit an allowed source file, and pass both a visible regression test and a
hidden neighboring-behavior test.

The primary quality metric is resolved-task rate. The performance metric is
quality-adjusted throughput in resolved tasks per hour. Agentrix is considered
non-inferior only when the lower bound of the paired bootstrap 95% confidence
interval for the resolved-rate difference is no lower than the predeclared
-5 percentage-point margin. Both arms must also resolve at least 50% of tasks;
equal zero-success arms cannot produce a vacuous non-inferiority claim.

## Compared Arms

| Arm | Attention | DP routing | Prompt compaction |
|---|---|---|---|
| Baseline | FlashAttention | ordinary DP | disabled |
| Agentrix | ForkAttention | prefix-aware DP | enabled |

Both arms use the same model, task order, greedy decoding, tool limits, hidden
tests, and source revisions. By default, the baseline receives 3,852 KV blocks
per rank and Agentrix receives 2,500. This deliberately tests whether the
optimized system can retain task quality under a smaller physical KV pool.
Override both values to the same count for a pure performance control.

The launcher refuses to run when a selected GPU already uses more than 1 GiB,
preventing unrelated processes from contaminating NVML measurements.

## Prepare Sources

```bash
benchmark/.venv/bin/python \
  benchmark/scripts/prepare_coding_oracle_sources.py \
  --task-root benchmark/coding_tasks \
  --output-root /path/to/coding_sources
```

The downloader obtains the exact Django, SQLite, and FFmpeg revisions declared
by the task manifests and writes a revision marker beside each snapshot.

## Run

```bash
MODEL_PATH=/path/to/Qwen3-8B \
SOURCE_ROOT=/path/to/coding_sources \
OUTPUT_ROOT=benchmark/results/coding_quality_ab \
GPU_IDS=0,1,2,3 \
DP_REPLICAS=4 \
BASELINE_NUM_GPU_BLOCKS=3852 \
AGENTRIX_NUM_GPU_BLOCKS=2500 \
benchmark/scripts/run_coding_agent_quality_matrix.sh
```

For a capacity-matched control:

```bash
BASELINE_NUM_GPU_BLOCKS=3852 \
AGENTRIX_NUM_GPU_BLOCKS=3852 \
OUTPUT_ROOT=benchmark/results/coding_quality_ab_equal_capacity \
benchmark/scripts/run_coding_agent_quality_matrix.sh
```

## Outputs

Each task writes its complete action trace, patch evaluation, public and hidden
test results, token counts, TTFT, and total wall time to `run.json`. Each arm
also records process-level NVML memory samples and its configured KV capacity.

The final `quality_ab.md` and `quality_ab.json` report:

- resolved-task rate;
- public and hidden test pass rates;
- invalid-patch rate;
- paired resolved-rate difference and bootstrap 95% interval;
- non-inferiority decision;
- resolved tasks per hour;
- input/output token counts and mean TTFT;
- aggregate peak NVML process memory and KV blocks per rank.

The initial 12-task suite is a formal pilot. A final competition result should
use at least 20 independently reviewed tasks per repository so the confidence
interval can support a five-point non-inferiority margin.
