# HotpotQA LangGraph End-to-End Experiment

## Scope

This document records the corrected 2026-07-30 live LangGraph experiment.
It measures the relationship between Agent activity, ForkAttention execution,
GPU KV-cache occupancy, and end-to-end performance.

This is online inference rather than trajectory replay. `StateGraph` and
dynamic `Send` construct the workflow at runtime, and every planner,
tool-selection, reflection, and reducer turn is sent to the local vLLM server
through its OpenAI-compatible HTTP API. Tool calls execute against the
HotpotQA candidate paragraphs.

Each model turn is an independent HTTP request. In particular, reflection
does not resume an in-process request that retained private KV across the tool
boundary. This preserves the repeated long-prefix request structure that the
experiment is intended to evaluate.

## Workflow

Each case follows this live dependency chain:

```text
HotpotQA question and candidate paragraphs
  -> case-scoped bootstrap retrieval
  -> online planner request
  -> StateGraph Send to 16 branches
  -> 16 online tool-selection requests
  -> 16 paragraph_search tool executions
  -> 16 online evidence-reflection requests
  -> online reducer request
```

The run contains 100 distinct HotpotQA cases. Four cases are active
concurrently, giving up to 64 simultaneous sibling branch requests. In total,
each arm completes 1,600 branches and records 5,100 Agent lifecycle events.

## Dataset and Prefix Distribution

The source is the official HotpotQA distractor development split with
SHA-256:

```text
4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea
```

The frozen manifest is
[`benchmark/configs/hotpot_agentrix_long_prefix_100.jsonl`](../benchmark/configs/hotpot_agentrix_long_prefix_100.jsonl).
It contains 100 different target questions and deterministic donor
paragraphs. Measured with the Qwen3 tokenizer, the shared root ranges from
7,871 to 14,311 tokens, with a mean of 11,236 tokens. The branch count is
overridden to 16 for this experiment.

Accuracy is deliberately not reported. The experiment studies systems
performance and KV residency, not a difficult-subset quality score.

## Compared Modes

| Arm | Attention | Exact prompt compaction | Prefix cache | TTL | KV offload | DP |
|---|---|---|---|---|---|---|
| Flash baseline | `FLASH_ATTN` | Off | On | Off | Off | Off |
| Optimized | `FORK_ATTN` | On | On | Off | Off | Off |

The comparison therefore measures the complete requested optimized mode,
ForkAttention plus exact prompt compaction, against an uncompressed
FlashAttention baseline. It is not a pure attention-kernel ablation.

The tool-selection phase supplies a cleaner ForkAttention signal because it
occurs before tool results exist and therefore cannot benefit from prompt
compaction.

## Executed Configuration

| Setting | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB |
| Model | Qwen3-14B, BF16 |
| Cases / concurrent cases | 100 / 4 |
| Branches per case / total branches | 16 / 1,600 |
| Maximum client request concurrency | 64 |
| Maximum server sequences | 80 |
| Model context limit | 24,576 tokens |
| Maximum batched tokens | 16,384 |
| GPU memory utilization | 0.9 |
| GPU KV capacity | 22,770 blocks / 364,320 tokens in both arms |
| Output limits | planner 128, tool selection 64, reflection 256, reducer 192 |
| Tool delay | zero |
| Async scheduling | disabled |
| Sampling | greedy, model thinking disabled |
| Fork CUDA Graph capture | `common:4,8,12;forest:2048` |
| Repetitions | one complete matched pair |

The server is warmed before measurement. Engine startup and CUDA Graph capture
are excluded from workflow wall time. The GPU KV pool is selected
automatically from the same 0.9 memory-utilization limit and is identical in
both arms.

## End-to-End Performance and Memory Results

Both arms completed all 100 cases, 1,600 branches, and 5,100 lifecycle events.

| Metric | Flash baseline, no compaction | ForkAttention + compaction | Change |
|---|---:|---:|---:|
| Workflow wall time | 1,395.17 s | 960.27 s | **-31.17% / 1.453x** |
| Prompt tokens | 44,081,743 | 40,117,888 | -8.99% |
| Completion tokens | 302,398 | 268,304 | -11.27% |
| Total model tokens | 44,384,141 | 40,386,192 | -9.01% |
| Total model tokens/s | 31,812.81 | 42,057.34 | **+32.20%** |
| GPU KV usage-seconds | 377.91 | 131.11 | **-65.31%** |
| Time-averaged GPU KV usage | 27.09% | 13.65% | **-13.44 pp / -49.60%** |
| Peak live GPU KV usage | 74.69% | 24.19% | **-50.50 pp / -67.62%** |
| Peak live KV tokens | 272,108 | 88,116 | **-183,992 / -67.62%** |
| Peak process GPU memory | 89,371 MiB | 89,693 MiB | +322 MiB |
| Maximum running / waiting requests | 64 / 31 | 64 / 31 | equal |

The optimized mode removes 15,137,150 repeated tool-result characters. This
explains the 9.01% reduction in total model tokens and contributes to the
reflection and reducer gains.

The live-KV metrics and process-level GPU-memory metric measure different
things. ForkAttention reduces the number and lifetime of live KV tokens, but
its CUDA Graph and kernel workspaces increase the sampled process peak by 322
MiB. Because vLLM preallocates the equal 364,320-token KV pool in both arms,
the higher process peak does not mean that ForkAttention retains more KV.

`GPU KV usage-seconds` is the time integral of the periodic KV utilization
series. It captures both how much KV is live and how long it remains live.
The optimized arm reduces this area by 65.31%, while also completing the
workflow 31.17% sooner.

## Stage-Level Performance

| LLM stage | Flash mean latency | ForkAttention + compaction | Change | Prompt-token interpretation |
|---|---:|---:|---:|---|
| Planner | 11,107.38 ms | 10,798.57 ms | -2.78% | identical prompts |
| Branch tool selection | 9,938.35 ms | 5,931.82 ms | **-40.31%** | identical prompt tokens |
| Branch reflection | 17,255.41 ms | 9,202.04 ms | **-46.67%** | ForkAttention plus compaction |
| Reducer | 6,876.43 ms | 5,837.72 ms | -15.11% | shorter branch outputs |

The branch tool-selection phase processes exactly 18,833,694 prompt tokens in
each arm. Completion volume differs by only 0.46%, yet mean latency falls by
40.31%. This same-token phase is the strongest end-to-end evidence that the
ForkAttention execution path itself accelerates the synchronized 16-branch
cohort.

The optimized server reports 9,839 active ForkAttention steps out of 11,122
observed measured steps, an 88.46% activation rate. It also records 326,528
shared and 687,945 singleton CTA plan entries. These counters confirm that
the configured ForkAttention operator executed during the measured workflow.

## Agent and KV Timeline

![Qwen3-14B 100-case LangGraph Agent and KV timeline](assets/hotpot_langgraph_http_100case_b16_no_ttl_qwen14b_c4.png)

The upper panels align outstanding planner, branch tool-selection, tool,
reflection, and reducer operations with wall-clock time. The lower panels show
GPU KV-cache utilization on the same clock. The Flash baseline reaches a
74.69% live-KV peak and repeatedly returns to a higher occupancy band. The
optimized arm remains below 24.19% and finishes 434.90 seconds earlier.

The plotted lifecycle and complete systems/stage summary are stored in
[`experiment_results/hotpot_langgraph_http_100case_b16_no_ttl_qwen14b_c4.csv`](experiment_results/hotpot_langgraph_http_100case_b16_no_ttl_qwen14b_c4.csv).
Raw run JSON, memory samples, Prometheus snapshots, and server logs remain on
the experiment server under
`benchmark/results/hotpot_http_100case_b16_no_ttl_qwen14b_c4/`.

## Interpretation

Under the corrected independent-request HTTP workflow, the requested
optimized mode is 1.453x faster and substantially reduces live KV residency.
The result is consistent with the intended ForkAttention operating region:
long case-specific roots followed by synchronized 16-way sibling fanout.

The full 1.453x result combines two mechanisms and must not be attributed to
ForkAttention alone. The pre-compaction tool-selection stage isolates the
shared-prefix effect more closely and shows a 40.31% latency reduction at
identical prompt-token volume.

This experiment disables TTL, offload, and DP. It therefore does not make a
claim about eviction, CPU recovery, or distributed routing.

## Reproduction

Run the baseline first:

```bash
export VLLM_FORK_ATTN_CUDAGRAPH_CAPTURE_BUCKETS='common:4,8,12;forest:2048'

MODEL_PATH=/root/autodl-tmp/models/Qwen3-14B \
HOTPOT_PATH=/root/autodl-tmp/data/HotpotQA/hotpot_dev_distractor_v1.json \
HOTPOT_CASE_FILE="$PWD/benchmark/configs/hotpot_agentrix_long_prefix_100.jsonl" \
VLLM_BIN="$PWD/vllm/.venv/bin/vllm" \
OUTPUT_ROOT="$PWD/benchmark/results/hotpot_http_100case_b16_no_ttl_qwen14b_c4" \
VARIANTS=baseline \
PROMPT_COMPACTION=0 \
CASES=100 CASE_CONCURRENCY=4 HOTPOT_BRANCHES=16 CONCURRENCY=64 \
GPU_MEMORY_UTILIZATION=0.9 \
MAX_MODEL_LEN=24576 MAX_NUM_BATCHED_TOKENS=16384 MAX_NUM_SEQS=80 \
bash benchmark/scripts/run_hotpot_agentrix_e2e.sh
```

Then run the optimized arm into the same result root:

```bash
MODEL_PATH=/root/autodl-tmp/models/Qwen3-14B \
HOTPOT_PATH=/root/autodl-tmp/data/HotpotQA/hotpot_dev_distractor_v1.json \
HOTPOT_CASE_FILE="$PWD/benchmark/configs/hotpot_agentrix_long_prefix_100.jsonl" \
VLLM_BIN="$PWD/vllm/.venv/bin/vllm" \
OUTPUT_ROOT="$PWD/benchmark/results/hotpot_http_100case_b16_no_ttl_qwen14b_c4" \
VARIANTS=forkattention \
PROMPT_COMPACTION=1 \
CASES=100 CASE_CONCURRENCY=4 HOTPOT_BRANCHES=16 CONCURRENCY=64 \
GPU_MEMORY_UTILIZATION=0.9 \
MAX_MODEL_LEN=24576 MAX_NUM_BATCHED_TOKENS=16384 MAX_NUM_SEQS=80 \
bash benchmark/scripts/run_hotpot_agentrix_e2e.sh
```
