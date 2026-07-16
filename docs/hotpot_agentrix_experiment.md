# HotpotQA Agentrix Long-Prefix Experiment

## Scope

This document records the 2026-07-16 HotpotQA positive-control experiment for
Agentrix ForkAttention. The workload intentionally emphasizes a long shared
parent prompt and a wide, synchronized sibling fanout. It is useful for
validating the intended ForkAttention operating region, but it is not an
estimate of the natural HotpotQA or production RAG request distribution.

The experiment is live end to end. It does not replay a fixed LLM trace or
inject precomputed planner, branch, or reducer responses.

## Pipeline

Each case follows this dependency chain:

```text
HotpotQA question and candidate paragraphs
  -> case-scoped BM25 bootstrap retrieval
  -> LangGraph planner
  -> dynamic ten-way fanout
  -> per-branch tool selection
  -> paragraph_search
  -> per-branch evidence reflection
  -> reducer
  -> HotpotQA answer and supporting-fact evaluation
```

The LangGraph runner sends every LLM stage through the OpenAI-compatible vLLM
server. The baseline uses `FLASH_ATTN`; the Agentrix variant uses `FORK_ATTN`.
Planner output, tool calls, retrieval results, branch outputs, and reducer
inputs propagate through the live graph.

“End to end” here refers to the complete LangGraph-to-Agentrix-vLLM inference
path. The experiment does not enable every optional Agentrix subsystem: data
parallel routing, tensor parallelism, KV offload, LMCache/CacheBlend,
multimodal inputs, and an external vector database are outside this run.

## Case Construction

The source is the official HotpotQA distractor development split. Its recorded
SHA-256 is:

```text
4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea
```

The generator ranks examples by tokenizer-measured context length and freezes
100 target IDs. Each target keeps its ten official candidate paragraphs and
adds three deterministic donor examples as realistic distractors. Duplicate
paragraph titles are removed when the case index is built. Gold answers and
supporting facts are retained only by the evaluator; they are not placed in
planner or branch prompts.

The committed manifest is
`benchmark/configs/hotpot_agentrix_long_prefix_100.jsonl`. It records the
target ID, donor IDs, question type, requested branch count, paragraph count,
and tokenizer-measured shared-context length.

This construction produces:

- 100 distinct target questions;
- 40 requested paragraphs per case before title deduplication;
- 7,871 to 14,311 shared-context tokens, with a mean of 11,236;
- exactly ten sibling branches per case;
- 1,000 total branches.

Because adjacent long examples are reused as donors, paragraph-level reuse
also exists across cases. Exact sibling request prefixes remain case-specific.

## Executed Configuration

| Setting | Value |
|---|---|
| GPU | NVIDIA RTX 5070, 12 GiB |
| Model | Qwen3-0.6B, BF16 |
| Backends | vLLM `FLASH_ATTN`; Agentrix `FORK_ATTN` |
| Cases | 100 |
| Concurrent cases | 2 |
| Branches per case | 10 |
| Client LLM concurrency | 20 |
| Bootstrap retrieval | up to 40 paragraphs |
| Bootstrap character cap | 100,000 |
| Artificial tool delay | zero, synchronized profile |
| Model context limit | 32,768 tokens |
| Maximum batched tokens | 16,384 |
| Maximum sequences | 32 |
| GPU memory utilization | 0.75 |
| Prefix caching | enabled for both backends |
| Async scheduling | disabled for both backends |
| Fork scheduling | Prefix Forest enabled |
| Fork CUDA Graph | Prefix Forest CUDA Graph enabled |
| Sampling | greedy; model thinking disabled |
| Offload / LMCache | disabled |

The same frozen manifest, request limits, warm-up procedure, concurrency, and
fresh-server lifecycle are used for both backends.

## Performance Results

Both variants completed all 100 cases, all 1,000 branches, and 3,300 graph
events without a request-level failure. Bootstrap retrieval contained both
gold supporting titles for every target case.

| Backend | Wall time | Speedup | Prompt tok/s | Request latency P50 | Request latency P95 | Prompt tokens |
|---|---:|---:|---:|---:|---:|---:|
| FlashAttention | 829.184 s | 1.00x | 31,881.6 | 2,423.1 ms | 8,707.5 ms | 26,435,716 |
| ForkAttention | 585.072 s | **1.42x** | 45,129.5 | 1,620.8 ms | 5,044.7 ms | 26,404,035 |

The prompt-volume difference is 31,681 tokens, or 0.12% of the FlashAttention
volume. ForkAttention physically activated on 22,400 of 28,163 measured steps
(79.5%), with 178,374 shared and 593,561 singleton CTA-plan entries.

| Backend | GPU after warm-up | GPU peak | Measured-phase GPU increment | KV capacity | Peak KV usage | Peak live KV tokens | Process-tree RSS peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| FlashAttention | 11,251 MiB | 11,730 MiB | 479 MiB | 65,024 | 99.4% | 64,656 | 3,824 MiB |
| ForkAttention | 10,796 MiB | 11,703 MiB | 907 MiB | 65,296 | 96.4% | 62,927 | 4,004 MiB |

ForkAttention reduced peak live KV by 1,729 tokens and total GPU peak by 27
MiB in this run, but used 428 MiB more transient GPU memory above its own
post-warm snapshot and about 180 MiB more process-tree RSS. Peak sampled memory
controller utilization was 91% for FlashAttention and 85% for ForkAttention.

The two backends exposed slightly different KV capacities, so live-token and
percentage values are both retained instead of treating the percentages as a
capacity-matched memory comparison. vLLM also preallocates most KV storage;
consequently the nearly equal `nvidia-smi` peaks do not imply equal live KV
occupancy. CPU offload and LMCache were disabled, so this experiment has no
offload-cache occupancy or KV transfer-volume result.

### HBM KV-read status

ForkAttention is expected to reduce HBM reads for the shared-prefix portion:
a shared CTA can load one K/V tile and apply it to multiple sibling queries,
whereas branch-local attention logically consumes that tile once per query.
For a perfectly grouped ten-way fanout, the ideal upper-bound reduction for
the shared-prefix component is `1 - 1/10 = 90%`. This is a logical kernel-level
bound, not a measured whole-model bandwidth reduction.

This run did not collect an Nsight Compute `dram__bytes_read.sum` equivalent,
so it does not provide a measured HBM KV-load byte count. The 79.5% physical
activation rate and shared-CTA counter prove that the reuse path executed, and
the sampled memory-controller peak changed from 91% to 85%, but neither metric
can be converted into KV bytes read. CUDA prefill, suffix attention, weights,
activations, L2 hits, and non-attention kernels also contribute to HBM traffic.

A quantitative follow-up must profile matched decode windows for both
backends, collect DRAM read bytes and L2 traffic per attention kernel, and
normalize them by completed decode token. Nsight Systems is useful for locating
the window; Nsight Compute is required for the byte-level claim.

## Execution and Quality Guardrails

Model-emitted tool calls were valid for 95.3% of FlashAttention selections and
95.2% of ForkAttention selections. A deterministic query fallback allowed all
branches to continue when Hermes rejected a truncated tool-call JSON object.
This fallback changes only tool-call robustness; the LLM selection request and
its attention work still execute normally.

The initial evaluator accepted only `[title, sentence_id]` arrays, while the
model commonly emitted `{ "title": ..., "sentence_id": ... }` objects. The
normalizer now accepts both forms. Saved reducer responses were reparsed rather
than regenerated.

| Metric | FlashAttention | ForkAttention |
|---|---:|---:|
| Answer EM | 0.110 | 0.100 |
| Answer F1 | 0.196 | 0.176 |
| Supporting-fact F1 | 0.064 | 0.062 |
| Joint F1 | 0.024 | 0.022 |

The low semantic scores reflect the small model and deliberately inflated
distractor set. They are similar enough to serve as an execution guardrail,
but this positive control must not be presented as a HotpotQA quality result.

## Interpretation

This run demonstrates that the current Agentrix ForkAttention backend can
produce a substantial single-GPU end-to-end gain when the application exposes
its intended structure: an approximately 8K-to-14K shared root followed by ten
concurrent sibling branches. The 79.5% physical activation rate confirms that
the gain is associated with actual ForkAttention execution rather than only a
logical shared-prefix estimate.

The result does not establish that ForkAttention accelerates short prompts,
unrelated request streams, narrow fanouts, DP routing, offload, multimodal
models, or naturally interleaved production RAG traffic. Those require
separate controls.

The current Qwen3-0.6B 100-case GPU-only and two-level CPU-offload follow-up is
recorded separately in
[`offload_restart_experiment.md`](offload_restart_experiment.md).

## Reproduction

Generate or refresh the frozen manifest with portable paths:

```bash
benchmark/.venv/bin/python benchmark/scripts/build_hotpot_agentrix_cases.py \
  --hotpot-path /path/to/hotpot_dev_distractor_v1.json \
  --tokenizer /path/to/Qwen3-0.6B \
  --cases 100 --paragraphs 10 --context-groups 4 --branches 10 \
  --output benchmark/configs/hotpot_agentrix_long_prefix_100.jsonl
```

Run the matched backend comparison from the repository root:

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
HOTPOT_PATH=/path/to/hotpot_dev_distractor_v1.json \
HOTPOT_CASE_FILE="$PWD/benchmark/configs/hotpot_agentrix_long_prefix_100.jsonl" \
VLLM_BIN="$PWD/benchmark/.venv/bin/vllm" \
OUTPUT_ROOT="$PWD/benchmark/results/hotpot_agentrix_100_e2e" \
VARIANTS='baseline forkattention' \
bash benchmark/scripts/run_hotpot_agentrix_e2e.sh
```

The script defaults to the executed 100-case configuration. `MODEL_PATH` and
`HOTPOT_PATH` are mandatory so that no machine-specific path is embedded in
the repository.

Each variant directory contains `run.json`, server logs, Prometheus snapshots,
and GPU memory samples. The root output directory contains `comparison.json`
and `comparison.md`; reparsed quality metrics are stored as
`reparsed_evaluation.json` beside the corresponding raw run.
