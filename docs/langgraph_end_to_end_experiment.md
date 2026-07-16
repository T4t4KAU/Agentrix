# LangGraph End-to-End and Ablation Experiment

## Scope

This experiment evaluates the complete Agentrix request path rather than an
isolated attention kernel. Twenty live LangGraph cases execute retrieval,
planning, model-generated tool calls, 16-way parallel branch reasoning, and a
reducer against Agentrix's vLLM server. The matrix measures attention,
application compaction, and CacheBlend paths with performance, memory,
physical-execution, and output-drift guardrails.

This is the latest six-variant run from 2026-07-15. It supersedes the earlier
two-variant case-major snapshot for compaction and memory analysis; that
historical result remains useful as an independent ForkAttention replication.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070, 12,227 MiB, 48 SMs |
| Driver | 590.48.01 |
| CUDA toolkit | 13.1 |
| PyTorch | 2.11.0+cu130, CUDA runtime 13.0 |
| Model | Qwen3-0.6B, BF16 |
| LangGraph / OpenAI client | 1.2.9 / 2.45.0 |
| Agentrix parent | `a76a763` plus the changes documented here |
| vLLM | `e139c755b` (`fork-attn`) plus CacheBlend model registration |
| LMCache | `768bd187` (`fork-attn`) plus Qwen3 separator/pin fixes |
| Date | 2026-07-15, Asia/Shanghai |

The server uses `max_model_len=32768`, `max_num_batched_tokens=16384`,
`max_num_seqs=32`, GPU utilization 0.75, disabled async scheduling, Hermes
tool parsing, greedy temperature 0, and Qwen thinking disabled. Flash and Fork
use APC and CUDA Graphs. CacheBlend uses FlashAttention, eager execution,
disabled APC, an 8 GiB local CPU cache, 256-token chunks, a 0.15 selective
recompute ratio, and a 512-token blending threshold.

## Workload and Fairness

Every case contains 24 bootstrap RAG chunks capped at 32,000 characters and 16
role-specific branches. The actual bootstrap sizes range from 28,816 to 31,798
characters. Only one independent case is active at a time, while its 16
sibling branches execute concurrently. This is case-major Agent fanout, not a
round-robin batch of unrelated prompts.

Across a variant, the graph produces:

- 20 bootstrap retrievals;
- 20 planner requests;
- 320 model tool-selection requests;
- 320 local RAG tool executions;
- 320 branch-reflection requests;
- 20 reducers.

That is 680 live LLM requests and 340 real tool events, or 1,020 recorded
events. The runner uses the same task file, frozen 101-chunk corpus, token
budgets, branch roles, and unrelated backend warm-up for all variants. Each
variant starts a fresh server. The corpus manifest preserves version
`6f2b83c2680700dd` for this formal run. The manifest freezes file membership,
and the runner verifies a content-addressed corpus version. The current
post-documentation corpus is `b01c0c10bb027921`; this deliberate version
change prevents later runs from being mistaken for byte-identical corpus
replications.

The 480 bootstrap chunk occurrences contain 83 unique chunks. Reusable
bootstrap characters are 82.81%, mean pairwise Jaccard overlap is 0.207, and
184 case pairs share at least two chunks in a different order. This provides
both case-local exact-prefix reuse for ForkAttention and cross-case reordered
document reuse for CacheBlend.

The experiment is live, not fixed-trace replay. A model-generated query changes
the tool result, branch response, and reducer input. This is the correct
end-to-end relevance test but means small within-backend timing differences
cannot be attributed solely to compaction. A fixed-trace replay remains
required for a pure serving attribution claim.

## Variants

| Variant | Attention/runtime | Application compaction | External KV path |
|---|---|---|---|
| Baseline | FlashAttention, APC, CUDA Graph | Off | None |
| Baseline compact | FlashAttention, APC, CUDA Graph | On | None |
| ForkAttention | ForkAttention, APC, forest CUDA Graph | Off | None |
| ForkAttention compact | ForkAttention, APC, forest CUDA Graph | On | None |
| CacheBlend | FlashAttention, eager, APC off | Off | LMCache local CPU + blend |
| CacheBlend compact | FlashAttention, eager, APC off | On | LMCache local CPU + blend |

ForkAttention and CacheBlend are deliberately separate. The current
layerwise blender rejects ForkAttention, so no combined or multiplied speedup
is reported.

## End-to-End Performance

| Variant | Wall time | Speedup vs baseline | Prompt tokens | Completion tokens | Latency P50 | Latency P95 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 258.456 s | 1.00x | 6,635,857 | 90,518 | 1,450.5 ms | 8,969.1 ms |
| Baseline compact | 256.930 s | 1.01x | 6,487,557 | 92,153 | 1,508.2 ms | 8,724.4 ms |
| ForkAttention | 164.141 s | **1.57x** | 6,635,205 | 91,064 | 859.7 ms | 4,281.3 ms |
| ForkAttention compact | 143.021 s | **1.81x** | 6,481,017 | 91,142 | 852.7 ms | 3,175.8 ms |
| CacheBlend | 498.500 s | 0.52x | 6,673,996 | 88,844 | 3,514.8 ms | 15,040.8 ms |
| CacheBlend compact | 483.363 s | 0.53x | 6,489,174 | 90,672 | 3,568.2 ms | 14,169.4 ms |

ForkAttention reduces wall time by 36.5% and cuts P95 request latency by
52.3% relative to the matched Flash baseline. The compact Fork live run is
44.7% faster than baseline. Completion volume differs by less than 1% between
baseline and either Fork variant, so output length does not explain the main
ForkAttention gain.

## Application Compaction Ablation

| Matched path | Exact chunks removed | Characters removed | Prompt-token reduction | Live wall speedup | Reducer lexical F1 vs uncompacted |
|---|---:|---:|---:|---:|---:|
| Flash | 419 / 595 | 545,976 | 2.23% | 1.006x | 0.564 |
| ForkAttention | 435 / 595 | 566,726 | 2.32% | 1.148x | 0.560 |
| CacheBlend | 518 / 707 | 675,781 | 2.77% | 1.031x | 0.487 |

The compactor removed only byte-identical chunks that were already visible in
the bootstrap context. The large character reduction becomes a 2.2%-2.8%
whole-run token reduction because planner, tool selection, shared bootstrap,
generated output, and reducer prompts are unchanged.

The 1.148x Fork live difference is promising but is not yet an isolated
operator claim. The uncompacted and compact live runs can generate different
queries and later prompts, as reflected by the lexical F1. The high-confidence
compaction conclusions from this run are exact section removal, lower prompt
volume, and lower live KV occupancy. A fixed-request replay and repetitions
with alternating order are needed before publishing 1.148x as a standalone
compactor speedup.

## Memory Results

| Variant | GPU peak used | Peak live KV | KV peak | Process-tree RSS sum | Post-warm transient GPU peak |
|---|---:|---:|---:|---:|---:|
| Baseline | 11,639 MiB | 31,784 tokens | 48.6% | 3,812 MiB | 636 MiB |
| Baseline compact | 11,724 MiB | 27,127 tokens | 41.6% | 3,816 MiB | 773 MiB |
| ForkAttention | 11,624 MiB | 31,016 tokens | 47.5% | 3,971 MiB | 449 MiB |
| ForkAttention compact | 11,728 MiB | 25,366 tokens | 38.9% | 3,984 MiB | 785 MiB |
| CacheBlend | 11,697 MiB | 65,248 tokens | 99.9% | 12,313 MiB | 226 MiB |
| CacheBlend compact | 11,369 MiB | 65,248 tokens | 99.9% | 12,297 MiB | 257 MiB |

Compaction lowers peak live KV tokens by 14.7% on Flash and 18.2% on
ForkAttention. Total GPU allocation does not fall because vLLM preallocates
the KV pool; small differences in `nvidia-smi` peaks are not evidence of
capacity savings. The live-occupancy gauge is the relevant physical measure.

CacheBlend fills the vLLM KV pool while also reserving an 8 GiB CPU cache. Its
process-tree RSS sum reaches about 12.3 GiB, roughly 8.3 GiB above the
non-LMCache paths. RSS sum can double-count shared pages, so it is a comparative
process-tree measure rather than unique host PSS. The LMCache local-cache gauge
was not exposed through this vLLM `/metrics` endpoint; it is reported as
unavailable, not as zero occupancy.

Memory was sampled every 0.5 seconds. GPU peak includes other display/system
allocation; the runner also records a pre-server idle snapshot and a
post-warm-up snapshot. The table avoids treating NVIDIA
`utilization.memory` as allocated memory because it is a memory-controller
activity proxy.

## Physical Execution and Cache Evidence

| Evidence | Uncompacted | Compacted |
|---|---:|---:|
| Fork observed steps | 12,218 | 12,185 |
| Fork active shared steps | 5,679 (46.5%) | 5,717 (46.9%) |
| Fork shared CTA entries | 19,096 | 19,238 |
| Fork singleton CTA entries | 86,927 | 86,580 |
| CacheBlend lookup hit ratio | 90.3% | 92.7% |
| CacheBlend retrieved tokens | 5,938,518 | 5,966,642 |
| LMCache negative pin warnings | 0 | 0 |

The Fork counters are measured deltas after warm-up, so they prove that the
specialized path physically executed. CacheBlend also achieves a high lookup
hit rate, but reuse is not automatically beneficial: on this small 0.6B model,
CPU KV movement, eager execution, and selective recomputation cost more than
the avoided prefill. CacheBlend takes 1.93x the baseline wall time despite the
90.3% hit ratio. This is a valuable negative result, not evidence that the
lookup path was inactive.

## Quality and Validity

All six variants produced 320/320 valid `rag_search` calls and 20/20 non-empty
reducers. Reducer bag-of-words F1 is 0.564 for Flash compaction versus Flash,
0.560 for Fork compaction versus Fork, and 0.487 for CacheBlend compaction
versus CacheBlend.

Lexical F1 is a drift alarm, not a semantic task score. Qwen3-0.6B outputs can
change phrasing across live runs, and removing repeated evidence changes
position even when no information is removed. The result therefore supports
workflow completion and information-preserving input transformation, but not
a claim of identical answers or improved task accuracy. Task-specific rubrics
or a stronger judge are required before a quality claim.

## Failure Found During the Matrix

The first CacheBlend startup used APC and failed during warm-up. APC had
already computed 5,632 tokens while LMCache reported a 5,646-token hit, leaving
the blender with incompatible Q/K lengths of 5,646 and 13. The corrected
CacheBlend path disables APC, matching its independent experiment setup. Both
corrected CacheBlend variants then completed all requests without pin warnings.

This failure establishes a deployment constraint: CacheBlend and APC cannot
be enabled together on the current layerwise path merely because both are
individually valid cache mechanisms.

## Reproduction

The corpus manifest excludes later result documents, and the content-version
gate fails closed if any included source changes. Run the six live variants
on the current `b01c0c10bb027921` corpus with:

```bash
CASES=20 CASE_CONCURRENCY=1 CONCURRENCY=16 \
MEMORY_SAMPLE_INTERVAL=0.5 \
ENABLE_CACHEBLEND=1 \
OUTPUT_ROOT=/tmp/agentrix_langgraph_full_ablation \
benchmark/scripts/run_langgraph_prompt_compaction_ablation.sh
```

CacheBlend is now disabled by default. Omitting `ENABLE_CACHEBLEND=1` runs only
the four Flash/Fork compaction variants and never loads its LMCache connector.

Each formal publication should add at least three repetitions, alternate
variant order, and report paired medians. Use live runs for Agent relevance and
quality. Use a raw/compacted fixed trace for strict serving attribution.

Only the valuable aggregate results are committed. Request JSON, telemetry
CSV, and server logs from this run were temporary validation artifacts and are
not repository results.
