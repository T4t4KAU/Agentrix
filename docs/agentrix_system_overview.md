# Agentrix System Overview

## Purpose

Agentrix is an inference system for Agent workloads with long shared context,
parallel reasoning branches, repeated RAG evidence, and KV pressure. It does
not treat every optimization as another attention kernel. The system acts at
three different representations:

| Layer | Unit optimized | Main mechanism |
|---|---|---|
| Application | Prompt sections and tool schemas | Exact, information-preserving compaction |
| KV memory | Stored or transferred KV chunks | vLLM prefix cache, LMCache CPU/disk tiers, CacheBlend |
| GPU execution | Attention work over resident KV | ForkAttention, fanout scheduling, CUDA Graphs |

These mechanisms are complementary only when their compatibility constraints
are satisfied. Prompt compaction removes repeated input representation;
ForkAttention reduces repeated GPU KV reads and attention work; LMCache changes
where KV is resident; CacheBlend reuses KV for reordered RAG chunks. Their
speedups must not be multiplied without a measured combined path.

## End-to-End Data Path

```text
LangGraph case
  retrieve shared RAG evidence
  -> planner/common analysis
  -> 16 parallel tool-selection branches
  -> local RAG tool results
  -> exact application compaction of repeated branch-local chunks
  -> branch reflection
  -> reducer
        |
        v
Agentrix vLLM OpenAI-compatible server
  scheduler / prefix-aware admission / APC
        |
        +-> FLASH_ATTN general path
        |
        +-> FORK_ATTN shared-prefix decode path
        |
        +-> LMCache connector -> CPU/disk KV tiers or CacheBlend
```

The LangGraph runner uses Agentrix's vLLM server as the inference backend. It
does not emulate LLM latency: planner, tool selection, branch reflection, and
reducer calls all go through the live OpenAI-compatible API. Local RAG is a
real deterministic BM25 index over a manifest-scoped, content-versioned
documentation corpus.

## Application Prompt Compaction

The `application/` package owns prompt transformations that can be audited
without model-specific heuristics. Its core types and functions are in
`application/src/agentrix_application/prompt_compactor.py`.

The compactor can:

- omit empty application-owned sections;
- omit a repeated stable segment ID only when heading and content are
  byte-for-byte identical;
- canonicalize JSON by removing representation-only whitespace;
- remove canonical-equivalent tool definitions;
- reject a reused segment or tool identity with conflicting content.

It does not summarize, paraphrase, truncate, reorder, or fuzzy-match free-form
text. This is the precise meaning of information-preserving compression in
Agentrix. It does not imply identical model output: removing the second copy of
an already-visible passage changes positional emphasis, so downstream output
drift is measured separately.

In the LangGraph RAG integration, every local chunk has a stable identity:

```text
rag:<relative source path>:<character offset>:<content hash>
```

The long bootstrap evidence remains unchanged and is inherited by every
branch. A later branch tool result is compacted only against that already
visible bootstrap. Therefore compaction shortens the private suffix without
weakening the exact shared parent that ForkAttention needs.

## vLLM Inference Acceleration

The `vllm/` submodule contains the primary high-performance inference path.
The current ForkAttention branch includes the CUDA backend, forest planning,
fanout-aware scheduling, CUDA Graph dispatch, adaptive tail splitting,
prefix-aware data-parallel routing, TP coverage, and physical execution
metrics.

### ForkAttention

FlashAttention independently reads the shared prefix for each active query.
ForkAttention creates a plan over the physical KV block table and lets one CTA
serve multiple sibling queries for a shared segment, followed by split-output
gather/merge. It targets single-token causal decode, not prefill.

ForkAttention is useful when all of the following are true:

- multiple sibling requests are concurrently decoding;
- they reference the same resident physical KV blocks;
- a 4K-8K or longer prefix dominates their private suffix;
- admission keeps the cohort together long enough to amortize planning and
  gather work;
- the shape is supported by the specialized backend.

Unsupported or weak shapes retain the FlashAttention path. A long textual
prefix alone is insufficient if it is evicted, recomputed, staggered, or split
across unrelated batches.

### Scheduling and CUDA Graphs

Agentrix aligns work before executing it:

- fanout admission groups sibling branches;
- prefix-aware DP first constrains rank load, then applies prefix affinity and
  query aggregation;
- forest plans represent multiple shared roots and private suffixes;
- sparse plan-capacity buckets allow CUDA Graph replay without capturing every
  possible batch/bucket product;
- adaptive prefix splitting adds useful CTAs for small tail cohorts while
  preserving the wide-cohort tile path.

Physical counters distinguish logical prefix similarity from actual operator
use: observed steps, active shared-prefix steps, shared CTA entries, and
singleton CTA entries are exported through Prometheus.

### Other Runtime Coverage

The vLLM path includes native CPU KV offload controls, hot-prefix protection,
TP model coverage, and prefix-aware internal DP. The `llama.cpp/` submodule
provides narrower ForkAttention implementations for CUDA, MUSA, and Apple
Metal portability. The LangGraph experiment in this document set uses vLLM;
llama.cpp is not part of its measured serving path.

## KV Memory Management

### vLLM GPU KV Cache

vLLM reserves a fixed GPU KV pool at startup. APC lets requests share physical
prefix blocks and avoids recomputing exact prefix tokens. This has two
important consequences:

1. lower live KV use does not necessarily lower `nvidia-smi` allocated VRAM;
2. ForkAttention does not claim another physical copy reduction on top of APC;
   it reduces repeated reads and attention work over the shared blocks.

Agentrix therefore reports both fixed allocation and the peak
`vllm:kv_cache_usage_perc`, converted to peak live KV tokens.

### LMCache Tiered Storage

The `LMCache/` submodule extends KV residency beyond the vLLM GPU pool. Its
Agentrix branch supports:

- local CPU KV storage;
- optional disk storage;
- default LRU and `FORK_AWARE` admission/eviction;
- HOT/COOLING/COLD fanout-prefix lifecycle hysteresis;
- CPU-to-disk demotion on eviction instead of mandatory write-through;
- guarded reload and DP handoff integration.

`FORK_AWARE` values a high-fanout shared prefix above low-value private suffix
chunks while retaining emergency eviction for correctness. CPU and disk
capacity, transfer traffic, allocation failures, and reload demand must be
measured independently from logical shared-tree savings.

### CacheBlend for RAG

CacheBlend addresses a different reuse pattern: a new RAG prompt may contain
previously cached document chunks in a different order. Stable separators
identify document segments, LMCache retrieves their KV, and selective
recomputation repairs cross-chunk attention state instead of blindly reusing
stale positions.

The current measured CacheBlend path has strict constraints:

- FlashAttention only; the layerwise blender does not accept
  `ForkAttentionImpl`;
- eager execution;
- vLLM APC disabled to prevent overlapping partial-hit ownership;
- `add_special_tokens=False` separator tokenization for Qwen3;
- application compaction can be enabled, but it removes some repeated text
  that CacheBlend might otherwise retrieve.

CacheBlend is therefore a separate serving variant, not a switch added to the
same CUDA-Graph ForkAttention process.

It is disabled by default because the current host experiment measured a
performance and host-memory regression. Benchmark launchers require the
explicit opt-in `ENABLE_CACHEBLEND=1`; without it they retain APC/CUDA Graphs
and do not load the CacheBlend LMCache configuration or connector.

## Compatibility Matrix

| Path | APC | CUDA Graph | ForkAttention | LMCache CPU/disk | CacheBlend |
|---|---|---|---|---|---|
| Flash baseline | On | On | No | No | No |
| Flash + compaction | On | On | No | No | No |
| ForkAttention | On | On | Yes | No in current LangGraph run | No |
| ForkAttention + compaction | On | On | Yes | No in current LangGraph run | No |
| CacheBlend (opt-in) | Off | Eager | No | 8 GiB local CPU | Yes |
| CacheBlend + compaction (opt-in) | Off | Eager | No | 8 GiB local CPU | Yes |

The broader repository also supports ForkAttention with ordinary or
fork-aware CPU/disk offload, but that is a different experiment matrix from
the CacheBlend RAG path.

## Observability and Decision Rule

Agentrix records four kinds of evidence:

- application: input/output sections, exact duplicates, characters removed,
  tokenizer-reported prompt tokens;
- serving: wall time, request latency P50/P95, prompt/completion volume, tool
  and reducer completion;
- execution: ForkAttention observed/active steps and CTA plan entries,
  CacheBlend lookup hits and retrieved tokens;
- memory: total GPU allocation, post-warm transient GPU allocation, peak live
  KV tokens, process-tree RSS sum, and LMCache gauges when exposed.

Routing should follow the workload, not a global backend preference:

- ordinary chat, prefill-heavy, short, or unrelated traffic -> FlashAttention;
- synchronized long-prefix multi-branch decode -> ForkAttention;
- reordered repeated RAG chunks with enough compute to amortize CPU transfer
  and eager selective recomputation -> evaluate CacheBlend;
- repeated application-owned sections already present in history -> exact
  compaction, subject to output-quality guardrails.

Current end-to-end results and limitations are in
[`langgraph_end_to_end_experiment.md`](langgraph_end_to_end_experiment.md).
The 20-case construction is specified in
[`langgraph_case_design.md`](langgraph_case_design.md).
