# Agentrix System Overview

This overview describes the application and backend integration paths.
Configuration and limitations are covered in
[KV memory](kv_memory_optimization_status.md) and [DP routing](dp_routing.md).

Current DP deployments use the official `vllm-router==0.1.15` package. The
private prefix/session router has been removed; older internal-routing
descriptions and measurements below are historical, not the current policy.
The [measured result inventory](README.md) is the authority for performance
claims. The LangGraph flow and compatibility matrix below describe a separate
historical workload, not the current official AgentX configuration. Repository
presence does not establish that an integration is connected or beneficial.

The current memory-management design is maintained in
[the lifecycle and branch plan](kv_memory_optimization_status.md#面向长生命周期与分支的统一方案).
It separates application object ownership from derived KV residency, reuses
official sharing and transfer mechanisms, and evaluates capacity and recovery
latency as well as throughput. The first new implementation bounds host buffers
for selected large file results; the complete lifecycle-driven KV controller
and heterogeneous migration remain planned work.

## Purpose

Agentrix is an inference system for Agent workloads with long shared context,
parallel reasoning branches, repeated RAG evidence, and KV pressure. It does
not treat every optimization as another attention kernel. The system acts at
three different representations:

| Layer | Unit optimized | Main mechanism |
| --- | --- | --- |
| Application | Prompt sections and tool schemas | Exact, information-preserving compaction |
| KV memory | Stored or transferred KV chunks | vLLM prefix cache, LMCache CPU/disk tiers, CacheBlend |
| GPU execution | Attention work over resident KV | ForkAttention, fanout scheduling, CUDA Graphs |

These mechanisms are complementary only when their compatibility constraints
are satisfied. Prompt compaction removes repeated input representation;
ForkAttention targets repeated GPU KV reads over shared segments; LMCache changes
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
TP coverage and physical execution
metrics.
Cache/session-aware DP routing now runs in the official external router.

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
- official external routing selects a replica using its configured cache or session policy;
- forest plans represent multiple shared roots and private suffixes;
- sparse plan-capacity buckets allow CUDA Graph replay without capturing every
  possible batch/bucket product;
- adaptive prefix splitting adds useful CTAs for small tail cohorts while
  preserving the wide-cohort tile path.

Physical counters distinguish logical prefix similarity from actual operator
use: observed steps, active shared-prefix steps, shared CTA entries, and
singleton CTA entries are exported through Prometheus.

### Other Runtime Coverage

The vLLM path includes upstream KV offload connectors, TP model coverage, and
official external cache/session-aware DP routing. Historical Agentrix native
fanout offload and hot-prefix protection options are not consumed by the current checkout.
Agent Hints for tool-wait offload and resume prefetch have not been connected.
The `llama.cpp/` submodule
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

The `LMCache/` submodule provides external KV storage infrastructure. Its current
local cache policy registry exposes LRU, LFU, FIFO, and MRU, not `FORK_AWARE`.
The historical fork-aware policy and HOT/COOLING/COLD lifecycle were implemented
on another branch; they are not active features of the pinned combination.
The current vLLM checkout also lacks the historical Agentrix residency and
placement hooks required by proactive backup. Remaining coordinator code or
old configuration files do not establish an operational path.

CPU and disk capacity, transfer traffic, allocation failures, and reload demand
must be measured independently from logical shared-tree savings. See
[the current code audit](kv_memory_optimization_status.md) before running an
offload recipe.

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
| --- | --- | --- | --- | --- | --- |
| Flash baseline | On | On | No | No | No |
| Flash + compaction | On | On | No | No | No |
| ForkAttention | On | On | Yes | No in current LangGraph run | No |
| ForkAttention + compaction | On | On | Yes | No in current LangGraph run | No |
| CacheBlend (opt-in) | Off | Eager | No | 8 GiB local CPU | Yes |
| CacheBlend + compaction (opt-in) | Off | Eager | No | 8 GiB local CPU | Yes |

Older benchmark recipes also describe ForkAttention with ordinary or
fork-aware CPU/disk offload. The fork-aware combination is unavailable in the
current pinned checkout; this historical matrix is not a compatibility claim
for the current Qwen3.5/AgentX serving configuration.

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
