# Agentrix application prompt compaction

This package removes representation-only prompt redundancy without rewriting
free-form text. It can omit empty sections, exact duplicates with the same
stable segment ID, canonical JSON whitespace, and byte-identical tool schemas.

For incremental Agent/RAG prompts, `compact_prompt_delta` also omits a new
section when the same ID and byte-identical rendered text already occur in an
earlier message. Reusing an ID for different content raises an error. The
operation preserves information, but it does not promise token-for-token model
output identity because removing a repeated passage changes its position.

```python
from agentrix_application import PromptSection, compact_prompt_delta

shared = [PromptSection("rag:doc-1", "body", "Document doc-1")]
private = [
    PromptSection("rag:doc-1", "body", "Document doc-1"),
    PromptSection("rag:doc-2", "new body", "Document doc-2"),
]
result = compact_prompt_delta(private, known_sections=shared)
assert result.text == "Document doc-2\nnew body"
```

## LangGraph integration and ablation

`benchmark/src/langgraph_runner.py --prompt-compaction` applies the incremental
operation to branch-local `rag_search` results. The shared bootstrap evidence
stays byte-for-byte unchanged, so sibling branches keep the same long parent
prefix. A branch-local chunk is omitted only if that exact chunk is already in
the bootstrap message. Each tool event records the section and character
counts, while vLLM usage records provide the actual tokenizer-level prompt
token count.

The executable ablation is:

```bash
CASES=20 CASE_CONCURRENCY=1 \
  benchmark/scripts/run_langgraph_prompt_compaction_ablation.sh
```

By default it runs the four Flash/Fork fresh-server variants over the same 20
tasks. CacheBlend is opt-in:

```bash
ENABLE_CACHEBLEND=1 CASES=20 CASE_CONCURRENCY=1 \
  benchmark/scripts/run_langgraph_prompt_compaction_ablation.sh
```

The opt-in run adds the final CacheBlend pair:

| Pair | Off | On | Live matched question |
|---|---|---|---|
| FlashAttention | `baseline` | `baseline_compact` | compaction without ForkAttention |
| ForkAttention | `forkattention` | `forkattention_compact` | compaction/Fork interaction |
| CacheBlend | `cacheblend` | `cacheblend_compact` | compaction/CacheBlend interaction |

All selected variants use the same task file, RAG corpus, token limits,
case-major admission, unrelated backend warm-up, and one fresh vLLM process.
CacheBlend is kept on its required FlashAttention/eager path and is not
combined with ForkAttention. Formal numbers should use at least three
repetitions with
alternating variant order; report median paired speedups, actual prompt-token
reduction, P50/P95 latency, valid tool-call and reducer completion rates,
ForkAttention physical counters, CacheBlend hit/retrieval counters, and reducer
lexical F1 as a drift guardrail. Lexical F1 is not a task-quality score, so a
material drop requires task-specific answer evaluation before claiming an
end-to-end win.

Memory is a first-class outcome, not inferred from prompt length. During the
measured interval the script samples total GPU memory, the complete vLLM
process-tree RSS, vLLM KV-cache utilization, and LMCache local/remote cache
bytes. The report derives peak live KV tokens from the fixed KV capacity and
the peak utilization gauge. It reports both GPU increment over the pre-server
idle snapshot and transient increment over the post-warm-up snapshot. This
distinction matters because vLLM preallocates its KV pool: compaction can lower
live KV occupancy without reducing `nvidia-smi` allocated VRAM.

Use two result layers. The six live variants are the real Agent workflow and
quality test; model-generated tool queries and branch answers are allowed to
affect later requests. For strict system attribution, capture one live trace,
construct its compacted reflection requests from the exact recorded chunk IDs,
and replay the raw/compacted pair against each backend. Only fixed-trace replay
may attribute a small latency or memory difference solely to compaction; live
results remain the stronger end-to-end relevance check.
