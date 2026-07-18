# Agentrix application optimizations

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
CASES=100 CASE_CONCURRENCY=2 \
  benchmark/scripts/run_langgraph_prompt_compaction_ablation.sh
```

By default it runs the four Flash/Fork fresh-server variants over the same 100
frozen HotpotQA cases. CacheBlend is opt-in:

```bash
ENABLE_CACHEBLEND=1 CASES=100 CASE_CONCURRENCY=2 \
  benchmark/scripts/run_langgraph_prompt_compaction_ablation.sh
```

The opt-in run adds the final CacheBlend pair:

| Pair | Off | On | Live matched question |
|---|---|---|---|
| FlashAttention | `baseline` | `baseline_compact` | compaction without ForkAttention |
| ForkAttention | `forkattention` | `forkattention_compact` | compaction/Fork interaction |
| CacheBlend | `cacheblend` | `cacheblend_compact` | compaction/CacheBlend interaction |

All selected variants use the same HotpotQA manifest, donor contexts, token
limits, case admission, unrelated backend warm-up, and one fresh vLLM process.
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

## Tool-call KV trimming

`ToolKVTrimmer` is an application-owned, opt-in policy for releasing the GPU
KV blocks of a vLLM resumable session while a slow tool is running. It waits a
short grace period, samples vLLM's live KV usage, and calls the narrow trim
endpoint only when usage crosses the configured threshold. Fast tools and
low-pressure periods keep their hot KV untouched.

```bash
export AGENTRIX_TOOL_KV_TRIM_ENABLED=1
export AGENTRIX_TOOL_KV_TRIM_GRACE_MS=500
export AGENTRIX_TOOL_KV_TRIM_PRESSURE_THRESHOLD=0.70
export AGENTRIX_TOOL_KV_TRIM_POST_TRIM_RECHECK_MS=25
export AGENTRIX_TOOL_KV_TRIM_USE_PREDICTED_TTL=0  # shadow mode first
```

```python
from agentrix_application import ToolKVTrimmer, VLLMToolKVClient

client = VLLMToolKVClient("http://127.0.0.1:8000")
trimmer = ToolKVTrimmer(client.kv_cache_usage, client.trim)

# Call when a resumable generation session yields a tool call.
trimmer.tool_started(session_id, vllm_request_id)
try:
    tool_result = await run_tool()
finally:
    await trimmer.tool_finished(session_id, vllm_request_id)
```

Pressure decisions are serialized across sessions. After one successful trim,
the policy briefly allows vLLM's usage metric to refresh and then rechecks
pressure before trimming another session. This avoids a cohort of tool calls
all acting on the same stale high-pressure sample. Passing `vllm_request_id`
to `tool_finished` also prevents a late completion from an older tool call from
cancelling a newer lifecycle for the same session. Once an HTTP trim begins,
`tool_finished` waits for it instead of leaving an uncancellable worker thread
running in the background.

`trimmer.stats` exposes trim attempts and rejections, pressure skips,
superseded/stale lifecycle events, released block references, and the summed
observed drop in vLLM KV usage. These counters are intended to tune the grace
period and pressure threshold from measured workloads rather than assumptions.

### Learned soft TTL

`OnlineHorizonTTLPredictor` is a dependency-free online model that predicts the
probability that a tool will still be running after 100, 250, 500, 1,000,
2,000, and 5,000 ms. It hashes the tool family and argument-size bucket and
uses only bounded numerical context; raw tool arguments are never retained.

```python
from agentrix_application import (
    OnlineHorizonTTLPredictor,
    ToolKVTrimmer,
    ToolTTLContext,
)

predictor = OnlineHorizonTTLPredictor(min_training_samples=50)
trimmer = ToolKVTrimmer(
    client.kv_cache_usage,
    client.trim,
    ttl_predictor=predictor,
)

context = ToolTTLContext(
    tool_family="public_test",
    argument_bytes=len(encoded_arguments),
    kv_tokens=session_kv_tokens,
    pressure=last_kv_pressure,
    active_tool_sessions=active_tool_sessions,
    shared_prefix_ratio=shared_prefix_ratio,
    timeout_ms=tool_timeout_ms,
)
trimmer.tool_started(session_id, vllm_request_id, context)
try:
    tool_result = await run_tool()
finally:
    await trimmer.tool_finished(session_id, vllm_request_id)
```

When a predictor is supplied but
`AGENTRIX_TOOL_KV_TRIM_USE_PREDICTED_TTL=0`, predictions and observations are
collected in shadow mode while the fixed `grace_ms` remains authoritative.
After validation, setting the switch to `1` lets the model shorten the soft TTL
within its configured bounds. Cold start, missing context, and prediction
errors always fall back to the fixed TTL. Model state can be persisted with
`predictor.save(path)` and restored with `OnlineHorizonTTLPredictor.load(path)`.

The current OpenAI-compatible coding-agent runner creates an independent vLLM
request for every model turn, so those requests are already freed at turn end.
The trim hook intentionally accepts only `WAITING_FOR_STREAMING_REQ` sessions;
it is useful when the application keeps a vLLM streaming-input session alive
across the tool call. Resumption first tries the normal prefix/connector path
and otherwise recomputes the preserved prompt. This lowers *live* KV-block
occupancy, not the preallocated CUDA memory shown by `nvidia-smi`.
