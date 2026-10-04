# Agentrix application optimizations

## Session routing and selective KV backup

For an official `consistent_hash` router and a backend supporting selective
CPU offload, `session_kv_options` builds standard OpenAI SDK request options:

```python
from agentrix_application.session_kv import session_kv_options

options = session_kv_options("opaque-session-id", terminal=False)
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    **options,
)
```

Use the same opaque session identity across turns and retries. `terminal=True`
adds `kv_transfer_params.max_offload_tokens=0`, skipping this request's new CPU
backups. Set it only when the application knows before dispatch that the
request will not be reused; a tool call or the end of an individual branch does
not establish this. Unknown reuse keeps the backend's default policy.
The option neither deletes existing copies nor disables restoration, and does
not force device-memory release. Backend support and an enabled offload
connector are prerequisites; this adapter does not negotiate capabilities.
If existing SDK calls use `extra_body` or `extra_headers`, merge explicitly
without dropping other options. This is an opt-in mapping to official APIs,
not a new routing or eviction algorithm. Hardware evidence and its limits are
recorded in [the KV status document](../docs/kv_memory_optimization_status.md).

## Prompt representation

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

Old, large file-read results can also be replaced conservatively with stable
retrieval handles. This mode is disabled by default and never rewrites user or
assistant text, tool calls, errors, recent results, or non-read tools. Exact
historical bodies remain in an application-owned content-addressed store.

```python
from agentrix_application import (
    ToolResultBackingStore,
    ToolResultCompactionConfig,
    compact_tool_results,
    restore_tool_results,
)

store = ToolResultBackingStore()
result = compact_tool_results(
    messages,
    config=ToolResultCompactionConfig(enabled=True),
    backing_store=store,
)
assert restore_tool_results(result.messages, store) == messages
```

The default policy considers only successful `read` and `read_file` results
of at least 4,096 characters after four later user turns. A tool call must
contain a stable `path`, `file_path`, or `filename`; otherwise its result is
left untouched. The environment switch is:

```bash
export AGENTRIX_PROMPT_TOOL_RESULT_COMPACTION_ENABLED=1
export AGENTRIX_PROMPT_COMPACTION_MIN_RESULT_CHARS=4096
export AGENTRIX_PROMPT_COMPACTION_MIN_AGE_TURNS=4
export AGENTRIX_PROMPT_COMPACTION_RECOVERABLE_TOOLS=read,read_file
```

Rewriting old messages changes the token prefix and can invalidate APC or a
shared ForkAttention parent. Compact the shared parent once before branching;
keep it stable while the cohort is active, and compact private suffixes
independently. Exact restoration requires retaining the historical backing
store; persist it if restoration must survive an application restart. Exact
text recovery does not establish equivalent task quality after paging.

## Paged tool snapshots and branch lifetimes

`PagedToolStore` stores immutable tool results in SQLite, split into 4,096-character
pages. Agents receive handles and fetch a bounded range or search for a literal
string. Identical pages occupy one copy, including pages shared by different
versions of a result. A fork inherits references; a changed result creates a
separate immutable snapshot that reuses its unchanged pages. Releasing the final
owner deletes an object and reclaims pages that no other object references.
Retrieval reads the historical result even if the original file has changed.

```python
from pathlib import Path
from agentrix_application import PagedToolStore

store = PagedToolStore(Path("/tmp/tool-results.sqlite"), max_bytes=256 << 20)
store.open_session("parent")
handle = store.put("parent", large_tool_output)
store.open_session("branch", parent="parent")
store.release_session("parent")
try:
    evidence = store.search("branch", handle, "failed test", limit=1024)
finally:
    store.release_session("branch")
    store.close()
```

The quota and `stats()["stored_bytes"]` count unique UTF-8 page payload bytes,
excluding SQLite metadata and
journals. The SQLite page cache is bounded; the currently produced tool result
can still exist in application memory, and the OS can cache database pages.
Calls must be serialized by the owning thread. Live snapshots are never silently
evicted to meet the quota: a failed insertion rolls back without publishing a
handle. Independent sessions cannot retrieve one another's objects.

`put_stream(session, pieces)` accepts a string iterator without assembling the
whole result. `replace_range(session, handle, offset=..., delete_chars=...,
content=...)` creates a new revision using bounded buffers. Equal-length edits
and appends share unchanged pages; insertions that shift 4,096-character page
boundaries can require new pages throughout the suffix. Both original and new
handles remain valid. Streams must not mutate the store while yielding data.
Stream interruption or quota failure rolls back the unpublished revision. Old
databases are migrated transactionally while retaining their handles and owners.

Existing `put` callers gain page sharing automatically. Producers must use the
streaming/range-update API to avoid building full result strings in host memory.
The coding runner's file-read tool now streams selected content into the store.
Search and other tool producers still supply complete strings. Range updates
still scan and hash the complete revision to preserve its content-addressed handle.

To compare storage and host-memory use against an earlier implementation:

```bash
git show 52083f0:application/src/agentrix_application/prompt_compactor.py > /tmp/tool-store-before.py
application/.venv/bin/python benchmark/scripts/benchmark_tool_snapshot_sharing.py \
  --baseline-store /tmp/tool-store-before.py --size-mib 8 --branches 8 \
  --output /tmp/tool-snapshot-sharing.json
```

This controlled workload makes eight 32-character edits to independent branches
of an 8 MiB snapshot. Each arm runs in a fresh process on the same host, retains
all versions, verifies their full SHA-256 hashes and checks final reclamation.
The output separates stored payload, SQLite file size, process peak RSS and peak
Python allocations during branch updates. It measures application data, without
model inference; it provides no GPU KV or Agent quality result.

### Release intermediate data at workflow boundaries

`checkpoint_session` lets the workflow declare which results the next stage
still needs. It removes the current session's other references and catalog
entries. Other sessions retain their own ownership, so a delayed branch can
finish reading an intermediate after its parent has advanced. The data is
reclaimed when its last owner releases it.

```python
store.open_session("reviewer", parent="parent")
state = store.put("parent", next_stage_state)
store.checkpoint_session("parent", keep_results=[configuration_handle, state])
# The reviewer can still read its inherited historical results here.
store.release_session("reviewer")
```

The caller must supply the complete live set, including any history needed for
later recovery or auditing. A checkpoint can permanently delete unreferenced
data; context-budget eviction continues to preserve it for retrieval. Unknown
or unowned keep handles abort the checkpoint without releasing anything.
Observation sequence numbers remain monotonic after reclamation and reopening.
This policy is opt-in and does not infer expiry from age or model output.

The existing storage benchmark also compares session-end retention with stage
reclamation, with page sharing enabled in both arms:

```bash
application/.venv/bin/python benchmark/scripts/benchmark_tool_snapshot_sharing.py \
  --stages 16 --size-mib 2 --output /tmp/tool-stage-lifecycle.json
```

Each stage produces a unique 2 MiB report and a durable checksum state. A verifier
branch reads the complete report one stage later; future stages consume the
durable state. Once that verifier finishes, the raw report is explicitly no
longer required. Both variants execute all the same reads and produce the same
checksum chain. Occupancy samples include the overlap between consecutive
stages, and the benchmark checks that the final session release reclaims all
data. This is a scripted lifecycle test, not a model-quality evaluation or a
policy for discarding arbitrary conversation history.

The file-read tool also bounds memory when the selected lines themselves are
large. It scans the source once, spools the selected text with a 256 KiB memory
threshold, then streams the line-count header and body through the existing
snapshot API. Small results retain the normal inline representation. With paging
disabled, large results retain the same truncated response and full-content hash.
The temporary file can grow to the selected text size and is closed on success
or failure; configure the temporary directory on appropriate storage. OS file
caching is outside the process-buffer limit. This does not bound the accumulated
tool-event history or every tool producer.

To compare this path with an archived earlier tool implementation, run on the
experiment server:

```bash
application/.venv/bin/python benchmark/scripts/benchmark_tool_snapshot_sharing.py \
  --read-file "${EXPERIMENT_DIR}/long-result.txt" \
  --baseline-tools "${EXPERIMENT_DIR}/baseline/coding_agent_tools.py" \
  --start-line 1 --end-line 1 --page-reads --repeats 3 \
  --output "${RESULTS_DIR}/tool-read-streaming.json"
```

The benchmark compares exact response hashes, restores each complete paged
snapshot, checks final reclamation, and measures RSS before allocation tracing.
The [streaming results](../docs/agentrix_cross_platform_optimizations.md#大结果流式入库)
reports memory and latency separately.

The coding runner accepts `--tool-result-paging`. Large `read` and `search`
observations become handles instead of truncated output. The agent can issue
`read_result` and `search_result` actions. `list_results` exposes a paginated
catalog with original paths and snapshot order, so dropping old conversation
turns does not lose access to their handles. Run completion or failure releases
the owned snapshots. Other tools retain their existing behavior. This is application
data sharing; vLLM's native prefix sharing and KV Copy-on-Write remain the baseline.

### Read file ranges with bounded scan buffers

The coding runner's `read` tool scans files in 65,536-character blocks and keeps
only the selected lines. It preserves the exact text, line numbers, total line
count and historical-snapshot behavior. It still scans the entire file to count
lines; this reduces temporary host memory, not file I/O or GPU KV occupancy.
Memory grows with the selected text plus the scan buffer, so selecting a very
large line or result can still require substantial memory.

Compare the real tool on an existing file, with identical inputs and output:

```bash
git show 0c1d542:benchmark/src/coding_agent_tools.py > /tmp/tool-read-before.py
application/.venv/bin/python benchmark/scripts/benchmark_tool_snapshot_sharing.py \
  --read-file /path/to/large.log --start-line 1001 --end-line 1032 \
  --baseline-tools /tmp/tool-read-before.py --output /tmp/tool-read-memory.json
```

The comparison alternates fresh processes and verifies output hashes, sizes and
truncation flags. Process RSS and time are captured before allocation profiling;
a second call measures Python allocations separately. The result is specific to
this tool process and selected range, without model inference.

### Bound the cumulative tool working set

Limiting each retrieval does not bound a long conversation: retrieved results
can accumulate again. `record_observation` archives those results, and
`render_context` selects recent evidence within a cumulative token budget.
Pass the model's tokenizer; the bound includes serialized metadata, handles and
tool contents. Other messages, chat-template overhead and generated tokens need
their own allowance. This controls the tool portion of the prompt, not the size
of vLLM's preallocated pool.

```python
store.record_observation("branch", "read:src/example.py", tool_output)
context = store.render_context(
    "branch",
    count_tokens=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
    max_tokens=2048,
)
messages.append({"role": "user", "content": context["text"]})
```

Replace that context message before each request; do not append every rendered
version to the history. The agent needs access to `list_observations` and `read`
to recover archived evidence. A result larger than the budget contains an exact
prefix marked `partial`, with a `next_offset` for continuation. Original bodies
remain unchanged. Forks inherit the observation catalog and shared objects;
subsequent branch observations are private. Releasing a session also removes its
catalog. The coding runner's existing history policy is unchanged; this API is
currently exercised by the multi-stage benchmark below.

### Reproducible closed-loop screening

`benchmark/scripts/benchmark_tool_result_paging.py` generates seeded JSONL build
and check reports. Each case ingests two immutable reports, forks three auditing
branches, executes model-selected retrieval actions, and reduces their answers.
Both variants expose identical reports and retrieval tools. The oracle checks
exact revisions, scores, decisions, and the final selected job. This controlled
synthetic workload is **not** AgentX or a real coding-task quality evaluation.
Tools are optional: the inline agent may answer directly from its supplied
reports. The paged agent chooses which evidence to retrieve.
Both modes validate the public threshold rule against the model's own proposed
score. An inconsistent decision is rejected and retried within the same step
budget, without committing the rejected answer to history. The validator never
reads the hidden answer; wrong evidence still fails the oracle. All retries,
requests, and their time/token costs are included in the measurements.

With `--restore-on-validation-error`, a rejected score decision can restore its
source (the checks report) once, without copying unrelated build metadata. The
report is restored byte-for-byte within `--max-restore-chars`; the branch retains
its previous retrieved evidence and access to both tools. This is an explicit
workflow dependency, not an oracle-assisted answer repair. Restoration and retries
can increase the required context window and are included in the totals. The
coding runner currently exposes retrieval tools, but does not implement this
task-specific score validator or automatic restoration policy.

For a cumulative-budget ablation, run `--mode paged --context-rounds 6
--search-excerpt-chars 1536 --tokenizer /path/to/model` twice, with and without
`--tool-context-tokens 2048`. Leave full-report restoration disabled. Both arms
archive observations, use the same retrieval tools and audit the same jobs; the
last stage revisits the first job. The model chooses searches and reads, and the
oracle checks every stage. Compare sampled occupied KV blocks and task success,
alongside `peak_tool_context_tokens`; token counts alone are not GPU measurements.

For Qwen3-8B on an H100, launch from the repository root. Set `MODEL_DIR` and
`RESULTS_DIR` privately to the model and server results directories; the GPU index
and localhost port below are examples:

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 VLLM_SERVER_DEV_MODE=1 \
VLLM_LOG_STATS_INTERVAL=0.1 VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm/.venv/bin/vllm serve "${MODEL_DIR}/Qwen3-8B" \
  --served-model-name agentrix-paging --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --attention-config '{"backend":"FLASH_ATTN"}' \
  --kernel-config '{"enable_flashinfer_autotune":false}' --enforce-eager \
  --gpu-memory-utilization 0.9 --max-num-seqs 8 \
  --max-num-batched-tokens 1024 --max-model-len 14336 \
  --kv-cache-memory-bytes 2147483648 \
  --enable-prefix-caching --enable-prompt-tokens-details
```

After `/health` is ready, run `inline, paged, paged, inline`, saving each result
separately. For example:

```bash
vllm/.venv/bin/python benchmark/scripts/benchmark_tool_result_paging.py \
  --base-url http://127.0.0.1:8000 --gpu 1 \
  --mode inline --seed 20260923 --cases 8 --rows 128 --branches 3 \
  --concurrency 3 --kv-cache-bytes 2147483648 \
  --restore-on-validation-error \
  --server-pid SERVER_PID --output "${RESULTS_DIR}/tool-paging-inline.json"
```

Use the actual server PID for CPU process-tree RSS sampling. Each arm resets
prefix caches and records hardware, server limits, source/data hashes, per-request
tokens/timing, all model answers, errors, GPU memory samples, and final snapshot
ownership. Report full workflow correctness as well as individual branch accuracy.
The GPU number is sampled total board memory during the measured workload,
including other board users; it is not an allocator-level lifetime high-water mark.

To measure capacity within a fixed KV pool, keep serving limits identical and
compare `--workflow-concurrency 4 --concurrency 12` in both modes, with the
server's `--max-num-seqs` at least 12. The first flag overlaps whole workflows;
the second caps their combined model requests. The default remains one workflow.
KV usage is sampled separately from slower driver/RSS queries. Results include
occupied KV blocks, running/waiting requests, and preemptions. Occupied blocks
exclude reclaimable prefix-cache blocks; the byte estimate multiplies usage by
the configured pool budget and includes rounding error. Both are sampled peaks,
not exact lifetime maxima. Lower working-set occupancy can leave room for more
concurrent work even when the preallocated pool and board allocation stay fixed.

For a separate capacity experiment, restart on the **same GPU and model** with
`--kv-cache-memory-bytes 1073741824 --max-model-len 7168`. Run paged mode with the
matching benchmark budget, then try inline mode as a capacity control. Report
the changed serving limits explicitly: the smaller request window fits paged
contexts but can reject full reports. Lowering the KV budget is a native setting;
the contribution is completing the same data-dependent task with smaller inputs.
Fixed-pool comparisons isolate paging; capacity comparisons measure whether that
reduction permits lower physical GPU allocation. Keep model accuracy and errors
alongside memory and latency; fewer tokens alone do not establish an Agent win.
The prior pure-paging screening regressed in correctness, even when prompt tokens
fell sharply. Treat this as an opt-in experiment; neither smaller-model quality
nor real coding-task quality is established by the synthetic Qwen3-8B result.

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
| --- | --- | --- | --- |
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

The application policy below remains available, but the current pinned vLLM
checkout lacks its `/v1/agentrix/tool-kv/trim` endpoint and engine operation.
These examples describe the historical integration and do not establish a
working offload/prefetch path or measured benefit. See
[current status and limitations](../docs/kv_memory_optimization_status.md#agent-hints-与选择性备份).

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
