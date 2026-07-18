# Application Prompt Compaction

## Scope

Agentrix's `application` layer performs exact deduplication and optional
recoverable tool-result paging before a prompt is sent to vLLM. The completed
DP=4 coding-agent benchmark enabled exact deduplication for both comparison
arms: FlashAttention with ordinary DP and ForkAttention with prefix-aware DP.
The planned DP=8 full-system comparison instead keeps the vLLM/FlashAttention
baseline uncompressed and enables exact deduplication only in the optimized
ForkAttention arm. These transformations are not attention optimizations and
do not alter DP routing.

The current implementation targets a common coding-agent redundancy pattern:
a repository source section is already present in the frozen parent context,
then a search or read tool returns the same section again in a later turn. The
application should reference the existing section rather than append a second
byte-identical copy.

## Design

`PromptSection` gives every application-owned section a stable `segment_id`,
content, and optional heading. `compact_prompt_delta()` fingerprints the
rendered bytes of sections already in the conversation and of newly produced
tool sections.

- An exact identity and fingerprint match is omitted from the delta.
- An empty section is omitted.
- Reusing an identity with different content raises an error. The compactor
  fails closed instead of silently deleting changed data.
- Free-form model messages are never summarized or rewritten.
- Tool schemas can be canonicalized and exact duplicates removed separately.

This is deliberately narrower than semantic summarization. It is deterministic,
replayable, and cannot remove unique evidence merely because two passages look
similar.

## Conservative Tool-result Paging

The optional tool-result path extends exact section deduplication with one
deliberately narrow rule: an old, large, successful result from a recoverable
file-read tool may be replaced with a deterministic retrieval handle. It is
disabled by default.

The default eligibility conditions are all required:

- the message is a string-valued OpenAI-compatible `role=tool` result;
- its `tool_call_id` resolves to an earlier assistant tool call;
- the tool is `read` or `read_file`;
- the call contains a non-empty `path`, `file_path`, or `filename`;
- the result is at least 4,096 characters;
- at least four later user turns and one later assistant message exist;
- the result is not marked as an error and does not begin with a recognized
  error prefix; and
- the deterministic handle is strictly shorter than the original result.

Search, command, test, network, write, edit, structured/multimodal, orphaned,
recent, small, and error results are preserved. User and assistant natural
language is never changed. Tool names, arguments, IDs, message order, and the
tool-result envelope are also unchanged.

The replacement is a stable prefix followed by canonical JSON containing the
tool name, resource identity, original character and line counts, SHA-256, and
a re-read hint. The exact historical body is stored in
`ToolResultBackingStore` under its SHA-256. This matters when a file is later
modified or deleted: re-running a read returns current state, while the backing
store can still restore the exact historical observation.

```python
store = ToolResultBackingStore()
compacted = compact_tool_results(
    messages,
    config=ToolResultCompactionConfig(enabled=True),
    backing_store=store,
)
assert restore_tool_results(compacted.messages, store) == messages
```

`ToolResultCompactionReport` records the number of tool results seen and
compacted, before/after characters, every skip reason, and per-page resource,
hash, size, age, message index, and tool-call ID. These fields make every
mutation auditable without retaining a second copy inside the prompt.

### Safety and Validation Contract

"Almost lossless" is separated into claims that can and cannot be proven
mechanically:

1. **Structural preservation:** all messages and fields other than an eligible
   tool result's `content` must be byte/value identical.
2. **Exact recoverability:** restoring from the supplied backing store must
   reproduce the original message list exactly.
3. **Operational non-inferiority:** task success and hidden-test pass rates must
   be evaluated with paired raw/compacted model runs; text similarity alone is
   not evidence of equivalent reasoning.

The application test suite includes deterministic boundary cases, distinct
versions of the same resource, missing/tampered backing data, and 250 seeded
randomized message histories. Every generated history is required to satisfy
`restore(compact(messages)) == messages`, while non-stub messages must remain
identical.

Captured OpenAI-compatible sessions can be audited without model calls:

```bash
benchmark/.venv/bin/python \
  benchmark/scripts/evaluate_prompt_tool_result_compaction.py trace.json \
  --output report.json
```

The audit reports exact round-trip failures and a conservative potential-fault
rate: a later invocation of the same tool and resource after an eligible page
is counted as a potential fault. A zero observed rate is accompanied by its
95% binomial upper bound, so a small sample cannot be presented as evidence of
a rare-fault guarantee. Actual causal extra reads still require paired live
model runs.

### Prefix-cache Interaction

Rewriting old messages changes the token prefix and can invalidate APC or a
shared ForkAttention parent. A shared parent should therefore be compacted at
most once before the cohort forks, producing one stable compacted parent for
all branches. Active cohorts must not repeatedly rewrite that parent. Private
suffixes can be compacted independently, preferably by batching several
eligible results into one mutation.

### Real-source Context Reduction

The reproducible tokenizer experiment uses the first coding-agent case from
each of Django, FFmpeg, and SQLite as an unchanged shared parent. Qwen3-0.6B's
local chat template measured those parents at 30,020, 30,582, and 31,158
tokens. The suffix consists of 1, 2, 4, 8, or 16 reads from real, task-relevant
source files, rendered in `RepositoryTools.read` format with line numbers and
the first 400 lines. Four later user turns make every sufficiently large read
eligible under the default age policy.

The table reports the mean full-context reduction across the three
repositories; parentheses give the repository min-max range. The 4 KiB arm
matches the current executable coding-agent runner's default maximum tool
output. The 32 KiB arm represents applications that retain the
`RepositoryTools` class default.

| Reads | 4 KiB cap: full-context reduction | Mean tokens saved | 32 KiB cap: full-context reduction | Mean tokens saved |
|---:|---:|---:|---:|---:|
| 1 | 3.81% (3.44%-4.04%) | 1,226 | 11.76% (10.98%-13.09%) | 4,127 |
| 2 | 7.20% (6.59%-7.64%) | 2,413 | 21.45% (18.28%-25.07%) | 8,550 |
| 4 | 13.68% (11.81%-15.16%) | 4,999 | 35.84% (30.67%-40.46%) | 17,803 |
| 8 | 23.03% (20.24%-25.24%) | 9,673 | 51.56% (44.82%-57.04%) | 35,085 |
| 16 | 34.19% (30.50%-37.31%) | 18,177 | 66.40% (62.74%-72.54%) | 69,926 |

The tool suffix itself falls by 80%-86% in the 4 KiB arm and 93%-95% in the
32 KiB arm. All 30 matrix cells restored the original message list exactly.
At 16 reads, 45/48 results in the 4 KiB arm and 46/48 in the 32 KiB arm were
eligible; the remaining small files were preserved by the 4,096-character
floor.

This is a tokenizer measurement, not a model-quality experiment. It proves
logical context reduction and exact application-side recovery, but not task
non-inferiority or a causal reduction in model-initiated re-reads. The raw
matrix is in
`benchmark/results/investigation_20260718/prompt_tool_result_context.json` and
is generated by
`benchmark/scripts/benchmark_prompt_tool_result_context.py`.

## Coding-Agent Integration

The generated case records include the rendered repository source sections
and their stable IDs. Forty-eight of the 128 model requests in each repository
receive a tool observation containing a source section that is already in the
30K parent. With compaction enabled, the runner retains the tool event and its
metadata but omits that repeated payload. Both DP variants receive the same
compacted request stream.

The completed DP=4 H20 experiment used compaction unconditionally; it did not
add a raw/compact performance arm. The DP=8 experiment changes that policy:
its baseline retains the repeated tool sections, while its optimized arm omits
them. Both arms still record the counterfactual duplicate bytes,
removed-section count, and actual input-token total so that compaction is
auditable.

## Static Compaction Opportunity

| Repository | Exact repeated sections | Duplicate characters omitted |
|---|---:|---:|
| Django | 48 | 763,820 |
| SQLite | 48 | 547,975 |
| FFmpeg | 48 | 414,950 |

These counts prove that the module is active rather than a no-op. They are not
an isolated speedup measurement because no uncompressed runtime arm is run.
Actual storage is reported as request input tokens plus sampled GPU memory,
server process RSS, and KV-cache occupancy in the main coding-agent results.

The compacted three-round traces contained 3,841,511 Django, 4,056,004 SQLite,
and 3,957,765 FFmpeg input tokens. The counts were identical between the Flash
and Fork arms for each repository. This verifies A/B workload equality, while
the omitted-character counters quantify what the application did. It does not
quantify a standalone compression speedup; that would require the raw arm the
formal experiment intentionally omitted.

Application compaction also does not promise lower process allocation. In the
H20 run, Fork's peak aggregate GPU memory was 334.53 GiB versus 327.16 GiB for
Flash, and peak server RSS was roughly 15.9 versus 12.6 GiB. Those differences
belong to attention/backend execution, not the identical application policy.
The relevant compaction storage result is that 0.41-0.76 million duplicate
characters were not sent into later conversations.

## Limitations

Exact deduplication only applies to sections owned and identified by the
application. Conservative paging additionally handles old, large file-read
results, but does not search arbitrary conversation text for approximate
matches, compact model-generated reasoning, summarize content, page other tool
families by default, or reclaim vLLM KV blocks by itself. The in-memory backing
store must be persisted or replaced by a durable implementation if exact
restoration must survive an application restart. Saved-character counters are
logical prompt savings, not HBM traffic. Prefix caching and prefix-aware DP are
still responsible for avoiding physical recomputation and misplaced KV state.

If future traces do not contain an exact repeated section, the module reports
zero removals and should be bypassed for that request. More aggressive
compression requires a separate accuracy evaluation and is outside this
lossless module.
