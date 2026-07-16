# Application Prompt Compaction

## Scope

Agentrix's `application` layer performs lossless compaction before a prompt is
sent to vLLM. The coding-agent benchmark enables it for both comparison arms:
FlashAttention with ordinary DP and ForkAttention with prefix-aware DP. It is
not an attention optimization and does not alter DP routing.

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

## Coding-Agent Integration

The generated case records include the rendered repository source sections
and their stable IDs. Forty-eight of the 128 model requests in each repository
receive a tool observation containing a source section that is already in the
30K parent. With compaction enabled, the runner retains the tool event and its
metadata but omits that repeated payload. Both DP variants receive the same
compacted request stream.

The formal H20 experiment uses compaction unconditionally; it does not add a
raw/compact performance arm. This keeps the primary comparison focused on
Flash ordinary DP versus Fork prefix-aware DP. The runner still records the
counterfactual duplicate bytes, removed-section count, and actual input-token
total so that compaction is auditable.

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

The implementation only deduplicates sections owned and identified by the
application. It does not search arbitrary conversation text for approximate
matches, compact model-generated reasoning, summarize large unique tool
outputs, or reclaim vLLM KV blocks by itself. Its saved-character counter is a
logical prompt saving, not HBM traffic. Prefix caching and prefix-aware DP are
still responsible for avoiding physical recomputation and misplaced KV state.

If future traces do not contain an exact repeated section, the module reports
zero removals and should be bypassed for that request. More aggressive
compression requires a separate accuracy evaluation and is outside this
lossless module.
