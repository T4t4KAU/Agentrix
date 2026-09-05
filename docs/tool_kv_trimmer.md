# Tool-call KV Cache Trimmer

## Executive Summary

`ToolKVTrimmer` is an application-owned policy that releases live GPU KV
blocks while a resumable vLLM session is idle waiting for a tool. It addresses
the tool-call KV white-occupancy interval without moving scheduling policy into
vLLM or coupling the application to vLLM's internal Python types.

The policy is disabled by default. When enabled, it:

1. waits for a fixed or learned soft TTL;
2. reads the current total vLLM KV-cache pressure;
3. trims only if pressure is above a configured threshold;
4. leaves request tokens and lifecycle state intact; and
5. lets vLLM use prefix cache or a KV connector on resume, with recomputation
   as the correctness fallback.

Historical Qwen3-0.6B measurements are retained through the
[historical index](historical_experiments.md#memory), not repeated as current
performance claims here. Releasing live KV returns blocks to vLLM's reserved
pool; it does not necessarily reduce physical memory shown by `nvidia-smi`.

## Problem

A stateful coding Agent commonly follows this sequence:

```text
prefill and decode
  -> emit tool call
  -> wait for filesystem / search / test / network tool
  -> append tool result
  -> resume decode
```

If the request remains resumable during the tool call, its KV blocks remain
live even though the GPU cannot use them until the tool returns. A cohort of
parallel agents can therefore retain several GiB of KV while producing no
tokens. New requests arriving during that interval see less effective KV
capacity and can be delayed or preempted.

Immediately evicting every session is also undesirable. Short tool calls
benefit from retaining hot KV, while trimming a long session introduces lookup,
reload, or recomputation work when it resumes. The policy must therefore trade
retention against expected tool duration and current global pressure.

## Design Goals and Non-goals

The design goals are:

- keep policy and learning in `application/`;
- keep the vLLM change limited to a narrow validated trim operation;
- protect fast tools with a grace interval;
- make low-pressure behavior a no-op;
- support a fixed TTL, predictor shadow mode, and active predicted TTL;
- preserve correctness if prediction, metrics, HTTP, or cache reuse fails;
- avoid retaining raw commands, paths, queries, or tool arguments;
- expose enough counters to tune the policy from measured workloads.

The module does not:

- shrink vLLM's startup CUDA allocation;
- trim active decode or ordinary non-resumable requests;
- delete request token history;
- guarantee a prefix-cache or CPU-cache hit on resume;
- decide ForkAttention operator routing;
- replace vLLM preemption, APC, or LMCache admission and eviction policy.

## Architecture and Ownership Boundary

```text
Application tool lifecycle
        |
        | tool_started(session_id, request_id, context)
        v
  ToolKVTrimmer -------------------- Tool TTL predictor
        |                             predict / observe
        | sleep soft TTL
        | serialize decisions
        | read KV pressure
        | pressure >= threshold
        v
  VLLMToolKVClient
        | GET /metrics
        | POST /v1/agentrix/tool-kv/trim
        v
  narrow vLLM hook
        | validate resumable + waiting state
        | free live request block references
        | retain token history and resume boundary
        v
  APC / KV connector lookup on resume
        |
        +-> hit: reuse or reload KV
        +-> miss: recompute preserved tokens
```

The application policy depends only on two asynchronous callables:

```python
PressureReader = Callable[[], Awaitable[float]]
TrimRequest = Callable[[str], Awaitable[Mapping[str, Any]]]
```

This makes the decision logic independently testable. The included HTTP
adapter is replaceable; the memory benchmark passes `AsyncLLM.trim_tool_kv`
directly without changing the trimmer.

## Application Implementation

The implementation is split into two files:

| File | Responsibility |
|---|---|
| [`application/src/agentrix_application/tool_kv_trimmer.py`](../application/src/agentrix_application/tool_kv_trimmer.py) | lifecycle, pressure gate, serialization, trim adapter, counters |
| [`application/src/agentrix_application/tool_ttl_predictor.py`](../application/src/agentrix_application/tool_ttl_predictor.py) | bounded features, online horizon model, persistence |

### Fixed-TTL Policy

For each tool lifecycle, `tool_started` creates one pending asynchronous
decision. With a fixed TTL, the decision is:

```text
sleep(grace_ms)
lock the pressure/read/trim transaction
read current kv_cache_usage
if pressure < pressure_threshold:
    keep KV resident
else:
    call trim(request_id)
    record released references and observed usage reduction
    optionally wait for the metric to refresh
unlock
```

If `tool_finished` arrives before the decision enters the trim call, the timer
is cancelled and the hot KV is retained. Once an HTTP trim has started it may
be running in an uncancellable worker thread, so `tool_finished` waits for it
instead of orphaning background work.

### Learned Soft TTL

`OnlineHorizonTTLPredictor` uses six online logistic classifiers to estimate:

```text
P(duration > 100 ms)    P(duration > 250 ms)
P(duration > 500 ms)    P(duration > 1,000 ms)
P(duration > 2,000 ms)  P(duration > 5,000 ms)
```

The bounded context is:

| Feature | Meaning |
|---|---|
| `tool_family` | coarse application-owned tool class |
| `argument_bytes` | encoded argument size, not argument content |
| `kv_tokens` | estimated live tokens for the session |
| `pressure` | total KV usage when the tool starts |
| `active_tool_sessions` | concurrent tool-wait cohort size |
| `shared_prefix_ratio` | optional fraction likely reusable elsewhere |
| `timeout_ms` | configured tool timeout |

Feature hashing bounds state at 128 values per horizon. Raw tool arguments are
never stored. Independently predicted probabilities are projected to a
non-increasing survival curve so that survival at a longer horizon cannot be
higher than survival at a shorter horizon.

The default model requires 50 completed observations. Before that point it
uses the fixed 500 ms fallback. After training, the longest horizon whose
survival probability reaches the confidence threshold controls the TTL:

```text
ttl = clamp(fallback_ttl * min_ttl / longest_confident_horizon,
            min_ttl,
            fallback_ttl)
```

The default bounds are 100-500 ms. Prediction can only shorten the fixed TTL;
it cannot extend retention beyond the fallback. A missing context, cold model,
or prediction exception also returns to the fixed TTL.

Every completed tool call supplies its observed duration to all six horizon
classifiers. The label is recorded whether or not the request was trimmed, so
the learner does not train only on policy-selected samples. State is a
versioned JSON object and can be saved and restored atomically with
`predictor.save(path)` and `OnlineHorizonTTLPredictor.load(path)`.

Predictor accuracy is evaluated separately from the memory experiment. A
controlled dataset contains 2,000 read, search, network, public-test, and build
events; each of five seeds uses a 70/30 held-out split:

| Predictor metric | Five-seed mean | Min-max |
|---|---:|---:|
| Macro six-horizon accuracy | 95.25% | 94.67%-95.78% |
| Macro Brier score, lower is better | 0.0349 | 0.0327-0.0370 |
| Shorten/fallback decision accuracy | 92.73% | 91.50%-93.67% |
| Exact TTL-bucket accuracy | 90.33% | 88.67%-91.83% |
| TTL within one adjacent bucket | 99.90% | 99.67%-100.00% |

The historical tool trace used in that evaluation contained only short
`paragraph_search` events, so its 100% fallback accuracy was not evidence of
cross-tool generalization. This is separate from the later TraceLab dataset.
Detailed historical calibration records are recoverable through the
[historical index](historical_experiments.md#memory).

### Shadow and Active Modes

There are three operational states:

| State | Trimming | TTL used | Predictor updates |
|---|---|---|---|
| Disabled | No | None | No |
| Fixed TTL / predictor shadow | Yes | `grace_ms` | Yes, if a predictor and context are supplied |
| Active predicted TTL | Yes | predicted, bounded by fixed fallback | Yes |

Shadow mode is important because it records the TTL the model would have used
while the fixed policy remains authoritative. Production activation should be
based on tool-family calibration, live KV savings, and resume cost, not only
classification accuracy.

### Lifecycle and Concurrency Safety

The pending map is keyed by application session ID and also stores the exact
vLLM request ID. This supports the following cases:

- a newer tool lifecycle supersedes an older timer for the same session;
- a late finish for an older request cannot cancel its replacement;
- fast completion cancels a grace or pressure-check phase;
- completion during an in-flight trim waits for a definitive result;
- shutdown cancels timers but drains in-flight trim operations.

Pressure decisions use one `_trim_lock`. This deliberately serializes the
sequence `read pressure -> decide -> trim`. Without it, a 16-request cohort
whose timers expire simultaneously could let every request act on the same
stale high-pressure value. After each trim the next session reads the updated
pressure and may keep its KV if the global threshold has already been met.

The policy rejects non-finite or out-of-range pressure values. Exceptions are
counted and contained in the background decision task rather than failing the
tool itself.

### Observability

`ToolKVTrimmerStats` exposes:

- started lifecycles, trim attempts, successful trims, and rejection reasons;
- pressure skips, grace cancellations, and cancellation before trim;
- finishes that overlap a trim;
- superseded calls and stale finish events;
- prediction, fallback, observation, and prediction-error counts;
- sum of predicted TTL values;
- released block references;
- sum of observed `kv_cache_usage_before - kv_cache_usage_after`;
- latest pressure and internal policy errors.

`released_block_references` must not be interpreted as unique physical blocks
when requests share APC blocks. Releasing one request decrements references;
the before/after total usage delta is authoritative for physical occupancy.

## Minimal vLLM Integration

The policy itself is outside vLLM. The vLLM branch adds only the mechanism
needed to trim a validated resumable request:

1. `POST /v1/agentrix/tool-kv/trim` accepts one `request_id`.
2. The engine resolves an external request ID to its internal request ID.
3. The scheduler accepts only a request that is both resumable and in
   `WAITING_FOR_STREAMING_REQ`.
4. It saves the computed-token boundary, frees the request's live KV block
   references, clears speculative tokens, and resets computed progress.
5. A preemption notification flushes model-runner and connector request state.
6. The next streaming update folds the prior computed output into the prompt,
   then performs normal cache lookup or recomputation.

Active, non-resumable, already-trimmed, missing, and zero-block requests are
rejected without mutation. The response reports:

```json
{
  "request_id": "request-123",
  "trimmed": true,
  "reason": "trimmed",
  "released_block_references": 256,
  "kv_cache_usage_before": 0.238,
  "kv_cache_usage_after": 0.178
}
```

Request token history remains the source of truth. A cache miss can therefore
cost additional compute, but cannot silently omit the pre-tool context.

## Relationship to ForkAttention and KV Offload

The trimmer is compatible with ForkAttention because the two mechanisms act
at different times and optimize different resources:

| Mechanism | State | Optimization |
|---|---|---|
| ForkAttention | sibling queries actively decoding | reuse shared resident KV reads across queries |
| Tool KV trimmer | session idle waiting for external input | release that session's live GPU KV references |

An idle trimmed session is not an active ForkAttention query, so there is no
operator dispatch conflict. When it resumes and rebuilds or reloads its KV, it
can participate in later ForkAttention cohorts normally. Trimming a request
does update prefix residency, so it should not be counted as a resident sibling
until resume completes.

Ordinary connector or fork-aware CPU/disk offload is also complementary. The
trimmer relinquishes the live GPU ownership; it does not delete a connector's
CPU or disk copy. Resume first tries the configured cache/connector path, then
recomputes on a miss. The scheduler sends a preemption-style notification so a
connector can flush stale per-request GPU state.

When several requests reference the same APC block, trimming one request only
removes its reference. The shared physical block stays resident as long as
another request or cache entry owns it. ForkAttention can continue to use that
block for the remaining active siblings; the trimmer neither double-frees it
nor claims memory savings that did not appear in total KV usage.

The trade-off is timing: an overly aggressive TTL can turn a short tool wait
into avoidable reload or recomputation. Pressure gating and shadow deployment
remain necessary even when offload is available.

## Configuration and Integration

The environment switches are:

| Variable | Default | Meaning |
|---|---:|---|
| `AGENTRIX_TOOL_KV_TRIM_ENABLED` | `0` | master switch |
| `AGENTRIX_TOOL_KV_TRIM_GRACE_MS` | `500` | fixed fallback TTL |
| `AGENTRIX_TOOL_KV_TRIM_PRESSURE_THRESHOLD` | `0.70` | minimum total KV usage for trim |
| `AGENTRIX_TOOL_KV_TRIM_POST_TRIM_RECHECK_MS` | `25` | metric refresh delay between serialized trims |
| `AGENTRIX_TOOL_KV_TRIM_USE_PREDICTED_TTL` | `0` | use predicted TTL instead of shadow-only prediction |

Recommended shadow configuration:

```bash
export AGENTRIX_TOOL_KV_TRIM_ENABLED=1
export AGENTRIX_TOOL_KV_TRIM_GRACE_MS=500
export AGENTRIX_TOOL_KV_TRIM_PRESSURE_THRESHOLD=0.70
export AGENTRIX_TOOL_KV_TRIM_POST_TRIM_RECHECK_MS=25
export AGENTRIX_TOOL_KV_TRIM_USE_PREDICTED_TTL=0
```

After validating predictions and resumption cost, activate learned TTL with:

```bash
export AGENTRIX_TOOL_KV_TRIM_USE_PREDICTED_TTL=1
```

Minimal application integration is:

```python
from agentrix_application import (
    OnlineHorizonTTLPredictor,
    ToolKVTrimmer,
    ToolTTLContext,
    VLLMToolKVClient,
)

client = VLLMToolKVClient("http://127.0.0.1:8000")
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
    await trimmer.tool_finished(
        session_id,
        vllm_request_id,
        duration_ms=measured_tool_duration_ms,
    )
```

This hook is useful only when the application keeps one vLLM streaming-input
session alive across the tool call. A conventional OpenAI request that ends at
the tool call already releases its request KV and has nothing to trim.

## Historical experiments and current status

Per-run Qwen3-0.6B, LangGraph, fanout-lifecycle and reload results have moved to
the [historical experiment index](historical_experiments.md#memory).
Those results do not validate the current CPU/Mooncake restore path, which is
paused after two TraceLab failures; see [KV memory status](kv_memory_optimization_status.md).

The standalone TTL predictor evaluates controlled synthetic tool-duration
samples, not production tool accuracy. Its mechanism and fallback are described
in “Learned Soft TTL” above. Use `benchmark/scripts/evaluate_tool_ttl_predictor.py --dataset synthetic` for
that controlled evaluation, or `--dataset trace --trace-root <results>` for
recorded tool events. Run both on the server, with a new `--output` path.

## Deployment Guidance

Use the following sequence for a real workload:

1. deploy fixed TTL with a conservative pressure threshold;
2. attach a predictor but keep predicted TTL in shadow mode;
3. stratify prediction error, live KV savings, and resume cost by tool family,
   KV length, cache hit source, and concurrency;
4. activate predicted TTL only for calibrated long-tool families;
5. retain the fixed TTL and recomputation path as fallbacks;
6. alert on rejection reasons, policy errors, recompute latency, and any drop
   in cache/connector hit rate.

The pressure threshold should reflect capacity risk. A low threshold maximizes
released KV but can add unnecessary resume work; a high threshold preserves
hot sessions until capacity is actually scarce. The preferred setting is the
one that improves admission and tail latency under the target agent workload,
not simply the setting with the lowest standalone KV curve.
