# TraceLab observed-time replay

## Protocol

This is the current observed-time protocol. It supersedes the synthetic 100 ms
session arrivals and summed-tool-wait loop documented in [the legacy record](#legacy-and-failures).
The legacy result remains a separate experiment, not a real-timeline baseline.
Inference, tests and profiling run only on `connect.bjb2.seetacloud.com`.
CPU/Mooncake remains excluded while its restore failure is unresolved.

## First completed comparison — 2026-09-05

Both groups completed all 132 requests with identical workload and benchmark
source hashes. One valid run per group was performed. **Agentrix GPU-only was
slower in this experiment**; the earlier closed-loop gain does not carry over.

| Metric | Official vLLM 0.25.0 | Agentrix GPU-only |
| --- | ---: | ---: |
| Completion time | 218.23 s | 259.79 s |
| Output throughput | 189.99 tokens/s | 159.59 tokens/s |
| TTFT p50 from intended arrival | 32.40 s | 53.90 s |
| TTFT p95 from intended arrival | 88.32 s | 127.12 s |
| Client submission lag p95 / max | 102.64 / 108.97 ms | 103.02 / 109.43 ms |
| Peak in-flight requests | 83 | 92 |
| Peak distinct in-flight sessions | 72 | 77 |
| Sampled peak running requests, both GPUs combined | 9 | 9 |
| Sampled peak queued requests, both GPUs combined | 71 | 84 |
| Scheduler preemptions, counter delta | 0 | 34 |

Completion time increased by 19.05%, throughput fell by 16.00%, and arrival-based
TTFT p95 increased by 43.93%. These are single-trial observations, not confidence
intervals. The two latency numbers include measured client lateness; no
millisecond-exact arrival claim is made. Source intervals remain unscaled, but
the actual submission jitter is approximately 0.1 s at p95 in both groups.

Agentrix's 34 preemptions all occurred on replica 0; upstream had none on either
replica. This and the larger queue are concrete follow-up signals, **not proof
of a unique root cause** or attribution to a particular optimization. No inference
engine changes or corrective ablations were made during this comparison.

Reported cached tokens increased from 85,472 to 226,256, but 165,024 of Agentrix's
reported cached tokens were on initial window requests, versus zero upstream.
With preemption, this counter must not be interpreted as a clean measure of
cross-round/shared-branch reuse: resumes can reuse a request's own cached work.
Both external-KV-transfer deltas were zero. The cache number is not evidence of
an overall speedup.

Source/configuration differences from the earlier 16-session test include
open-loop timing, synthetic input construction, sample composition, load copies,
sequence limit and KV capacity. Do not compare the two experiments as a
controlled before/after optimization series.

Artifacts (Git-ignored, retained locally and on the server):

- [Upstream result](../../benchmark/results/tracelab_timeline_compare/upstream_4x_2/replay/result.json),
  [configuration](../../benchmark/results/tracelab_timeline_compare/upstream_4x_2/configuration.json).
- [Agentrix result](../../benchmark/results/tracelab_timeline_compare/agentrix_gpu_4x_1/replay/result.json),
  [configuration](../../benchmark/results/tracelab_timeline_compare/agentrix_gpu_4x_1/configuration.json).
- Both directories retain `replay/requests.jsonl`, `replay/metrics_samples.jsonl`,
  before/after Prometheus counters, engine logs and GPU samples. The 213 upstream
  and 253 Agentrix metric samples reported no scraping errors.

## Timing semantics

The public TraceLab release does not expose server request-receipt timestamps.
We use its timing-analysis definition: the latest `user_message` or `tool_result`
timestamp preceding the first `reasoning`, `text` or `tool_call` event is an
**input-ready proxy**. `usage_report` alone is not a model-output boundary.
Missing, invalid or timezone-less timestamps are not replaced with made-up times.

Requests are submitted at `input_ready_time - window_start`, on a monotonic
replay clock. They do not wait for this engine's previous response. Original
inter-request and inter-session time differences remain unchanged, including
observable idle gaps. Tool durations are not summed or slept a second time.
This is open-loop serving-load replay, not execution of the original agent's
causal tool graph: a slow engine can still be processing one recorded round
when the next recorded arrival is submitted.

There are two explicitly distinguished load settings:

- `--load-copies 1`: one copy of the observed arrival timeline.
- `--load-copies N`: N independent copies overlaid on that same clock. Original
  intervals are preserved but offered load is multiplied; this is not a claim
  that the source trace had N times as much concurrency. Session IDs and
  synthetic prefixes are isolated across copies.

The public data omits original prompt/output content and a complete parent-child
session graph. Prompts are therefore fixed before the replay clock starts,
retaining the requested portion of the preceding synthetic **prompt**, and
filling missing history with deterministic synthetic tokens. Actual model outputs
are measured but do not change later inputs or arrival times. This gives both
engines identical inputs, but cannot reproduce original generated-output KV
reuse, cross-session prefix identity or branch dependencies. Skipped rounds and
window boundaries reset synthetic history rather than assuming continuity.

## Window selection and load

The release, license, source SHA256, model and engine checkpoints are documented
in [provenance](#provenance) below. Window selection uses only source data, before either
engine is measured. Among requests fitting a 32,768-token prompt-plus-output
limit, it maximizes distinct sessions in a time window, then request count;
earliest time breaks ties. No output caps, time compression, provider balancing
or long-tool-wait filtering are applied. A fixed `--window-start` can select
another half-open time interval explicitly.

The first expanded experiment uses:

| Parameter | Value |
| --- | --- |
| Source window start | 2026-05-29 03:09:38.807 UTC |
| Window duration | 120 s |
| Source sessions / requests | 23 / 33 |
| Independent load copies | 4 |
| Replay sessions / requests | 92 / 132 |
| Prompt / output tokens | 2,162,444 / 41,460 |
| Maximum prompt | 22,297 tokens |
| Eligible timed requests across release | 24,269 |
| Rejected for missing input-ready proxy | 36,607 |
| Rejected for context length, after timing gate | 296,285 |

The shared manifest SHA256 is
`cfa240fc2826f81094a773eb8a39acb1fbd418a221eb5490c55de3f716fb9352`.
Its legacy-named `tool_wait_ms` metadata is only a sum of recorded tool durations;
it is not an applied delay in this mode. Per-request applied `tool_wait_ms` is zero.

This short, cold-start, context-filtered window has only 40 noninitial requests
after duplication. It is an initial concurrency/timing validation, not a
representative long-running agent-memory evaluation. Selected sessions are not
necessarily all active at once.

Both groups use GPUs 0/1, Qwen3-VL-8B-Instruct BF16, TP=1 / DP=2, eager execution,
synchronous scheduling, 4,608 GPU KV blocks per replica (73,728 token slots),
a GPU memory utilization budget of 0.90, and 8,192 scheduled
tokens per replica per step. `max_num_seqs` is raised from 16 to **64 per replica**.
It is a ceiling, not a fixed batch size; long contexts and KV capacity can keep
the actual running count far below 64. GPU-only Agentrix enables the same
ForkAttention/session-aware DP/GPU-event/placement combination as before.
The official upstream 0.25.0 wheel remains the independent baseline; the
CUDA 12.9 versus locally built CUDA 12.8 extension caveat still applies.

A preliminary upstream attempt (`upstream_4x_1`) retained the old 2,048-block
limit. Both replicas frequently had only one running request while approximately
100 requests remained in flight. It was stopped as a capacity precheck, before
completion, and is excluded from comparisons. The common capacity was increased
for both measured groups; it was not changed within a measured run.

The client permits up to 512 in-flight requests and disables aiohttp's default
100-connection queue. Exceeding the explicit client limit aborts the trial,
rather than silently delaying arrivals. Every request must finish with the
exact recorded token lengths; no POST retries or success-only aggregates.

## Measurements and reproducibility

Each request records intended arrival, actual submission, completion, client
arrival lag, TTFT/E2E and latency relative to its **intended** arrival. The summary
includes peak in-flight requests and peak distinct in-flight sessions. One-second
Prometheus samples record each replica's running/queued requests and KV usage;
these sampled scheduler gauges must not be described as exact per-step kernel
batch sizes. GPU samples, runtime identity, command, configuration, and before/
after engine counters are retained with the raw request log.

From the server repository root:

```bash
benchmark/.venv/bin/python benchmark/src/tracelab_workload.py \
  --source benchmark/results/tracelab_release.jsonl.gz \
  --output benchmark/results/tracelab_timeline_120s_4x.json \
  --timing trace --window-seconds 120 --load-copies 4

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
benchmark/.venv/bin/python benchmark/scripts/profile_tracelab.py \
  --runtime-root "$PWD/benchmark/results/upstream_vllm_0_25_0" \
  --workload benchmark/results/tracelab_timeline_120s_4x.json \
  --output benchmark/results/tracelab_timeline_compare/upstream_new \
  --model /root/autodl-tmp/models/Qwen3-VL-8B-Instruct \
  --label upstream_timeline_4x --backend FLASH_ATTN \
  --max-num-seqs 64 --gpu-blocks 4608 --gpu-memory-utilization 0.90
```

For Agentrix use `$PWD/vllm` as the runtime root, a fresh output directory, and
`--backend FORK_ATTN --policy session_aware --gpu-events --active-placement`.
All other arguments and the workload manifest must be identical. For unamplified
load prepare a separate manifest with `--load-copies 1` and the same window start.
Existing manifests/results are never overwritten.

Server regression tests: 20 passed, covering timezone handling, input-ready
boundaries, dense-window selection, half-open windows, context gaps, independent
copies, overlapping submissions, cancellation on admission failure, streaming
accounting and summary calculations. Dense-window selection is also checked
against an exhaustive reference.

## provenance

Dataset: **TraceLab / SyFI Lab, University of Washington**, public
[v0.0.1 release](https://github.com/uw-syfi/TraceLab/releases/tag/v0.0.1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The compressed JSONL SHA256 is
`9d265eae69a31cae203848bea936f018148eed7ca8bf56050c5abe96da0b4e6b`.
Local tooling was inspected at `11b8b14c6005808ab272b3431487066832582414`.
There are 357,161 rows and 4,265 provider/session-ID groups.

The release's prefix-token counts describe provider cache accounting, not proof
of shared physical pages across branches. Subagent/parallel-tool activity exists,
but the public rows omit prompt contents and a complete parent-child graph.
The replay therefore cannot reconstruct exact cross-session prefix identity.

### Baseline isolation

The baseline is the **official vLLM 0.25.0 CUDA 12.9 wheel**, independently
installed using `uv pip install --no-deps --target ...`. Its release corresponds
to upstream commit `702f4814fe54fabff350d43cb753ae3e47c0c276`, before Agentrix's
changes. It is not the Agentrix tree with optimization flags disabled.

Official wheel: `vllm-0.25.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl`.
SHA256: `16670fbbad1483ae8d06794c0dd3c4e6b583f4bd1a61612a73b91af5848fdeb1`.
Both the core extension and FlashAttention extension were imported successfully
in the existing server environment. Baseline Python modules and native extensions
come from this wheel, not the modified source tree.

Agentrix source checkpoints are parent `61ef01a`, vLLM `1c9a17fd9`, and LMCache
`2fbdc42d`. Each run records the actual imported vLLM module path, its version,
PyTorch version, complete server command and relevant environment variables.
The source installation's generated version label is stale; its source checkpoint
is the reference above, not that generated label.

The official wheel was built with CUDA 12.9; Agentrix's native extensions were
built locally with CUDA 12.8. Both use the same Python 3.12 / PyTorch
`2.11.0+cu128` runtime. This compiler-build difference is a residual confound and
must not be attributed specifically to the memory policy.


## legacy-and-failures

The legacy manifest selects eight Claude and eight Codex sessions, each with
eight rounds (seed 20260905), requiring prompt plus output to fit 32,768 tokens
and interior summed tool waits to be at most 60 seconds. It has 128 requests,
1,896,544 prompt tokens and 44,965 output tokens. Session arrivals are synthetic,
100 ms apart; each session waits for its previous response and then the summed
tool duration. Unlike the current fixed-input protocol, generated outputs feed
later synthetic prompts.

Its SHA256 is `ebf863f0c6f4a711f6682b592f4ed78a289c31fb2bd82b0390a1144f5ca728c5`.
Both engines used 2,048 GPU blocks per replica, max_num_seqs=16 and GPU budget
0.80. One valid run per arm:

| Metric | Official upstream | Agentrix GPU-only |
| --- | ---: | ---: |
| Completion time | 458.48 s | 419.37 s |
| Output throughput | 98.07 tokens/s | 107.22 tokens/s |
| Follow-up TTFT P95 | 72.17 s | 54.69 s |
| TPOT P50 | 15.61 ms | 17.67 ms |

The 8.53% completion-time improvement is not a general speedup and cannot
override the current open-loop regression. Valid results remain under
`benchmark/results/tracelab_compare/upstream_1` and
`benchmark/results/tracelab_compare/agentrix_gpu_verified_1`.
To reproduce this historical protocol, pass `--timing legacy` when preparing
the workload and explicitly use `--max-num-seqs 16 --gpu-blocks 2048
--gpu-memory-utilization 0.80` in the profile harness.

Excluded attempts are retained for diagnosis, not averaged into valid runs:

- `tracelab_compare/agentrix_gpu_1`: client ServerDisconnectedError after 83
  successful requests, without a preceding engine crash; fresh connections
  were used for the subsequent valid comparison.
- `tracelab_compare/agentrix_tiered_verified_1` and `_2`: CPU/Mooncake restore
  encountered invalidated MemoryObj warnings and a GPU-connector assertion.
  Neither produced a valid completed-run aggregate.
- `tracelab_timeline_compare/upstream_4x_1`: aborted capacity precheck with
  2,048 blocks, excluded from the matched 4,608-block comparison.

The two tiered attempts used same-host TCP, 1 GiB CPU and 4 GiB Mooncake per
replica, 64 MiB scratch, 128 MiB pending remote ownership and a 4,096-token
external-load limit. Their logs remain locally and on the server.
Investigation is paused at the user's request. Virtualization has not been
established as the cause. Earlier small restore smokes do not invalidate these
failures; see [current KV status](../kv_memory_optimization_status.md).
