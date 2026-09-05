# GPU residency feedback for DP routing

`VLLM_AGENTRIX_DP_KV_EVENTS=1` lets the existing prefix-aware or session-aware
router use GPU cache store, removal, and reset events. It is opt-in and does
not introduce another routing policy or a synchronous scheduler RPC.

## Decisions and boundaries

The scheduler forwards its existing GPU event batch in `EngineCoreOutputs`,
before merging connector events. A bounded frontend index reconstructs token
prefixes from these events. A missing ancestor stops the match even if later
blocks survive. Cache reset clears the index. Capacity eviction and unknown
parents can undercount residency, but cannot create a claimed contiguous hit.

For eligible requests, event-derived prefix lengths replace logical residency
estimates. A follow-up may leave its historical session rank when another rank
has a longer verified GPU prefix. Existing first-turn balancing, overload
limits, work limits, and session-history guards still apply.

This is a routing hint, not a KV validity authority: events can be in transit
when a request arrives. The destination engine still performs its normal hash
lookup, restore, and recomputation checks. No GPU block is pinned for routing,
and no KV is copied between GPUs by the router itself.

Supported scope is one API frontend, internal DP, plain token prompts, and a
single full-attention or MLA cache group. Salted requests, LoRA, multimodal
identities, embeddings, hybrid groups, and unsupported block sizes retain the
existing logical routing policy. Index capacity per rank is bounded by the
existing warm-checkpoint limit divided by DP size, with a minimum of one.

The scheduler gate uses its internal-DP completion-reporting contract rather
than `data_parallel_size > 1`: dense DP cores rewrite that size to one. Worker
cache specs are normalized through vLLM's existing scheduler configuration
builder; no duplicate normalization layer is added.

## Server profiling

All measurements ran on `connect.bjb2.seetacloud.com`, not the development
machine. The matched comparison used Qwen3-VL-8B-Instruct, two RTX 5090 GPUs,
BF16, FORK_ATTN, eager execution, 384 GPU blocks per replica, active placement,
and no external connector. Each of three trials primed a 512-token common
prefix, sent 12 concurrent first turns with 1,024 private tokens, then appended
64 tokens for follow-ups. Each request generated one token.

Values below are medians across trials:

| Metric | Logical hints | GPU event hints |
| --- | ---: | ---: |
| First-turn P50 TTFT | 471.57 ms | 474.85 ms |
| Follow-up P50 TTFT | 394.48 ms | 396.27 ms |
| Follow-up throughput | 28.83 req/s | 29.34 req/s |
| Follow-up GPU cached-token rate | 53.00% | 53.00% |
| Router-only P50, 5,000 decisions | 35.30 us | 72.54 us |
| Router-only P99, 5,000 decisions | 41.73 us | 82.59 us |

This workload does not establish a material speedup. The event path costs
about 37 us more per decision, primarily for reconstructing exact contiguous
prefix matches. The microbenchmark measures routing only, excluding producer
serialization and event application; the end-to-end comparison includes both.
The feature remains disabled by default.

Server artifacts:

- `benchmark/results/dp_placement_logical`
- `benchmark/results/dp_placement_physical_verified`
- `benchmark/results/dp_kv_event_cpu.json`

An earlier `dp_placement_physical` run did not activate the dense-DP producer
gate; it is not evidence for the event path and is excluded from the table.
Both ranks in the verified run logged producer enablement and frontend event
receipt.

## CPU/Mooncake integration

Two additional trials enabled the same event path together with asynchronous
proactive backup, a 128 MiB CPU allocator and a 1 GiB Mooncake contribution per
replica, a 64 MiB native scratch buffer, and a 128 MiB pending-write budget.
Both replicas initialized Mooncake and completed asynchronous stores and all
requests. Follow-up GPU reuse was 59.17%, P50 TTFT was 335.46 ms, and throughput
was 27.52 req/s; first-turn P50 was 535.18 ms. These are integration results,
not an isolated routing speedup: the external-memory configuration also changed.

External retrieval counters were zero in this DP trace. It validates concurrent
DP routing and tiered writes, not cross-replica restore. The separate forced
GPU/CPU eviction replay in [the tier guide](kv_tier_memory.md) verifies actual
512-token Mooncake restores over TCP and RDMA. Cross-host RDMA bandwidth and
multi-node DP have not been benchmarked.

The combined test exposed a device-selection error: model-parallel rank zero
inside a second DP replica was incorrectly mapped to CUDA device zero.
LMCache now retains vLLM's selected device ordinal, resolves the concrete KV
tensor device for background backup, and uses local device ordinals for MLA
broadcast buffers. Tests cover remapped visibility, DP+TP, and multi-node
model-parallel sizing; actual broadcast-buffer placement is tested on GPUs 2/3.
Artifacts are under `benchmark/results/dp_placement_mooncake_verified`.

## Reproduction and regression checks

Run from `/root/autodl-tmp/Agentrix` on the profiling server:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTHONPATH="$PWD/vllm:$PWD/LMCache"
export VLLM_BIN="$PWD/benchmark/.venv/bin/vllm"
export POLICIES=session_aware TRIALS=3 SESSIONS=12
export SHARED_PREFIX_TOKENS=512 SESSION_TOKENS=1024
export NUM_GPU_BLOCKS=384 ENFORCE_EAGER=1
export VLLM_AGENTRIX_KV_PLACEMENT_ACTIVE=1
VLLM_AGENTRIX_DP_KV_EVENTS=0 OUTPUT_ROOT=benchmark/results/dp_logical_replay \
  bash benchmark/scripts/run_agent_session_dp_profile.sh
VLLM_AGENTRIX_DP_KV_EVENTS=1 OUTPUT_ROOT=benchmark/results/dp_events_replay \
  bash benchmark/scripts/run_agent_session_dp_profile.sh
benchmark/.venv/bin/python benchmark/scripts/profile_dp_kv_events.py \
  --output benchmark/results/dp_kv_event_cpu_replay.json
```

To add existing Mooncake services, set `LMCACHE_CONFIG_FILE` to a tiered config,
enable `VLLM_AGENTRIX_KV_PROACTIVE_BACKUP=1` and
`VLLM_AGENTRIX_KV_PROACTIVE_ASYNC=1`, and pass the connector through
`KV_TRANSFER_CONFIG`. The combined run used `LMCacheConnectorV1`, `kv_both`,
`kv_load_failure_policy: recompute`, and
`kv_connector_extra_config: {"lmcache.max_tokens_per_load": 512}`.

The vLLM regression set covers scheduler-to-frontend event serialization,
dense-DP configuration, disabled-path behavior, prefix holes, bounded index
eviction, unsupported layouts, session reassignment, load guards, residency,
and placement. LMCache regression tests additionally cover device mapping,
MLA buffer placement, asynchronous store ownership, and native transfer failure.
