# CPU and Mooncake KV tiers

Proactive backup can now publish the same completed D2H buffers to LMCache's
CPU cache and Mooncake. The CPU cache is the hot tier; Mooncake supplies shared
DRAM capacity after local CPU eviction. Both use LMCache's existing storage
manager and registered CPU allocator, without another GPU copy or allocator
inside vLLM.

Enable `VLLM_AGENTRIX_KV_PROACTIVE_BACKUP=1` and optionally
`VLLM_AGENTRIX_KV_PROACTIVE_ASYNC=1`, using
`benchmark/configs/lmcache_mooncake_tiered.yaml`. The service configuration must
match the Mooncake master and metadata services. The supplied configuration
uses local TCP endpoints. `protocol: rdma` and a valid `device_name` select the
existing Mooncake RDMA transport.

## Capacity and ownership

- `max_local_cpu_size` limits the CPU hot-cache/transfer allocator in GiB.
- `global_segment_size` controls DRAM contributed by this Mooncake client.
- `local_buffer_size` controls Mooncake's native scratch allocation.
- `remote_max_inflight_bytes` limits host buffers retained by background remote
  batches, defaulting to 128 MiB when `proactive_remote_backup` is enabled.

These allocations are distinct: the remote in-flight budget bounds ownership
inside the CPU allocator; it does not allocate another pool of that size. The
native scratch buffer and contributed Mooncake segment do consume additional
host memory. Reuse the same registered CPU buffers for zero-copy Mooncake puts
and gets with `save_chunk_meta: false` and `remote_serde: naive`.

`BoundedRemoteWriter` admits an entire batch or immediately rejects it. Duplicate
in-flight keys and excess bytes take no additional buffer references. Rejected
remote writes leave the CPU copy usable. Capacity is released after the native
transfer and serialized-buffer cleanup, including failure paths. Shutdown drains
remote writes and unregisters buffers before closing their CPU allocator.

A Python timeout cannot stop a C++ transfer running in another thread. The
Mooncake connector therefore drains the native call before releasing buffer
ownership, even after cancellation. A stuck native call can exhaust the bounded
write budget and delay shutdown, but cannot cause unbounded buffer retention or
early reuse of memory still accessed by the transfer.

## Restore and validity

Mooncake prefix lookup uses a batched existence query off the event-loop thread.
Negative backend status codes are misses. Asynchronous prefetch uses the native
batch-get API and returns only the consecutive valid prefix, releasing all later
chunks after a hole. The vLLM connector bounds lookup/prefetch with
`kv_connector_extra_config: {"lmcache.max_tokens_per_load": 512}` and uses
`kv_load_failure_policy: recompute` if a key disappears between lookup and load.

CPU residency is verified after restore and invalidated by local eviction
feedback. Mooncake objects can be independently evicted, so remote presence is
checked on retrieval and is not recorded as a permanent `REMOTE` residency bit.
The GPU planner may use a verified CPU copy; a historical remote write alone
does not grant permission to discard the only known copy without recomputation.

## Server validation

The functional pressure run uses 96 GPU blocks, a 128 MiB CPU allocator, a
1 GiB Mooncake segment, a 64 MiB native scratch buffer, and 128 MiB of permitted
in-flight remote writes. P/A/B/C/D/A prompts force both GPU recycling and local
CPU eviction. The final A restored 512 tokens through Mooncake and reproduced
the original deterministic output. Its single-request latency was 86.7 ms;
this smoke does not establish a throughput gain.

Reproduce on the profiling server:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PATH="$PWD/benchmark/.venv/bin:$PATH"
RUNTIME_PYTHON="$PWD/benchmark/.venv/bin/python" \
VLLM_BIN="$PWD/benchmark/.venv/bin/vllm" \
PROFILE_RESTORE=1 LMCACHE_LOCAL_CPU=true LMCACHE_CPU_SIZE_GB=0.125 \
LMCACHE_ASYNC_LOADING=true PROACTIVE_REMOTE_BACKUP=true \
MOONCAKE_MEMORY_SIZE_GB=1 MOONCAKE_BUFFER_BYTES=67108864 \
OUTPUT_DIR=results/unified_mooncake_tcp_verified PORT=9012 \
bash benchmark/scripts/run_lmcache_mooncake_smoke.sh
```

Raw results and master/worker logs remain in the server's
`benchmark/results/unified_mooncake_tcp_verified` directory. Regression tests
cover capacity rejection, ownership cleanup, serialization and write failures,
timeouts, repeated cancellation, prefix holes, status validation, and shutdown
ordering.

The same replay also passed with `MOONCAKE_PROTOCOL=rdma` and
`MOONCAKE_DEVICE_NAME=mlx5_0`; the final restore took 98.5 ms. The device's port
was active. Artifacts are in `benchmark/results/unified_mooncake_rdma`.
Both replays use a single host with a client-contributed segment, so they verify
transport initialization and registered-buffer compatibility, not cross-host
RDMA bandwidth or a transfer speedup.
