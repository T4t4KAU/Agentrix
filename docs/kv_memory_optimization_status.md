# KV memory optimization checkpoints

The six planned stages have working, opt-in implementations. Each stage was
reviewed and committed separately; runtime tests and profiling ran on the
server at `connect.bjb2.seetacloud.com`.

| Stage | Implemented scope | Review checkpoint |
| --- | --- | --- |
| Baseline | Reviewed asynchronous backup and residency integration | Agentrix `4b41ca2` |
| Pressure-aware admission | Occupancy-based whole-chunk admission, retained-block accounting, bounded in-flight work | Agentrix `6c3287e` |
| Cost-aware GPU eviction | Bounded tier/reuse/age ordering, shared-prefix protection, generation revalidation | Agentrix `efcbd6e` |
| Restore and prefetch | Pre-lookup load budget, partial-load recomputation, generation-safe restore and CPU eviction feedback | Agentrix `3fbf77d` |
| CPU/Mooncake tiers | Shared transfer buffers, bounded remote writes, failure-safe native ownership, TCP/RDMA restore | Agentrix `4b60569` |
| DP/placement linkage | GPU eviction feedback into existing DP routing; replica-local device mapping | vLLM `1c9a17fd9`, LMCache `2fbdc42d` |

Implementation and measurement details:

- [Residency, pressure admission, eviction and restore](kv_residency_shadow.md)
- [CPU/Mooncake capacity, ownership and restore](kv_tier_memory.md)
- [DP residency feedback, integration fixes and profiling](dp_kv_placement.md)

## Final validation

The final server regression pass completed with 93 vLLM tests passing and
95 LMCache tests passing, with two skips. It includes actual scheduler event
export and MLA broadcast-buffer placement on the second GPU pair. vLLM's
staged pre-commit hooks, including mypy, passed; the changed LMCache files
passed Ruff lint and formatting checks.

Self-review and combined deployment found and corrected the dense-DP event
enablement gate, duplicate cache-spec normalization, and a replica-local rank
being incorrectly interpreted as a CUDA ordinal. The combined DP/LMCache/
Mooncake replay subsequently completed on both replicas. No known critical
error remains in these tested paths; this is not a claim of exhaustive model,
transport or distributed-layout coverage.

## Performance and remaining limits

The GPU eviction cost is a small deterministic estimate from tier, reuse and
age, not an online calibrated recompute/transfer latency model. The CPU and
Mooncake allocators retain separate capacity limits; remote writes have an
additional ownership budget, not another allocation pool.

DP GPU-event feedback did not establish a material speedup in the measured
trace and roughly doubled router-only decision cost from 35 to 73 us. It
remains disabled by default. Combined tiered writes and DP routing work, but
that DP trace did not exercise external restores. Separate pressure traces
verified 512-token CPU/Mooncake restores with matching deterministic output.

Cross-host RDMA bandwidth, multi-node DP, and realistic long-running agent
traces still need performance evaluation. Single-host RDMA initialization and
restore success must not be presented as proof of cross-host transfer gains.
