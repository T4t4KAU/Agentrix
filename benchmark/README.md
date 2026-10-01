# Agentrix Bench

This directory contains the Agentrix shared-prefix simulator, API benchmarks,
and local vLLM end-to-end benchmarks. See
[`../README.md`](../README.md) for the complete installation, build, and
reproduction workflow.

## Common Commands

```bash
.venv/bin/agentrix-bench inspect-data
.venv/bin/agentrix-bench simulate
.venv/bin/python -m pytest

BACKENDS="FLASH_ATTN FORK_ATTN" \
PREFIX_TOKENS=8192 \
BRANCHES=16 \
OUTPUT_TOKENS=64 \
./scripts/run_vllm_benchmark.sh

MODEL_PATH=/path/to/Qwen3-0.6B \
PREFIX_TOKENS=2048 \
BRANCHES=2 \
OUTPUT_TOKENS=64 \
./scripts/run_sglang_benchmark.sh

DRY_RUN=1 ./scripts/run_vllm_fanout_matrix.sh
```

## LangGraph RAG Agent Benchmark

Install the optional Agent dependency and run a live graph against an
OpenAI-compatible Agentrix vLLM server:

```bash
uv pip install -e ".[agent,test]"

.venv/bin/agentrix-langgraph live \
  --base-url http://127.0.0.1:9000/v1 \
  --model qwen3-0.6b \
  --task-file configs/langgraph_agent_tasks.jsonl \
  --cases 1 \
  --branches 16 \
  --rag-root ../docs \
  --bootstrap-chunks 24 \
  --bootstrap-max-chars 30000 \
  --concurrency 16 \
  --output results/langgraph/live.json
```

The graph performs an initial retrieval over real local files, plans over the
retrieved evidence, fans out parallel research branches, requires every branch
to issue an OpenAI function call to `rag_search`, feeds each tool result back
to the model, and reduces the branch answers. The live result records every
exact LLM request and tool result.

Use dependency-ordered replay for fair backend comparisons. It preserves the
captured request bodies and runs `planner -> parallel tool calls -> parallel
reflections -> reducer` without carrying live orchestration idle time into the
backend measurement:

```bash
.venv/bin/agentrix-langgraph replay \
  --base-url http://127.0.0.1:9000/v1 \
  --model qwen3-0.6b \
  --trace results/langgraph/live.json \
  --concurrency 16 \
  --output results/langgraph/replay.json
```

Set `--timing captured` only when reproducing the original absolute arrival
timeline. It is not the default throughput comparison mode.

The default model is `Qwen/Qwen3-0.6B`. Set `MODEL_PATH` to use another
Hugging Face model or a local model directory. All output is written under
the Git-ignored `results/` directory. The vLLM script writes one subdirectory
per backend plus `backend_comparison.csv` and `backend_comparison.md` with the
end-to-end latency and throughput deltas.

## WebLINX Multimodal 8-DP Benchmark

Build a deterministic eight-case subset from the WebLINX validation split:

```bash
.venv/bin/python -m weblinx_data \
  --output-dir results/weblinx_subset \
  --split validation \
  --case-count 8 \
  --branch-count 8 \
  --seed 2026
```

The builder selects distinct demonstrations with good screenshots and eight
ranked candidates, downloads only the selected replay files and PNGs, resizes
the screenshots to 1280x720, and writes a reproducible `manifest.json`. The
downloaded data remains under the Git-ignored `results/` directory.

Run the Pressure32K/32-shaped workload on eight DP replicas:

```bash
MODEL_PATH=/path/to/Qwen3.6-27B \
GPU_IDS=0,1,2,3,4,5,6,7 \
MANIFEST="$PWD/results/weblinx_subset/manifest.json" \
NUM_GPU_BLOCKS_OVERRIDE=84 \
./scripts/run_weblinx_8dp.sh
```

Each of the eight WebLINX states first issues one natural multimodal bootstrap
request. Its eight candidates are then expanded into four independent rollout
strategies each, giving 32 branches per state and 256 globally shuffled branch
requests. The client sends no DP-rank header: placement is entirely controlled
by the internal-DP server.

The default matrix contains the three ablations needed to reproduce the text
Pressure32K/32 design: `flash_ordinary`, `fork_ordinary`, and
`fork_prefix_aware`. Only the last arm enables prefix-aware DP routing and
fanout scheduling. All arms use the same 256-token output limit, seeded
lognormal 256-token suffix distribution, KV capacity, and request order.

`TEXT_PREFIX_TOKENS` defaults to 28,000, leaving room for the image tokens,
64-token common analysis, candidate/rollout suffix, and 256 generated tokens
within a 32K context. `MAX_NUM_SEQS` defaults to 64 and Forest CUDA Graphs are
enabled. Calibrate `NUM_GPU_BLOCKS_OVERRIDE` so one root cohort fits per rank
but two independent roots do not; `84` produces 57,344 KV tokens per rank for
the validated Qwen3.6/H20 build. At that deliberately tight boundary, the
validated optimized run still recorded 27 preemptions because Qwen3.6 uses
coarse 784-token hybrid cache pages. The result directory contains per-variant
CSV/JSON summaries, server logs, Prometheus metrics, and `comparison.md`. See
`docs/fork_attention/qwen35_qwen36_forkattention_design.md` for the validated result and its
limitations.

## SGLang Local Benchmark

After adding and installing the `sglang` submodule, run the same Agentrix
OpenAI-compatible workload against SGLang:

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
SGLANG_PYTHON=/path/to/python \
PREFIX_TOKENS=2048 \
BRANCHES=2 \
OUTPUT_TOKENS=64 \
./scripts/run_sglang_benchmark.sh
```

The script launches one SGLang server per `DP_REPLICAS`, routes Agentrix
branches through the existing `agentrix-bench run-api` client, and writes
results under `benchmark/results/sglang_*`. It defaults to `--no-stream` for
the benchmark client because SGLang deployments vary in streaming usage
reporting support. Set `BENCHMARK_EXTRA_ARGS=""` to request streaming mode.

Use `run_vllm_fanout_matrix.sh` for stronger shared-prefix cases. It compares
`FLASH_ATTN` and `FORK_ATTN` over five long-prefix, high-branch-count workloads
and writes an aggregate `matrix_summary.md`.

Use `run_offload_backend_comparison.sh` for the seven-way offload comparison:

```bash
MODEL_PATH=/path/to/Qwen3-1.7B \
CPU_SIZE_GB=0.5 \
DISK_SIZE_GB=2 \
./scripts/run_offload_backend_comparison.sh
```

It covers ForkAttention with no offload, native CPU offload, default and
fork-aware LMCache CPU offload, and fork-aware LMCache CPU plus disk. It also
covers FlashAttention with no offload and ordinary native LRU CPU offload. The
generated `offload_comparison.md` includes pairwise throughput deltas, logical
KV footprint reduction, KV movement, disk footprint, and load failures.

### Controlled KV lifecycle probe

`scripts/benchmark_agent_kv_tiering.py` compares native APC, official CPU offload,
and selective backup through `kv_transfer_params.max_offload_tokens`. Use the
vLLM virtual environment, which provides Transformers and prometheus-client.
Run only on a dedicated backend: each trial resets its entire prefix cache.
Enable `VLLM_SERVER_DEV_MODE=1` for the official reset API. Keep results on the
experiment server.

First start the offload backend with the same model, attention backend and
scheduler as the APC baseline, an explicit `--kv-cache-memory-bytes` budget,
and the official connector configuration:

```bash
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":2147483648,"blocks_per_chunk":1}}'
```

This fragment reserves a 2 GiB CPU cache; include that cost in comparisons.
It is for the CUDA runtime, not a version-independent Ascend recipe. Verify
mixed-state restoration before running pressure or capacity comparisons:

```bash
"${REPO_ROOT}/vllm/.venv/bin/python" \
  "${REPO_ROOT}/benchmark/scripts/benchmark_agent_kv_tiering.py" \
  --base-url "${BACKEND_URL}" --model "${SERVED_MODEL}" \
  --tokenizer "${MODEL_DIR}" --mode offload --scenario roundtrip \
  --sessions 1 --prompt-tokens 8192 --trials 3 \
  --output "${RESULTS_DIR}/offload-roundtrip.json"
```

The probe resets device KV while retaining external KV, checks actual transfer
bytes and prompt-source counters, and requires identical generated token IDs.
Repeat with unaligned prompt lengths and the runtime's supported finer caching
configuration to exercise partial recurrent tails; an aligned prompt alone
does not cover that path. Output equality is a correctness gate for these
inputs, not a task-quality evaluation.

For the partial-tail cap fix, restart with a finer `--prefix-match-unit` and
one physical block per offload chunk. Use the actual uniform physical block
size reported for the mixed model; the following values match the tested
Qwen3.5 configuration, not arbitrary models:

```bash
"${REPO_ROOT}/vllm/.venv/bin/python" \
  "${REPO_ROOT}/benchmark/scripts/check_agent_kv_boundaries.py" \
  --base-url "${BACKEND_URL}" --model "${SERVED_MODEL}" \
  --tokenizer "${MODEL_DIR}" --block-tokens 528 --prefix-match-unit 16 \
  --prompt-tokens 8192 --output "${RESULTS_DIR}/partial-tail-caps.json"
```

The server must also use `--prefix-match-unit 16`. This probe appends one token
to the saved prompt so the partial boundary is eligible for restoration. It
compares against a cold reference, covers six backup caps, and sets a zero
cap on every resume: existing copies must remain readable without new writes.
It checks the exact restored/computed boundary and both transfer directions.
The reusable probe has local validation tests; its hardware run is pending.

For the pressure comparison use `--scenario pressure --sessions 4
--pressure-requests 8` and separate outputs for `--mode apc`, `--mode offload`,
and `--mode selective`. Restart when changing the connector or pool budget;
APC omits the connector. Offload and selective use identical GPU and CPU
budgets and may share a backend with both caches reset before each trial.
Use the same seed, prompt length, output length and session plan in all arms.
Only the selective arm disables new backups for the declared terminal
requests. Inspect local hits, actual restores and recomputation to establish
that the chosen workload exceeds device-cache capacity before interpreting
the comparison. Counters must account for all input tokens and remain stable
before each phase snapshot; the quiet interval is not a transfer fence.
Transfer counters can appear only after the first transfer; validity requires
actual nonzero writes and, for the roundtrip, reads after exercising the path.
Both probes publish progress using an atomic file replacement, preserving the
last complete JSON snapshot if a write is interrupted. A progress file with
`valid: false` is incomplete or failed and must not be included in comparisons.

This is a sequential lifecycle probe, **not official AgentX**. It does not
measure peak VRAM, host RSS, a capacity limit or tool-wait prefetch. Use server
telemetry and repeated pool-size comparisons for those separate claims.

For a small-buffer Ascend DMA correctness check, activate the matching CANN and
Ascend virtual environment, then run:

```bash
"${ASCEND_VENV_DIR}/bin/python" \
  "${REPO_ROOT}/benchmark/scripts/check_ascend_kv_transfers.py" \
  --device 0 --output "${RESULTS_DIR}/ascend-transfers-device0.json"
```

Repeat on another visible device with a separate output path. This exercises
installed block-copy helpers, event completion, reused buffers and untouched
pages with BF16/FP32/uint8 payloads. It neither loads a model nor resets a
serving engine. It does not establish hybrid KV restoration, scheduler CoW,
selective backup support or performance; those require matching runtime APIs
and separate model-level experiments.
Current verification and hardware availability are recorded in
[KV memory status](../docs/kv_memory_optimization_status.md#选择性备份边界修复与验证入口).

### Ascend native mixed-cache offload compatibility

The experimental `vllm_ascend.kv_offload.native` adapter backports official
Ascend native offload to the older canonical-cache handler API. It retains the
official CPU cache manager. Do not substitute it into a newer runtime that
already provides the matching native connector.

For the tested older source shape, create an isolated vLLM package overlay:

```bash
"${ASCEND_VENV_DIR}/bin/python" \
  "${REPO_ROOT}/benchmark/scripts/prepare_ascend_offload_overlay.py" \
  --source "${INSTALLED_VLLM_PACKAGE}" --target "${OVERLAY_DIR}/vllm"
```

The helper refuses an existing target or unexpected patch anchors. It backports
request offload caps and reset acknowledgements, allows zero-compute async
restores through scheduling, and rounds Mamba restore hits down to a valid
state boundary. Installed sources remain untouched. Activate the isolated
package directory through `PYTHONPATH`, together with the matching Ascend
adapter, and use this connector configuration:

```json
{
  "kv_connector": "AscendOffloadingConnector",
  "kv_connector_module_path": "vllm_ascend.kv_offload.native.offloading_connector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "NPUOffloadingSpec",
    "spec_module_path": "vllm_ascend.kv_offload.native.npu",
    "cpu_bytes_to_use": 2147483648
  }
}
```

First run `benchmark_agent_kv_tiering.py --scenario roundtrip --mode offload`
against an isolated development server with reset endpoints enabled. Require
actual H2D transfers, no local prefix hits, and identical cold/restored output
tokens before pressure comparisons. Preserve failed runs on the server; do
not treat a completed transfer as proof of recurrent-state correctness.

### DP routing

DP routing now uses the official `vllm-router==0.1.15` package in a separate
router environment. `consistent_hash` consumes `X-Session-ID` for multi-turn
sessions; `cache_aware` and `round_robin` use their official implementations.
Direct requests to the backend exercise native internal DP load balancing.

Install `requirements-router.txt` in the router environment and `.[dp]` in the
benchmark environment, then run `scripts/serve_dp_router.py` with
the official CLI options. `scripts/run_agent_session_dp_profile.sh` compares
these policies using FlashAttention. Set `OUTPUT_ROOT` to a results directory
on the experiment server. Session/revisit drivers accept `--base-url` for the
router and `--control-url` for backend metrics and resets. Restart both services
for cold comparisons; an engine reset does not clear router estimates.

See [DP routing](../docs/dp_routing.md) for setup and verification limits.
The private internal DP router and its profiling simulators have been removed.
Historical `VLLM_AGENTRIX_DP_ROUTING_POLICY` and `VLLM_FORK_ATTN_DP_PREFIX_*`
controls are no longer supported; old results do not measure the new router.

The following full-dataset recipe targets those older routing controls and is
kept as historical source and now exits with migration guidance. Restore its
matching historical code only when reproducing old results:

```bash
MODEL_PATH=/path/to/Qwen3-8B \
GPU_IDS=0,1 \
./scripts/run_vllm_dp_full_dataset.sh
```

This produces separate `flash_dp`, `fork_dp`, and `fork_optimized_dp` results,
plus an optional pressure-aware offload run. Existing result CSV files are
treated as checkpoints, so an interrupted full-dataset run can be resumed with
the same `OUTPUT_ROOT`. The full-dataset optimized variant uses strict Graph
bucket placement and a 10 ms arrival wave; both ordinary-DP baselines keep the
optimized router and its telemetry path disabled.
