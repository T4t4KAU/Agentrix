# Agentrix Bench

This directory contains the Agentrix shared-prefix simulator, API benchmarks,
and local vLLM end-to-end benchmarks. See
[`../README.md`](../README.md) for the complete installation, build, and
reproduction workflow.

## Script entry points

| Purpose | Entry points |
| --- | --- |
| Official routing and document QA | `serve_dp_router.py`, `run_agent_session_dp_profile.sh`, `run_ascend_router_comparison.py` |
| KV backup and capacity | `benchmark_agent_kv_tiering.py`, `benchmark_dp_kv_lifecycle.py`, `benchmark_dp_kv_growth.py` |
| Official AgentX KV comparison | `run_agentx_kv_comparison.py`, `summarize_agentx_memory.py` |
| ForkAttention correctness and scale | `benchmark_fork_numerics.py`, `benchmark_fork_scale.py`, their `summarize_*` scripts |
| Attention operator profiling | `benchmark_flashinfer_cascade.py`, `benchmark_fork_decode.py`, `run_fork_attention_ncu.sh` |
| Tool data and host memory | `benchmark_tool_result_paging.py`, `benchmark_tool_snapshot_sharing.py`, `benchmark_prompt_tool_result_context.py` |
| General backend comparison | `run_vllm_benchmark.sh`, `run_sglang_benchmark.sh`, `run_offload_backend_comparison.sh` |

Retired private-router launchers and experiments requiring removed KV
residency/placement or tool-trim APIs have been removed, along with their
dedicated plots. Current HTTP clients, dataset builders, validation tools and
summarizers remain. Coding demos use the Python entry points documented in
[the demo guide](../docs/coding_agent/coding_agent_live_demo.md).
Historical source snapshots are retained on the experiment server under
`${RESULTS_DIR}/script-cleanup-20261004/`; no archived copies are kept here.

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

## WebLINX dataset preparation

The dataset builder remains available independently of the retired private
8-DP routing recipe:

```bash
.venv/bin/python -m weblinx_data \
  --output-dir "${RESULTS_DIR}/weblinx_subset" \
  --split validation --case-count 8 --branch-count 8 --seed 2026
```

It selects distinct demonstrations, downloads their replay files and images,
and writes a reproducible manifest. The historical routing comparison requires
its archived runtime; its old launcher is no longer a current reproduction
entry point. See [the workload design](../docs/fork_attention/qwen35_qwen36_forkattention_design.md).

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

For the dual-rank version, keep the official router on `consistent_hash`, with
`--intra-node-data-parallel-size 2`. Use a DP=2, TP=1 backend and explicit device
and CPU budgets **per rank**. Control operations bypass the router:

```bash
"${VLLM_PYTHON}" "${REPO_ROOT}/benchmark/scripts/benchmark_dp_kv_lifecycle.py" \
  --base-url "${ROUTER_URL}" --control-url "${BACKEND_URL}" \
  --model "${SERVED_MODEL}" --tokenizer "${MODEL_DIR}" \
  --mode selective --sessions 8 --pressure-requests 16 \
  --prompt-tokens 8193 --output-tokens 8 --trials 2 --seed 20261001 \
  --output "${RESULTS_DIR}/selective.json"
```

Compare `apc`, `offload` and `selective`; the APC server omits the connector.
The tested Ascend mixed-state boundary is 8193 tokens; verify boundaries for
other runtimes before reusing this setting. Session IDs remain unchanged across
arms and phases. Only application-known terminal pressure requests skip backup.
The probe checks per-rank prompt accounting, both ranks participating, no
preemption, and exact cold/resumed output token equality. No reset occurs
between prime, pressure and resume. `--tool-gap-seconds` adds idle time after
pressure; the pressure phase itself also contributes to inter-turn gaps. This
sequential experiment measures restoration under cache pressure, not concurrent
capacity or official AgentX throughput.

`run_dp_kv_lifecycle.py --config "${PRIVATE_CONFIG}" --output "${RESULTS_DIR}"`
runs the Ascend matrix from a server-private JSON with `env`, `cwd`,
`router_python`, optional `seeds`/`trials`, and `cases`. Each case supplies
`label`, complete server `argv`, and `modes` (APC or both offload modes).
It requires two idle NPUs before each server launch, owns only its child
processes, reverses offload-mode order for the second seed, and stops its
services on failure. Start with a fresh output directory. After fixing a
failure, `--resume` archives failed outputs and logs and preserves valid cells;
keep the original source snapshot on the server before replacing scripts.
Do not change completed cells' workload or launch configuration when resuming.
Use `--extend` to add cases after a completed matrix; prior backend commands
must remain identical and prior summaries are archived until the expanded
matrix is validated.
`sample_npu_experiment.py --pid "${CONTROLLER_PID}" --output
"${RESULTS_DIR}/npu-samples.jsonl"` records board observations separately;
sampling can miss transient allocation peaks.
After completion, run `summarize_dp_kv_lifecycle.py "${RESULTS_DIR}"` in the
same virtual environment. It rejects missing cells, unequal input plans or
cross-run output differences before writing `validated-summary.json`.

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
The lifecycle strategy and measured scope are described in
[KV memory management](../docs/kv_memory_optimization_status.md#agent-hints-与选择性备份).

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

For the Ascend shared-document QA comparison, use
`scripts/run_ascend_router_comparison.py --config "${PRIVATE_CONFIG}"
--output "${RESULTS_DIR}/runs" --router-python "${ROUTER_PYTHON}"
--qa-cases "${CASE_FILE}" --trials 1`. Create a fresh output directory first.
The default compares official policies with question waves. To keep official
`consistent_hash` fixed and compare FIA ForkAttention off/on, add
`--qa-fork-attention --qa-fork-min-shared-tokens 4096 --qa-arrival fanout`.
Both arms use the same diagnostics and shared-prefix threshold; only `enabled`
changes. The threshold override is experimental; the operator default is 32768.
Use `--qa-arrival waves` separately to check the original arrival pattern.
Fanout submits a document's questions together, allowing overlapping decode;
actual physical sharing must still be confirmed from planner/execution logs.
The ForkAttention comparison excludes `--qa-prefill-gate`. Keep QA quality,
output-token differences, memory observations and trigger counts with latency
results; an enabled flag alone is not evidence of shared execution or a gain.

To scan larger shared prefixes with fixed output length, use the same controller
with `--fork-scale --scale-trials 3 --trials 1 --fork-min-shared-tokens 16384`
and omit QA flags. `benchmark_fork_scale.py` tests 16K/32K/64K prefixes with
2/4/8 branches per rank, two ranks, 128-token private tails and exactly 64
generated tokens per request. Set the backend context limit to at least 65728.
This requires a dedicated backend with prefix-cache reset enabled. It probes
official hash placement, warms shared prefixes and branches, then checks
per-rank completions, cache hits and preemptions. Synthetic token inputs are a
controlled scaling probe, not QA or AgentX. Run eager and Decode ACLGraph with
separate private configs and result directories; compare Fork off/on within
each mode. Keep output-hash differences alongside fixed-length timing results.

For independent service repeats, use `--trials 2` with explicit `--seeds`.
Each seed's baseline/candidate services now run consecutively; their order
reverses across seeds and independent repeats. `--scale-trials` controls timed
batches within a service and must not be counted as independent service runs.
Use `--scale-prefix-tokens 16384 32768 65536 --scale-branches 4 8` to retain
short-prefix controls while concentrating on the candidate fanout shapes.
Keep the runtime fixed and preserve a source manifest; new seeds and restarts
test reproducibility, without changing the operator or promoting it to default.

After all services complete, run `scripts/summarize_fork_scale.py` with
`--runs "${RESULTS_DIR}/runs" --seeds 20261003 20261004 --restarts 2
--output "${RESULTS_DIR}/validated-comparison.json"` on the server. It rejects
incomplete plans, unequal input/output tokens, changed prompt-source work and
preemptions. It averages each service's timed batches before combining equally
weighted seeds/restarts and reports the range of paired changes. Runtime/config
isolation and formal-window memory observations must also be audited; this is
not a significance test or a task-quality evaluation.
When a strict output check fails, `--diagnose-output-differences` retains all
shapes and reports mismatch counts and within-service output variation. Shapes
with any mismatched paired output are explicitly ineligible as equal-output
latency comparisons; the diagnostic mode does not turn them into passing runs.

For full-token diagnostics, run `scripts/benchmark_fork_numerics.py` against
each isolated service with `--base-url "${ROUTER_URL}" --control-url
"${BACKEND_URL}" --model "${MODEL_NAME}" --request-mode concurrent` and a
fresh server-side `--output`. Repeat with `--request-mode batched`: this submits
one prompt list per rank, preserving the same input plan and cache preparation.
Defaults cover 16K/32K/64K, 4/8 branches per rank, three repeats and 64 generated
tokens. Both modes retain full token IDs and top-five logprobs. These instrumented
timings are diagnostics, not speedup measurements. Compare native repeats,
independent service restarts, and native/Fork pairs before attributing differences.
Only compare probabilities through the first differing token, while both
sequences still have the same generated history.

Build the long-document QA subset with
`scripts/build_longbench_agentrix_cases.py --data-dir "${DATA_DIR}"
--datasets narrativeqa --model "${MODEL_DIR}" --cases 8 --min-questions 8
--max-questions 8 --unique-questions --min-context-tokens 16384
--max-context-tokens 65536 --output "${CASE_FILE}"`.
Selection uses original document length and distinct-question count; duplicate
questions keep their first source row. No answers or generated outputs are used
to choose cases. Freeze this file before either arm runs. Use
`src/longbench_qa_runner.py --base-url "${ROUTER_URL}/v1" --model "${MODEL_NAME}"
--cases "${CASE_FILE}" --concurrency 16 --max-tokens 256 --document-routing
--prime-first-question --seed "${SEED}" --output "${RESULT_FILE}"` after resetting
the dedicated backend cache. It scores the first original question per document,
then the remaining original questions; both phases count toward total time and
quality. This is a LongBench subset with a controlled arrival pattern, not an
official AgentX replay or a full LongBench evaluation.

The four-service validation protocol uses native-0, fork-0, fork-1, native-1,
with both numeric modes and QA in every service. Preserve `validation-plan.json`,
`progress.json`, frozen `qa-cases.jsonl`/`qa-ready.json`, and labeled reports under
`runs/` on the server. `scripts/summarize_fork_numerics.py --root "${RESULTS_DIR}"
--output "${RESULTS_DIR}/validated-results.json"` (with `benchmark/src` on
`PYTHONPATH`) requires all declared runs to finish. It audits token/digest alignment,
original QA references, recalculated scores, prompt lengths and quality per seed
before combining service pairs. Separately verify runtime hashes, the sole Fork
switch difference, prompt-source counters, execution logs and device ownership.

For official AgentX KV-tiering comparisons, use
`scripts/run_agentx_kv_comparison.py --config "${PRIVATE_CONFIG}" --output "${RESULTS_DIR}/runs"`
with a fresh server-side output directory. The private JSON supplies `cwd`,
`python`, `harness`, `harness_commit`, `server_env`, `client_env`, `backend_url`,
`router_url`, `router` argv, `summarizer` (the existing AgentX `summarize.py`),
and `cells` containing `label`, `server` argv and `client` argv. Client argv
uses `{artifacts}` as its output directory. Keep all private configuration and
artifacts on the server. The controller requires two idle NPUs, verifies the
clean pinned harness checkout, restarts services for each cell, samples backend
metrics, validates the official report and cleans up its own child processes.
Use `sample_npu_experiment.py` alongside it for board-memory observations.
It does not inject lifecycle hints or modify official traces. Configuration and
pending results are tracked in the existing [KV status](../docs/kv_memory_optimization_status.md).
The private internal DP router and its profiling simulators have been removed.
Historical `VLLM_AGENTRIX_DP_ROUTING_POLICY` and `VLLM_FORK_ATTN_DP_PREFIX_*`
controls are no longer supported; old results do not measure the new router.
