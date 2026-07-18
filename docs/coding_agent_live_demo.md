# Coding-Agent Live Demo

## Purpose

This demo presents the original DP=4 coding-agent workload as a simultaneous,
split-screen comparison on one eight-GPU host. Agentrix owns GPUs 0–3 and the
vLLM baseline owns GPUs 4–7. Both sides receive the same cases, branch
instructions, release time, generation parameters, and output limit. The
default live run uses 12 cases and 16 branches per case.

The main area shows real SSE output from active coding subagents. A compact
header provides supporting evidence without turning the first view into a
monitoring dashboard: generated tokens, completed requests, active and queued
requests, output rate, HBM, KV occupancy, and per-engine running/waiting load.

Colors have stable meanings:

- teal identifies Agentrix and slate identifies the baseline;
- yellow is prefill, green is generating, gray is queued, cyan is complete,
  and red is an error;
- purple identifies shared-prefix and KV information;
- the routing view uses red, yellow, blue, and green for the four prefix
  affinity classes; and
- agent text remains neutral white for readability.

Symbols accompany every color so that recordings, low-color terminals, and
color-vision deficiencies do not make the state ambiguous.

## Compared Systems

The left side runs ForkAttention, prefix-aware DP routing, and application
prompt compaction. The right side runs FlashAttention, ordinary internal DP,
and the uncompressed prompt. Tool-KV trimming and predicted TTL are disabled on
both sides. Each side starts a one-token bootstrap request per case before
releasing the branch fanout, matching the established coding-agent benchmark
methodology.

Live token counts include exact completed usage plus a tokenizer estimate for
responses still in flight. When a request completes, the estimate is replaced
by the exact usage returned by the server. DP rank load comes from labeled
vLLM Prometheus metrics. The four aligned routing rows show the actual rank
selected by the API server for every sequence. Dot color is derived from the
prefix affinity learned from the Agentrix placement, so an affinity-preserving
rank is nearly monochrome while ordinary DP mixes colors across ranks.

## Running

Install the optional UI dependency once:

```bash
cd benchmark
uv sync --extra agent --extra demo
```

Preview the interface without GPUs or model servers:

```bash
MOCK=1 benchmark/scripts/run_coding_agent_demo.sh
```

Run the live eight-GPU comparison after the benchmark GPUs are free:

```bash
MODEL_PATH=/path/to/Qwen3-32B \
  benchmark/scripts/run_coding_agent_demo.sh
```

The launcher prints the latest real Agentrix vLLM startup event every two
seconds while both servers start in parallel: model shard loading, compilation,
KV-cache initialization, CUDA-graph capture, and API readiness. Empty log
matches are explicitly non-fatal and cannot stop either server. The baseline
log and the complete unfiltered Agentrix log remain on disk.

Before loading either model, the launcher terminates only processes recorded by
the previous demo run and listeners on the demo's configured API/telemetry
ports. It waits for those ports to become free and records fresh PID files for
reliable cleanup on the next run; it does not issue a broad `pkill` against
unrelated vLLM experiments.

The demo defaults to 12 distinct commit-derived Django cases with 16 branches
per case (192 agent requests per side). This keeps the original coding-agent
workload construction intact while leaving enough time to observe both live
token streams. Set `CASE_COUNT=4` for the shorter 64-request preview.

When both servers have already been prestarted and warmed, launch only the TUI
without loading the model again:

```bash
CLIENT_ONLY=1 benchmark/scripts/run_coding_agent_demo.sh
```

Environment variables can override `CASES_PATH`, `CASE_OFFSET`, `CASE_COUNT`,
`ROUNDS`, `MAX_TOKENS`, ports, model names, and the two GPU lists. Press Space
to freeze or resume rendering without pausing inference, and press Q to stop
the demo and both model services.

Server logs are written under `benchmark/results/coding_agent_demo/`. The live
run should only be used after both model servers have warmed up; the controller
waits for both health endpoints and then releases both workloads through one
shared start gate.

## Host-Side Browser Dashboard

The browser dashboard runs on the graphical host while inference remains on
the GPU server. One authenticated SSH control connection starts both remote
DP=4 services and forwards three loopback-only endpoints: the Agentrix API, the
baseline API, and read-only GPU telemetry. The browser talks only to the local
dashboard over WebSocket. No vLLM or telemetry port is exposed publicly, and
the SSH password is never stored in a script or browser.

After the current GPU experiment has finished and this working tree has been
synced to the server, configure a local SSH alias (for example,
`agentrix-demo`) outside the repository and run on the host:

```bash
SSH_TARGET=agentrix-demo benchmark/scripts/run_coding_agent_demo_web_host.sh
```

The script deliberately contains no server address or credentials. If an SSH
alias is not used, supply `SSH_TARGET` and the optional `SSH_PORT` through the
local environment. Authentication remains the responsibility of the SSH
client or agent.

The SSH client prompts for authentication, waits for both remote model servers,
starts the dashboard on `http://127.0.0.1:8088`, and opens the default browser.
Closing the dashboard process closes the SSH control connection; the remote
launcher then terminates both vLLM services and the telemetry helper.

The browser can also be previewed locally without starting the SSH tunnel or
using a GPU:

```bash
PYTHONPATH=benchmark/src:application/src \
  benchmark/.venv/bin/python -m coding_agent_demo_web \
  --cases benchmark/data/django_agentrix/cases_30k_b16.jsonl --mock
```

The Web UI keeps agent streams as the dominant visual area. Supporting data is
limited to a compact strip and can be read at a glance: token work, completed
agents, active/queued requests, HBM, KV occupancy, per-DP-rank load, and the
measured throughput ratio. Token counters animate as SSE fragments arrive,
while exact server usage replaces the in-flight estimate at completion.
