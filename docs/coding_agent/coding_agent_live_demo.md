# Coding-Agent Live Demo

The terminal and browser clients display two coding-agent workloads side by
side. They remain available as Python entry points. The old launcher depended
on private DP controls that have been removed; server startup and SSH tunnels
are configured separately.

## Preview without model servers

Install `benchmark[demo]`, then run from the repository root:

```bash
PYTHONPATH=benchmark/src:application/src \
  benchmark/.venv/bin/python -m coding_agent_demo_tui \
  --cases "${CASE_FILE}" --mock

PYTHONPATH=benchmark/src:application/src \
  benchmark/.venv/bin/python -m coding_agent_demo_web \
  --cases "${CASE_FILE}" --mock --no-open-browser
```

`CASE_FILE` must contain coding cases built with `build_django_agentrix_cases.py`.
Mock mode previews the interface; it does not provide benchmark measurements.

## Connect to running services

Start two dedicated OpenAI-compatible services and configure their model names
and endpoints. Keep remote access details in private SSH configuration.

```bash
PYTHONPATH=benchmark/src:application/src \
  benchmark/.venv/bin/python -m coding_agent_demo_web \
  --cases "${CASE_FILE}" \
  --left-base-url "${LEFT_BASE_URL}" --left-model "${LEFT_MODEL}" \
  --right-base-url "${RIGHT_BASE_URL}" --right-model "${RIGHT_MODEL}" \
  --no-open-browser
```

Use `--left-gpus` and `--right-gpus` for local GPU telemetry, or
`--left-gpu-metrics-url` and `--right-gpu-metrics-url` for dedicated remote
telemetry endpoints. The clients do not start or terminate model services.
For controlled routing measurements, use the [official DP guide](../dp_routing.md).

The interface shows streamed agent text, completed and queued requests, token
usage, KV occupancy and available GPU telemetry. In-flight token estimates are
replaced by exact server usage on completion. UI comparisons alone do not
establish task-quality or routing gains.
