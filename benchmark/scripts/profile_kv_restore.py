#!/usr/bin/env python3
"""Exercise KV backup and restore under a deliberately small GPU cache."""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def request_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9012)
    parser.add_argument("--load-budget", type=int, default=512)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"Port {args.port} is already occupied")
    connector = {
        "kv_connector": "LMCacheConnectorV1",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {
            "lmcache.max_tokens_per_load": args.load_budget,
        },
    }
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        "agentrix-restore",
        "--dtype",
        "float16",
        "--max-model-len",
        "1536",
        "--num-gpu-blocks-override",
        "96",
        "--max-num-seqs",
        "2",
        "--gpu-memory-utilization",
        "0.7",
        "--enforce-eager",
        "--enable-prefix-caching",
        "--attention-backend",
        "FLASH_ATTN",
        "--generation-config",
        "vllm",
        "--kv-transfer-config",
        json.dumps(connector),
    ]
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=f"{root / 'LMCache'}:{root / 'vllm'}",
        LMCACHE_CONFIG_FILE=str(args.config.resolve()),
        VLLM_USE_FLASHINFER_SAMPLER="0",
        VLLM_AGENTRIX_KV_PROACTIVE_BACKUP="1",
        VLLM_AGENTRIX_KV_PROACTIVE_ASYNC="1",
        VLLM_AGENTRIX_KV_BACKUP_HIGH_WATERMARK="0.8",
        PYTHONHASHSEED="0",
    )
    endpoint = f"http://127.0.0.1:{args.port}"
    results = []
    with (args.output / "vllm_server.log").open("w") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 420
            while True:
                if process.poll() is not None:
                    raise RuntimeError("vLLM exited during startup; inspect server log")
                try:
                    request_json(endpoint + "/v1/models")
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("vLLM startup timed out") from None
                    time.sleep(1)
            for label in ("P", "A", "B", "C", "D", "A"):
                # Distinct first tokens prevent cross-prompt prefix reuse.
                prompt = [1000 + ord(label)] * 857
                started = time.perf_counter()
                response = request_json(
                    endpoint + "/v1/completions",
                    {
                        "model": "agentrix-restore",
                        "prompt": prompt,
                        "max_tokens": 1,
                        "temperature": 0,
                        "seed": 42,
                    },
                )
                results.append(
                    {
                        "label": label,
                        "latency_ms": (time.perf_counter() - started) * 1000,
                        "usage": response["usage"],
                        "text": response["choices"][0]["text"],
                    }
                )
                time.sleep(0.2)
            if results[1]["text"] != results[-1]["text"]:
                raise AssertionError("Restored request changed deterministic output")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            (args.output / "results.json").write_text(json.dumps(results, indent=2))
    server_log = (args.output / "vllm_server.log").read_text()
    retrieved = [
        int(count) for count in re.findall(r"Retrieved (\d+) out of", server_log)
    ]
    if not retrieved or max(retrieved) == 0:
        raise AssertionError("Workload did not exercise an external KV restore")
    print(json.dumps({"requests": results, "retrieved_tokens": retrieved}, indent=2))


if __name__ == "__main__":
    main()
