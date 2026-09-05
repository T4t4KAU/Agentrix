#!/usr/bin/env python3
"""Run one isolated TraceLab configuration with owned server processes."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path


def require_free_port(port: int) -> None:
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} is occupied; refusing to reuse a service")


@contextlib.contextmanager
def service(
    command: list[str], environment: dict[str, str], log_path: Path
) -> Iterator[subprocess.Popen]:
    with log_path.open("x") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            yield process
        finally:
            # Workers can outlive the API process after a startup failure.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def wait_for_service(process: subprocess.Popen, port: int, *, health: bool) -> None:
    deadline = time.monotonic() + (600 if health else 60)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Service exited during startup; inspect its log")
        try:
            if health:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=2
                ):
                    return
            else:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    raise TimeoutError(f"Service startup timed out on port {port}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--backend", choices=("FLASH_ATTN", "FORK_ATTN"), default="FLASH_ATTN"
    )
    parser.add_argument(
        "--policy", choices=("native", "session_aware"), default="native"
    )
    parser.add_argument("--gpu-events", action="store_true")
    parser.add_argument("--active-placement", action="store_true")
    parser.add_argument("--tier-config", type=Path)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--gpu-blocks", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-inflight", type=int, default=512)
    parser.add_argument("--port", type=int, default=8145)
    parser.add_argument("--load-budget", type=int, default=4096)
    args = parser.parse_args()
    if args.max_num_seqs < 1 or args.max_inflight < 1 or args.gpu_blocks < 0:
        parser.error(
            "Sequence/in-flight limits must be positive; GPU blocks nonnegative"
        )
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("GPU memory utilization must be in (0, 1]")
    root = Path(__file__).resolve().parents[2]
    workload = json.loads(args.workload.read_text())
    tier_config = args.tier_config.read_text() if args.tier_config else None
    args.output.mkdir(parents=True, exist_ok=False)
    require_free_port(args.port)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("VLLM_", "LMCACHE_")) and key != "PYTHONPATH"
    }
    environment.update(
        PYTHONPATH=str(args.runtime_root.resolve()),
        CUDA_VISIBLE_DEVICES=args.gpu_ids,
        PYTHONHASHSEED="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_USE_FLASHINFER_SAMPLER="0",
        VLLM_SERVER_DEV_MODE="1",
        VLLM_PLUGINS="",
        VLLM_AGENTRIX_DP_ROUTING_POLICY=args.policy,
        VLLM_AGENTRIX_DP_KV_EVENTS=str(int(args.gpu_events)),
        VLLM_AGENTRIX_KV_PLACEMENT_ACTIVE=str(int(args.active_placement)),
        VLLM_FORK_ATTN_ENABLE_FOREST="1",
        VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH="1",
        VLLM_FORK_ATTN_FANOUT_SCHEDULING_ENABLED="0",
    )
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        "tracelab-qwen",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        args.backend,
        "--generation-config",
        "vllm",
        "--data-parallel-size",
        "2",
        "--api-server-count",
        "1",
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--no-async-scheduling",
        "--enforce-eager",
        "--block-size",
        "16",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(workload["metadata"]["max_model_len"]),
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        str(args.max_num_seqs),
    ]
    if args.gpu_blocks:
        command.extend(["--num-gpu-blocks-override", str(args.gpu_blocks)])
    if args.tier_config:
        config_snapshot = args.output.resolve() / "lmcache_config.yaml"
        assert tier_config is not None
        config_snapshot.write_text(tier_config)
        environment.update(
            PYTHONPATH=f"{args.runtime_root.resolve()}:{root / 'LMCache'}",
            LMCACHE_CONFIG_FILE=str(config_snapshot),
            VLLM_AGENTRIX_KV_PROACTIVE_BACKUP="1",
            VLLM_AGENTRIX_KV_PROACTIVE_ASYNC="1",
            VLLM_AGENTRIX_KV_BACKUP_HIGH_WATERMARK="0.8",
        )
        command.extend(
            [
                "--kv-transfer-config",
                json.dumps(
                    {
                        "kv_connector": "LMCacheConnectorV1",
                        "kv_role": "kv_both",
                        "kv_load_failure_policy": "recompute",
                        "kv_connector_extra_config": {
                            "lmcache.max_tokens_per_load": args.load_budget
                        },
                    }
                ),
            ]
        )
    identity = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import vllm,torch; print(vllm.__file__, vllm.__version__, torch.__version__)",
        ],
        env=environment,
        text=True,
    )
    (args.output / "configuration.json").write_text(
        json.dumps(
            {
                "command": command,
                "environment": {
                    key: value
                    for key, value in environment.items()
                    if key.startswith(
                        ("VLLM_", "LMCACHE_", "CUDA_VISIBLE", "PYTHONPATH")
                    )
                },
                "runtime_identity": identity,
                "benchmark_sha256": {
                    str(path.relative_to(root)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in (
                        Path(__file__).resolve(),
                        root / "benchmark/src/tracelab_replay.py",
                        root / "benchmark/src/tracelab_workload.py",
                        root / "benchmark/src/tracelab_timeline.py",
                    )
                },
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            },
            indent=2,
        )
    )
    with contextlib.ExitStack() as stack:
        if args.tier_config:
            # The checked-in tier configuration uses these loopback services.
            for name, port, options in (
                ("mooncake_http_metadata_server", 8005, ["--port", "8005"]),
                (
                    "mooncake_master",
                    50051,
                    ["--rpc_address", "127.0.0.1", "--rpc_port", "50051", "-v=1"],
                ),
            ):
                require_free_port(port)
                process = stack.enter_context(
                    service(
                        [str(Path(sys.executable).parent / name), *options],
                        environment,
                        args.output / f"{name}.log",
                    )
                )
                wait_for_service(process, port, health=False)
        server = stack.enter_context(
            service(command, environment, args.output / "vllm_server.log")
        )
        wait_for_service(server, args.port, health=True)
        stack.enter_context(
            service(
                ["nvidia-smi", "dmon", "-i", args.gpu_ids, "-s", "pucvmet", "-d", "1"],
                environment,
                args.output / "nvidia_dmon.log",
            )
        )
        subprocess.run(
            [
                sys.executable,
                str(root / "benchmark/src/tracelab_replay.py"),
                "--workload",
                str(args.workload),
                "--output",
                str(args.output / "replay"),
                "--base-url",
                f"http://127.0.0.1:{args.port}",
                "--label",
                args.label,
                "--max-inflight",
                str(args.max_inflight),
            ],
            check=True,
            timeout=3600,
        )


if __name__ == "__main__":
    main()
