"""Restart an existing experiment service and run one recorded retention policy.

Run with the activated Ascend environment. --previous-run must be a directory
containing server.pid/router.pid and run-manifest.json from our experiments.
Correctness checks use --validate-only in a separate service lifetime.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

from compare_retention import read_run


def stop_previous(path):
    for name, expected in (("router", "session_router.py"), ("server", "vllm")):
        pid_file = path / f"{name}.pid"
        if not pid_file.exists():
            # A failed server startup may never have reached router startup.
            continue
        pid = int(pid_file.read_text())
        proc = Path(f"/proc/{pid}")
        if not proc.exists():
            continue
        command = (proc / "cmdline").read_bytes().decode()
        if expected not in command:
            raise RuntimeError(f"PID {pid} does not match the previous {name}")
        os.kill(pid, signal.SIGTERM)
        for _ in range(90):
            if not proc.exists() or (proc / "stat").read_text().split()[2] == "Z":
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"Previous {name} did not stop")


def start(command, path, name, env):
    with (path / f"{name}.log").open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    (path / f"{name}.pid").write_text(f"{process.pid}\n")
    return process


def wait_ready(process, url):
    for _ in range(300):
        if process.poll() is not None:
            raise RuntimeError(f"Process {process.pid} exited: {process.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError(url)


def retention_interval(value):
    if value == "native":
        return None
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("interval must be nonnegative or native")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval", type=retention_interval, required=True)
    parser.add_argument("--prefer-reuse-boundaries", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--validate-only", action="store_true")
    action.add_argument("--serve-only", action="store_true")
    parser.add_argument(
        "--execution-mode", choices=("eager", "decode-graph"), default="eager"
    )
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--npugraph-ex", action="store_true")
    parser.add_argument("--batch-diagnostics", action="store_true")
    parser.add_argument(
        "--cpu-binding", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.interval is None and args.prefer_reuse_boundaries:
        parser.error("prefer-reuse-boundaries requires a non-native interval")
    if args.npugraph_ex and args.execution_mode != "decode-graph":
        parser.error("npugraph-ex requires decode-graph")
    if args.kv_cache_memory_bytes is not None and args.kv_cache_memory_bytes <= 0:
        parser.error("kv-cache-memory-bytes must be positive")
    if args.profile_dir and not args.serve_only:
        parser.error(
            "profile-dir requires serve-only; profiling is separate from benchmarking"
        )
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    run = args.run_dir.resolve()
    previous = args.previous_run.resolve()
    manifest = json.loads((previous / "run-manifest.json").read_text())
    # Check source/runtime consistency before stopping a working service.
    # Track the affinity fix as well as the hybrid-cache patches.
    plugin_files = set(manifest["plugin_sha256"]) | {"vllm_ascend/cpu_binding.py"}
    for name in sorted(plugin_files):
        source = (root / "vllm-ascend" / name).read_bytes()
        installed = (
            Path(sys.prefix) / "lib/python3.11/site-packages" / name
        ).read_bytes()
        if source != installed:
            raise RuntimeError(f"Source/runtime mismatch: {name}")
        manifest["plugin_sha256"][name] = hashlib.sha256(installed).hexdigest()
    run.mkdir(parents=True, exist_ok=False)
    stop_previous(previous)
    env = os.environ.copy()
    # Do not inherit profiler/cache overrides from the controller shell.
    for key in ("PROFILE_DIR", "KV_CACHE_MEMORY_BYTES", "MAMBA_RETENTION_INTERVAL"):
        env.pop(key, None)
    env.update(
        MODEL_NAME="Qwen3.5-9B",
        LONG_PREFILL_TOKEN_THRESHOLD="1024",
        MAMBA_RETENTION_DIAGNOSTICS="1",
        MAMBA_PREFER_REUSE_BOUNDARIES=str(int(args.prefer_reuse_boundaries)),
        EXECUTION_MODE=args.execution_mode,
        ENABLE_NPUGRAPH_EX=str(int(args.npugraph_ex)),
        ENABLE_CPU_BINDING=str(int(args.cpu_binding)),
        BATCH_DIAGNOSTICS=str(int(args.batch_diagnostics)),
    )
    if args.interval is not None:
        env["MAMBA_RETENTION_INTERVAL"] = str(args.interval)
    if args.kv_cache_memory_bytes is not None:
        env["KV_CACHE_MEMORY_BYTES"] = str(args.kv_cache_memory_bytes)
    if args.profile_dir:
        env["PROFILE_DIR"] = str(args.profile_dir.resolve())
    server = start(["bash", "experiments/agentx-ascend/serve.sh"], run, "server", env)
    manifest.update(
        artifacts=str(run / "artifacts"),
        server_pid=server.pid,
        mamba_cache_retention={
            "interval": args.interval,
            "diagnostics": True,
            "prefer_reuse_boundaries": args.prefer_reuse_boundaries,
        },
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        comparison="Recorded hybrid retention and execution configuration",
        validation_only=args.validate_only,
        serve_only=args.serve_only,
        execution_mode=args.execution_mode,
        enforce_eager=args.execution_mode == "eager",
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        profiler_dir=env.get("PROFILE_DIR"),
        compilation_config=(
            {
                "mode": 3,
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 2, 4, 8],
            }
            if args.execution_mode == "decode-graph"
            else None
        ),
        npugraph_ex=args.npugraph_ex,
        cpu_binding=args.cpu_binding,
        batch_diagnostics=args.batch_diagnostics,
        launcher_sha256=hashlib.sha256(
            (root / "experiments/agentx-ascend/serve.sh").read_bytes()
        ).hexdigest(),
    )
    (run / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    wait_ready(server, "http://127.0.0.1:8001")
    manifest["kv_capacity_tokens_per_rank"] = {
        rank: int(tokens.replace(",", ""))
        for rank, tokens in re.findall(
            r"EngineCore_DP(\d+).*GPU KV cache size: ([\d,]+) tokens",
            (run / "server.log").read_text(),
        )
    }
    if len(manifest["kv_capacity_tokens_per_rank"]) != manifest["data_parallel_size"]:
        raise RuntimeError("Could not verify KV cache capacity for every DP rank")
    (run / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    router = start(
        [
            str(root / "agentx/.venv/bin/python"),
            "experiments/agentx-ascend/session_router.py",
            "--policy",
            "sticky",
            "--port",
            "8000",
        ],
        run,
        "router",
        env,
    )
    wait_ready(router, "http://127.0.0.1:8000")
    if args.serve_only:
        print("SERVICE_READY", flush=True)
        return
    if args.validate_only:
        command = [
            sys.executable,
            "experiments/agentx-ascend/hybrid_cache_smoke.py",
            "--output",
            str(run / "correctness.json"),
        ]
        if args.interval is not None:
            command.append("--expect-sparse")
        log_name = "smoke.log"
    else:
        env["ARTIFACT_DIR"] = str(run / "artifacts")
        command = ["bash", "experiments/agentx-ascend/benchmark.sh"]
        log_name = "benchmark.log"
    with (run / log_name).open("w") as log:
        subprocess.run(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    if not args.validate_only:
        subprocess.run(
            [sys.executable, "experiments/agentx-ascend/summarize.py", str(run)],
            check=True,
        )
        with urllib.request.urlopen("http://127.0.0.1:8000/routing-stats") as response:
            (run / "routing-stats.json").write_bytes(response.read())
        read_run(run)  # Freeze diagnostic snapshots before any later probes.
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
