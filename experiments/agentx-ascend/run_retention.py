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
import signal
import subprocess
import sys
import time
import urllib.request

from compare_retention import read_run


def stop_previous(path):
    for name, expected in (("router", "session_router.py"), ("server", "vllm")):
        pid = int((path / f"{name}.pid").read_text())
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, required=True)
    parser.add_argument("--prefer-reuse-boundaries", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.interval < 0:
        parser.error("interval must be nonnegative")
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    run = args.run_dir.resolve()
    previous = args.previous_run.resolve()
    manifest = json.loads((previous / "run-manifest.json").read_text())
    # Check source/runtime consistency before stopping a working service.
    for name in manifest["plugin_sha256"]:
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
    env.update(
        MODEL_NAME="Qwen3.5-9B",
        LONG_PREFILL_TOKEN_THRESHOLD="1024",
        MAMBA_RETENTION_INTERVAL=str(args.interval),
        MAMBA_RETENTION_DIAGNOSTICS="1",
        MAMBA_PREFER_REUSE_BOUNDARIES=str(int(args.prefer_reuse_boundaries)),
    )
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
        comparison="Retention v2: recorded code and policy comparison",
        validation_only=args.validate_only,
    )
    (run / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    wait_ready(server, "http://127.0.0.1:8001")
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
    if args.validate_only:
        command = [
            sys.executable,
            "experiments/agentx-ascend/hybrid_cache_smoke.py",
            "--expect-sparse",
            "--output",
            str(run / "correctness.json"),
        ]
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
