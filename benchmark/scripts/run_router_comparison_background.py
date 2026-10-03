"""Replay archived prefix routing against the pinned official router.

Run on the experiment server. The archive supplies the frozen model launch
recipe and runtime; raw artifacts and private commands stay in the output dir.
"""

import argparse
import hashlib
import importlib.util
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--router-python", required=True, type=Path)
    parser.add_argument("--port", type=int, default=19300)
    args = parser.parse_args()
    run = args.output.resolve()
    if (run / "progress.json").exists():
        raise FileExistsError("Use a fresh experiment directory")
    spec = importlib.util.spec_from_file_location(
        "archived_controller", args.archive / "controller.py"
    )
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    legacy.RUN = run
    legacy.PORT = args.port
    legacy.BASE = f"http://127.0.0.1:{args.port}"
    backend = legacy.BASE
    router_url = f"http://127.0.0.1:{args.port + 1}"
    scripts = Path(__file__).parent
    router = None
    client = None
    completed = []

    # Every arm uses precisely the same archived engine/runtime. Only the
    # prefix routing environment flag and external router policy differ.
    for variant in ("old", "new"):
        (run / f"runtime-{variant}").symlink_to(
            (args.archive / "runtime-new").resolve()
        )

    def gpu_jobs():
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "0,1",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return {
            int(row.strip()) for row in output.splitlines() if row.strip().isdigit()
        }

    def wait_for_idle():
        deadline = time.monotonic() + 86400
        consecutive = 0
        while time.monotonic() < deadline:
            jobs = gpu_jobs()
            consecutive = 0 if jobs else consecutive + 1
            legacy.status(
                phase="waiting_for_two_idle_gpus",
                occupied_processes=len(jobs),
                completed=completed,
            )
            if consecutive >= 2:
                return
            time.sleep(15)
        raise TimeoutError("GPUs did not become idle within 24 hours")

    def stop_router():
        nonlocal router
        if router is not None and router.poll() is None:
            router.terminate()
            try:
                router.wait(timeout=15)
            except subprocess.TimeoutExpired:
                router.kill()
                router.wait()
        router = None

    def start_router(policy, label):
        nonlocal router
        if policy in ("native", "prefix_aware"):
            return backend
        command = [
            str(args.router_python),
            str(scripts / "serve_dp_router.py"),
            "--worker-urls",
            backend,
            "--policy",
            policy,
            "--intra-node-data-parallel-size",
            "2",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port + 1),
            "--prometheus-port",
            str(args.port + 2),
        ]
        legacy.save(label + "-router-command.json", command)
        with (run / (label + "-router.log")).open("w") as log:
            router = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if router.poll() is not None:
                raise RuntimeError("Official router exited during startup")
            try:
                with urllib.request.urlopen(
                    router_url + "/health", timeout=2
                ) as response:
                    if response.status == 200:
                        return router_url
            except OSError:
                pass
            time.sleep(1)
        raise TimeoutError("Router startup")

    def cancel(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        subprocess.run(
            [str(args.router_python), str(scripts / "serve_dp_router.py"), "--check"],
            check=True,
        )
        manifest_files = [args.archive / "controller.py", *scripts.glob("*.py")]
        manifest_files.extend(
            (args.archive / "runtime-new/vllm/v1/engine").glob("prefix_router.py")
        )
        legacy.save(
            "source-manifest.json",
            {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in manifest_files
            },
        )
        for seed, policies in [
            (20260927, ["native", "prefix_aware", "consistent_hash", "cache_aware"]),
            (20261020, ["cache_aware", "consistent_hash", "prefix_aware", "native"]),
        ]:
            for policy in policies:
                for trial in range(3):
                    wait_for_idle()
                    label = f"{seed}-{policy}-{trial}"
                    env = legacy.launch(
                        label, "new" if policy == "prefix_aware" else "native"
                    )
                    legacy.quality(label)
                    for workload, documents, tokens in [
                        ("revisit", 12, 32),
                        ("replicated", 16, 256),
                        ("cold", 12, 32),
                        ("sessions", 12, 32),
                    ]:
                        cell = f"{label}-{workload}"
                        url = start_router(policy, cell)
                        script = (
                            "benchmark_agent_session_dp.py"
                            if workload == "sessions"
                            else "benchmark_prefix_aware_dp.py"
                        )
                        command = [
                            str(legacy.PY),
                            str(scripts / script),
                            "--base-url",
                            url,
                            "--control-url",
                            backend,
                            "--model",
                            "agentrix-dp",
                            "--policy-label",
                            policy,
                            "--seed",
                            str(seed),
                            "--trials",
                            "1",
                            "--output-tokens",
                            str(tokens),
                            "--output",
                            str(run / (cell + ".json")),
                        ]
                        if workload != "sessions":
                            command += [
                                "--workload",
                                workload,
                                "--documents",
                                str(documents),
                                "--prefix-tokens",
                                "4096",
                                "--suffix-tokens",
                                "64",
                                "--revisit-order",
                                "shuffled",
                            ]
                        legacy.save(cell + "-command.json", command)
                        legacy.status(phase="benchmark", cell=cell, completed=completed)
                        with (run / (cell + ".log")).open("w") as log:
                            client = subprocess.Popen(
                                command, env=env, stdout=log, stderr=subprocess.STDOUT
                            )
                            code = client.wait(timeout=900)
                        if code:
                            raise RuntimeError(f"Failed benchmark: {cell}")
                        owned = {legacy.SERVICE.pid} | {
                            p.pid
                            for p in psutil.Process(legacy.SERVICE.pid).children(
                                recursive=True
                            )
                        }
                        if gpu_jobs() - owned:
                            raise RuntimeError(
                                "Another GPU job appeared; this cell is contaminated"
                            )
                        completed.append(cell)
                        stop_router()
                    legacy.stop()
        legacy.save("complete.json", {"complete": True, "completed": completed})
        legacy.status(phase="complete", completed=completed)
    except BaseException as error:
        legacy.status(phase="failed", error=repr(error), completed=completed)
        raise
    finally:
        if client is not None and client.poll() is None:
            client.terminate()
            try:
                client.wait(timeout=10)
            except subprocess.TimeoutExpired:
                client.kill()
                client.wait()
        stop_router()
        legacy.stop()


if __name__ == "__main__":
    main()
