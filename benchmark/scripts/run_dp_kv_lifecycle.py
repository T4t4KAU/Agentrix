"""Run a detached, reproducible Ascend lifecycle matrix from a private config.

Config contains env/cwd/router_python and cases with argv/modes/label. Results
and process metadata must stay on the experiment server. Never stop other jobs.
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import psutil
from benchmark_agent_kv_tiering import save_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a failed matrix, preserving valid cells",
    )
    parser.add_argument(
        "--extend", action="store_true", help="Add cases to a completed matrix"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    completed = []
    if (output / "progress.json").exists():
        old_progress = json.loads((output / "progress.json").read_text())
        permitted = (args.resume and old_progress["phase"] == "failed") or (
            args.extend and old_progress["phase"] == "complete"
        )
        if (args.resume and args.extend) or not permitted:
            raise FileExistsError(
                "Use --resume for a failed matrix or --extend for a completed matrix"
            )
        completed = old_progress["completed"]
        for cell in completed:
            if not json.loads((output / f"{cell}.json").read_text()).get("valid"):
                raise ValueError(f"Previously completed cell is invalid: {cell}")
        planned = {
            f"{case['label']}-{mode}-{seed}"
            for case in config["cases"]
            for mode in case["modes"]
            for seed in config.get("seeds", [20261001, 20261020])
        }
        if not set(completed).issubset(planned):
            raise ValueError("Cannot remove previously completed cases")
        for case in config["cases"]:
            if any(cell.startswith(case["label"] + "-") for cell in completed):
                prior = json.loads(
                    (output / f"{case['label']}-server-process.json").read_text()
                )
                if prior["argv"] != case["argv"]:
                    raise ValueError("Cannot change a completed case's backend command")
        # Keep failures and their script versions for source provenance.
        archive = output / f"attempt-{time.time_ns()}"
        archive.mkdir()
        for path in output.glob("*.log"):
            shutil.copy2(path, archive / path.name)
        for path in output.glob("*.json"):
            if (
                path.name not in {f"{cell}.json" for cell in completed}
                and path.name != "launch-private.json"
            ):
                shutil.copy2(path, archive / path.name)
        for case in config["cases"]:
            for seed in config.get("seeds", [20261001, 20261020]):
                for mode in case["modes"]:
                    cell = f"{case['label']}-{mode}-{seed}"
                    path = output / f"{cell}.json"
                    if cell not in completed and path.exists():
                        path.rename(archive / path.name)
        summary = output / "validated-summary.json"
        if summary.exists():
            # Its archived copy remains valid for the old plan, not this one.
            summary.unlink()
    scripts = Path(__file__).parent
    processes = []

    def status(phase, **extra):
        state = {"phase": phase, "time": time.time(), "completed": completed, **extra}
        save_result(output / "progress.json", state)
        print(json.dumps(state), flush=True)

    def start(argv, label, env=None):
        with (output / f"{label}.log").open("w") as log:
            proc = subprocess.Popen(
                argv,
                cwd=config["cwd"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        processes.append(proc)
        save_result(output / f"{label}-process.json", {"pid": proc.pid, "argv": argv})
        return proc

    def stop(proc):
        try:
            children = psutil.Process(proc.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(children, timeout=10)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        processes.remove(proc)

    def ready(proc, url):
        deadline = time.monotonic() + 480
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("Service exited; inspect its server log")
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                pass
            time.sleep(2)
        raise TimeoutError("Service startup")

    def cancel(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    manifest = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in scripts.glob("*.py")
    }
    save_result(output / "source-manifest.json", manifest)
    try:
        for case in config["cases"]:
            planned = {
                f"{case['label']}-{mode}-{seed}"
                for mode in case["modes"]
                for seed in config.get("seeds", [20261001, 20261020])
            }
            if planned.issubset(completed):
                continue
            state = subprocess.check_output(["npu-smi", "info"], text=True)
            if state.count("No running processes found") != 2:
                raise RuntimeError("Both NPUs must be idle before starting a case")
            argv = case["argv"]
            port = int(argv[argv.index("--port") + 1])
            backend = f"http://127.0.0.1:{port}"
            router_url = f"http://127.0.0.1:{port + 1}"
            model = argv[argv.index("--served-model-name") + 1]
            tokenizer = argv[argv.index("--model") + 1]
            label = case["label"]
            status("server_startup", case=label)
            server = start(argv, label + "-server", config["env"])
            ready(server, backend)
            router = start(
                [
                    config["router_python"],
                    str(scripts / "serve_dp_router.py"),
                    "--worker-urls",
                    backend,
                    "--policy",
                    "consistent_hash",
                    "--intra-node-data-parallel-size",
                    "2",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port + 1),
                    "--prometheus-port",
                    str(port + 2),
                ],
                label + "-router",
            )
            ready(router, router_url)
            for seed_index, seed in enumerate(
                config.get("seeds", [20261001, 20261020])
            ):
                modes = case["modes"] if seed_index == 0 else case["modes"][::-1]
                for mode in modes:
                    cell = f"{label}-{mode}-{seed}"
                    if cell in completed:
                        continue
                    status("benchmark", cell=cell)
                    client = start(
                        [
                            argv[0],
                            str(scripts / "benchmark_dp_kv_lifecycle.py"),
                            "--base-url",
                            router_url,
                            "--control-url",
                            backend,
                            "--model",
                            model,
                            "--tokenizer",
                            tokenizer,
                            "--mode",
                            mode,
                            "--seed",
                            str(seed),
                            "--output-tokens",
                            "8",
                            "--trials",
                            str(config.get("trials", 2)),
                            "--output",
                            str(output / f"{cell}.json"),
                        ],
                        cell,
                        config["env"],
                    )
                    code = client.wait(timeout=1800)
                    processes.remove(client)
                    if code:
                        raise RuntimeError(f"Benchmark failed: {cell}")
                    result = json.loads((output / f"{cell}.json").read_text())
                    if not result.get("valid"):
                        raise RuntimeError(f"Invalid result: {cell}")
                    completed.append(cell)
            stop(router)
            stop(server)
        status("complete")
    except BaseException as error:
        status("failed", error=repr(error))
        raise
    finally:
        for proc in processes.copy()[::-1]:
            stop(proc)


if __name__ == "__main__":
    main()
