"""Run official AgentX cells from a server-private launch config.

Each cell supplies server/router/client argv and environments. The client argv
uses {artifacts} for its fresh output directory. No trace or hint injection is
performed. Keep config, raw metrics, logs and results on the experiment server.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import psutil


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    save(output / "launch-private.json", config)
    (output / "launch-private.json").chmod(0o600)
    harness = Path(config["harness"])
    commit = subprocess.check_output(
        ["git", "-C", str(harness), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(harness), "status", "--porcelain"], text=True
    )
    if commit != config["harness_commit"] or dirty:
        raise ValueError("Official harness must match the clean pinned revision")
    save(
        output / "source-manifest.json",
        {
            "harness_commit": commit,
            "harness_sources": {
                str(p.relative_to(harness)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (harness / "src").rglob("*.py")
            },
            "controller_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
    )
    completed = []
    active = []

    def status(phase, **extra):
        state = dict(phase=phase, time=time.time(), completed=completed, **extra)
        save(output / "progress.json", state)
        print(json.dumps(state), flush=True)

    def start(argv, env, directory, label):
        with (directory / f"{label}.log").open("x") as log:
            proc = subprocess.Popen(
                argv,
                env=env,
                cwd=config["cwd"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        active.append(proc)
        save(directory / f"{label}-process.json", {"pid": proc.pid, "argv": argv})
        return proc

    def ready(proc, url):
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("Service exited during startup")
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                pass
            time.sleep(2)
        raise TimeoutError("Service startup exceeded 600 seconds")

    def cancel(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        status("tokenizer_preflight")
        checked = set()
        for cell in config["cells"]:
            client = cell["client"]
            tokenizer = client[client.index("--tokenizer") + 1]
            client_python = str(Path(client[0]).with_name("python"))
            if (client_python, tokenizer) in checked:
                continue
            with (output / "tokenizer-preflight.log").open("a") as log:
                subprocess.run(
                    [
                        client_python,
                        "-c",
                        (
                            "import sys, pathlib, aiperf; "
                            "from aiperf.common.tokenizer import Tokenizer; "
                            "assert pathlib.Path(aiperf.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[2]).resolve()); "
                            "Tokenizer.from_pretrained(sys.argv[1]); print('Tokenizer loaded')"
                        ),
                        tokenizer,
                        str(harness),
                    ],
                    env=config["client_env"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=120,
                )
            checked.add((client_python, tokenizer))
        for cell in config["cells"]:
            label = cell["label"]
            state = subprocess.check_output(["npu-smi", "info"], text=True)
            if state.count("No running processes found") != 2:
                raise RuntimeError(
                    "Both NPUs must be idle; refusing to disturb other jobs"
                )
            directory = output / label
            directory.mkdir()
            status("server_startup", cell=label)
            server = start(cell["server"], config["server_env"], directory, "server")
            ready(server, config["backend_url"])
            router = start(config["router"], config["client_env"], directory, "router")
            ready(router, config["router_url"])
            save(
                directory / "run-manifest.json",
                {"tensor_parallel_size": 1, "data_parallel_size": 2, "cell": cell},
            )
            status("benchmark", cell=label)
            argv = [
                arg.replace("{artifacts}", str(directory / "artifacts"))
                for arg in cell["client"]
            ]
            client = start(argv, config["client_env"], directory, "benchmark")
            deadline = time.monotonic() + 3000
            with (directory / "metrics.jsonl").open("x") as metrics:
                while client.poll() is None:
                    if server.poll() is not None or router.poll() is not None:
                        raise RuntimeError("Serving process exited during benchmark")
                    if time.monotonic() > deadline:
                        raise TimeoutError("AgentX exceeded 3000 seconds")
                    with urllib.request.urlopen(
                        config["backend_url"] + "/metrics", timeout=10
                    ) as response:
                        metrics.write(
                            json.dumps(
                                {
                                    "time": time.time(),
                                    "metrics": response.read().decode(),
                                }
                            )
                            + "\n"
                        )
                        metrics.flush()
                    time.sleep(5)
            if client.returncode:
                raise RuntimeError(f"AgentX exited with {client.returncode}")
            subprocess.run(
                [config["python"], config["summarizer"], str(directory)],
                check=True,
                env=config["client_env"],
                stdout=subprocess.DEVNULL,
            )
            summary = json.loads((directory / "result-summary.json").read_text())
            if (
                not summary["submission_valid"]
                or summary["was_cancelled"]
                or summary["request_errors"]
            ):
                raise RuntimeError(
                    "Official report failed validation; preserving all evidence"
                )
            completed.append(label)
            for proc in reversed(active):
                stop(proc)
            active.clear()
            status("cell_complete", cell=label)
        status("complete")
    except BaseException as exc:
        status("failed", error=repr(exc))
        raise
    finally:
        for proc in reversed(active):
            stop(proc)


if __name__ == "__main__":
    main()
