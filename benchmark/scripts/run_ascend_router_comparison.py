"""Run the official routing matrix with an isolated Ascend DP launch config.

The private JSON supplies argv/env/cwd. All logs and results stay in --output.
This does not include the historical CUDA-only prefix-router implementation.
"""

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import psutil


def fork_attention_command(command, enabled, min_shared_tokens):
    """Keep both arms identical except for the existing ForkAttention switch."""
    command = list(command)
    option = "--additional-config"
    if option in command:
        index = command.index(option) + 1
        additional = json.loads(command[index])
    else:
        command.extend([option, "{}"])
        index = len(command) - 1
        additional = {}
    additional["fork_attention"] = {
        **additional.get("fork_attention", {}),
        "enabled": enabled,
        "min_shared_tokens": min_shared_tokens,
        "diagnostics": True,
    }
    command[index] = json.dumps(additional)
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--router-python", required=True, type=Path)
    parser.add_argument("--qa-cases", type=Path)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--qa-max-tokens", type=int, default=256)
    parser.add_argument("--qa-prefill-gate", action="store_true")
    parser.add_argument("--qa-fork-attention", action="store_true")
    parser.add_argument("--fork-scale", action="store_true")
    parser.add_argument("--scale-trials", type=int, default=3)
    parser.add_argument(
        "--qa-fork-min-shared-tokens",
        "--fork-min-shared-tokens",
        type=int,
        default=32768,
    )
    parser.add_argument("--qa-arrival", choices=("waves", "fanout"), default="waves")
    args = parser.parse_args()
    if args.fork_scale and (
        args.qa_cases or args.qa_prefill_gate or args.qa_fork_attention
    ):
        parser.error("fork-scale is a separate fixed-length workload; omit QA flags")
    if args.scale_trials < 1:
        parser.error("scale-trials must be positive")
    fork_comparison = args.qa_fork_attention or args.fork_scale
    if args.qa_fork_attention and (not args.qa_cases or args.qa_prefill_gate):
        parser.error("qa-fork-attention requires qa-cases and excludes prefill-gate")
    if args.qa_fork_min_shared_tokens < 128 or args.qa_fork_min_shared_tokens % 128:
        parser.error("qa-fork-min-shared-tokens must be a positive multiple of 128")
    if args.qa_prefill_gate and not args.qa_cases:
        parser.error("qa-prefill-gate requires qa-cases")
    if args.trials < 1 or args.qa_max_tokens < 1:
        parser.error("trials and qa-max-tokens must be positive")
    output = args.output.resolve()
    if (output / "progress.json").exists():
        raise FileExistsError("Choose a fresh result directory")
    config = json.loads(args.config.read_text())
    command = config["argv"]
    port = int(command[command.index("--port") + 1])
    model = command[command.index("--served-model-name") + 1]
    backend = f"http://127.0.0.1:{port}"
    router_url = f"http://127.0.0.1:{port + 1}"
    scripts = Path(__file__).parent
    processes = []
    completed = []

    def save(name, data):
        path = output / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        tmp.replace(path)

    def status(phase, **details):
        data = {"phase": phase, "time": time.time(), "completed": completed, **details}
        save("progress.json", data)
        print(json.dumps(data), flush=True)

    def start(argv, name, env=None):
        with (output / (name + ".log")).open("w") as log:
            proc = subprocess.Popen(
                argv,
                env=env,
                cwd=config["cwd"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        processes.append(proc)
        save(name + "-process.json", {"pid": proc.pid, "argv": argv})
        return proc

    def stop(proc):
        try:
            children = psutil.Process(proc.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=45)
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
                raise RuntimeError(f"Startup failed: {proc.pid}")
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                pass
            time.sleep(2)
        raise TimeoutError("Service health check")

    def wait_idle():
        deadline = time.monotonic() + 86400
        while time.monotonic() < deadline:
            state = subprocess.check_output(["npu-smi", "info"], text=True)
            if state.count("No running processes found") == 2:
                return
            status("waiting_for_two_idle_npus")
            time.sleep(15)
        raise TimeoutError("Two idle NPUs unavailable")

    def cancel(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        subprocess.run(
            [str(args.router_python), str(scripts / "serve_dp_router.py"), "--check"],
            check=True,
        )
        schedule = [
            (20260927, ["consistent_hash", "native", "cache_aware"]),
            (20261020, ["cache_aware", "native", "consistent_hash"]),
        ]
        if args.qa_prefill_gate:
            schedule = [
                (20260927, ["consistent_hash", "consistent_hash_gate"]),
                (20261020, ["consistent_hash_gate", "consistent_hash"]),
            ]
        if fork_comparison:
            schedule = [
                (20260927, ["consistent_hash_fork", "consistent_hash"]),
                (20261020, ["consistent_hash", "consistent_hash_fork"]),
            ]
        for seed, policies in schedule:
            for policy in policies:
                for trial in range(args.trials):
                    wait_idle()
                    label = f"{seed}-{policy}-{trial}"
                    status("server_startup", label=label)
                    server_command = (
                        fork_attention_command(
                            command,
                            policy == "consistent_hash_fork",
                            args.qa_fork_min_shared_tokens,
                        )
                        if fork_comparison
                        else command
                    )
                    server = start(server_command, label + "-server", config["env"])
                    ready(server, backend)
                    for workload in (
                        ["fork-scale"]
                        if args.fork_scale
                        else ["longbench"]
                        if args.qa_cases
                        else ["sessions", "revisit", "replicated", "cold"]
                    ):
                        cell = label + "-" + workload
                        router = None
                        url = backend
                        if policy != "native":
                            router = start(
                                [
                                    str(args.router_python),
                                    str(scripts / "serve_dp_router.py"),
                                    "--worker-urls",
                                    backend,
                                    "--policy",
                                    "consistent_hash"
                                    if policy
                                    in {"consistent_hash_gate", "consistent_hash_fork"}
                                    else policy,
                                    "--intra-node-data-parallel-size",
                                    "2",
                                    "--host",
                                    "127.0.0.1",
                                    "--port",
                                    str(port + 1),
                                    "--prometheus-port",
                                    str(port + 2),
                                ],
                                cell + "-router",
                            )
                            ready(router, router_url)
                            url = router_url
                        name = (
                            "benchmark_agent_session_dp.py"
                            if workload == "sessions"
                            else "benchmark_prefix_aware_dp.py"
                        )
                        bench = [
                            command[0],
                            str(scripts / name),
                            "--base-url",
                            url,
                            "--control-url",
                            backend,
                            "--model",
                            model,
                            "--policy-label",
                            policy,
                            "--seed",
                            str(seed),
                            "--trials",
                            "1",
                            "--output-tokens",
                            "16",
                            "--output",
                            str(output / (cell + ".json")),
                        ]
                        if workload != "sessions":
                            bench += [
                                "--allow-missing-prompt-details",
                                "--workload",
                                workload,
                                "--documents",
                                "12",
                                "--prefix-tokens",
                                "4096",
                                "--suffix-tokens",
                                "64",
                                "--revisit-order",
                                "shuffled",
                            ]
                        if args.qa_cases:
                            bench = [
                                command[0],
                                str(scripts.parent / "src/longbench_qa_runner.py"),
                                "--base-url",
                                url + "/v1",
                                "--model",
                                model,
                                "--cases",
                                str(args.qa_cases),
                                "--seed",
                                str(seed),
                                "--concurrency",
                                "4",
                                "--max-tokens",
                                str(args.qa_max_tokens),
                                "--document-routing",
                                "--output",
                                str(output / (cell + ".json")),
                            ]
                            if args.qa_arrival == "waves" and not args.qa_prefill_gate:
                                bench.append("--question-waves")
                            if (
                                args.qa_prefill_gate
                                and policy == "consistent_hash_gate"
                            ):
                                bench.append("--coalesce-prefill")
                            with urllib.request.urlopen(
                                backend + "/metrics", timeout=10
                            ) as response:
                                (output / (cell + "-metrics-before.txt")).write_bytes(
                                    response.read()
                                )
                        if args.fork_scale:
                            bench = [
                                command[0],
                                str(scripts / "benchmark_fork_scale.py"),
                                "--base-url",
                                url,
                                "--control-url",
                                backend,
                                "--model",
                                model,
                                "--seed",
                                str(seed),
                                "--trials",
                                str(args.scale_trials),
                                "--output",
                                str(output / (cell + ".json")),
                            ]
                        status("benchmark", cell=cell)
                        client = start(bench, cell, config["env"])
                        code = client.wait(timeout=7200 if args.fork_scale else 900)
                        processes.remove(client)
                        if code:
                            raise RuntimeError(f"Benchmark failed: {cell}")
                        if args.qa_cases:
                            time.sleep(2)
                            with urllib.request.urlopen(
                                backend + "/metrics", timeout=10
                            ) as response:
                                (output / (cell + "-metrics-after.txt")).write_bytes(
                                    response.read()
                                )
                        completed.append(cell)
                        if router is not None:
                            stop(router)
                    stop(server)
        save("complete.json", {"complete": True, "completed": completed})
        status("complete")
    except BaseException as error:
        status("failed", error=repr(error))
        raise
    finally:
        for proc in processes.copy()[::-1]:
            stop(proc)


if __name__ == "__main__":
    main()
