from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


running = True


def stop(*_: object) -> None:
    global running
    running = False


def process_tree_rss_kib(root_pid: int) -> int:
    processes: dict[int, tuple[int, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
            ppid = int(re.search(r"^PPid:\s+(\d+)", status, re.M).group(1))
            rss_match = re.search(r"^VmRSS:\s+(\d+)\s+kB", status, re.M)
            processes[int(entry.name)] = (
                ppid,
                int(rss_match.group(1)) if rss_match else 0,
            )
        except (FileNotFoundError, PermissionError, AttributeError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _) in processes.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(processes.get(pid, (0, 0))[1] for pid in descendants)


def sample_gpu(gpu_ids: set[int] | None = None) -> list[dict[str, float | int]]:
    result = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.memory",
            "--format=csv,noheader,nounits",
        ),
        text=True,
        capture_output=True,
        timeout=5,
        check=True,
    )
    samples = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, used, total, utilization = (item.strip() for item in line.split(","))
        gpu_index = int(index)
        if gpu_ids is not None and gpu_index not in gpu_ids:
            continue
        samples.append(
            {
                "index": gpu_index,
                "memory_used_mib": float(used),
                "memory_total_mib": float(total),
                "memory_controller_utilization_percent": float(utilization),
            }
        )
    return samples


def sample_kv(url: str) -> list[float]:
    with urllib.request.urlopen(url, timeout=3) as response:
        text = response.read().decode()
    by_name: dict[str, list[tuple[int, float]]] = {
        "vllm:kv_cache_usage_perc": [],
        "vllm:gpu_cache_usage_perc": [],
    }
    for line in text.splitlines():
        for name in by_name:
            if not line.startswith(name):
                continue
            engine = re.search(r'engine="(\d+)"', line)
            engine_index = int(engine.group(1)) if engine else len(by_name[name])
            by_name[name].append((engine_index, float(line.rsplit(" ", 1)[1])))
            break
    values = by_name["vllm:kv_cache_usage_perc"]
    if not values:
        values = by_name["vllm:gpu_cache_usage_perc"]
    return [value for _, value in sorted(values)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--application-pid-file", type=Path)
    args = parser.parse_args()
    gpu_ids = (
        {int(item) for item in args.gpu_ids.split(",") if item.strip()}
        if args.gpu_ids
        else None
    )
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        while running and Path(f"/proc/{args.server_pid}").exists():
            record = {"time": time.time()}
            try:
                gpu_samples = sample_gpu(gpu_ids)
                record["gpus"] = gpu_samples
                record["gpu_memory_used_mib"] = [
                    sample["memory_used_mib"] for sample in gpu_samples
                ]
            except Exception as error:
                record["gpu_error"] = str(error)
            try:
                record["kv_usage"] = sample_kv(args.metrics_url)
            except Exception as error:
                record["kv_error"] = str(error)
            record["process_tree_rss_kib"] = process_tree_rss_kib(args.server_pid)
            if args.application_pid_file is not None:
                try:
                    application_pid = int(
                        args.application_pid_file.read_text(encoding="utf-8").strip()
                    )
                    record["application_tree_rss_kib"] = process_tree_rss_kib(
                        application_pid
                    )
                except (FileNotFoundError, ValueError):
                    record["application_tree_rss_kib"] = 0
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
