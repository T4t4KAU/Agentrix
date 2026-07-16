from __future__ import annotations

import argparse
import json
import os
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
            processes[int(entry.name)] = (ppid, int(rss_match.group(1)) if rss_match else 0)
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


def sample_gpu() -> list[int]:
    result = subprocess.run(
        ("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"),
        text=True, capture_output=True, timeout=5, check=True,
    )
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def sample_kv(url: str) -> list[float]:
    with urllib.request.urlopen(url, timeout=3) as response:
        text = response.read().decode()
    values = []
    for line in text.splitlines():
        if line.startswith("vllm:gpu_cache_usage_perc"):
            values.append(float(line.rsplit(" ", 1)[1]))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        while running and Path(f"/proc/{args.server_pid}").exists():
            record = {"time": time.time()}
            try:
                record["gpu_memory_used_mib"] = sample_gpu()
            except Exception as error:
                record["gpu_error"] = str(error)
            try:
                record["kv_usage"] = sample_kv(args.metrics_url)
            except Exception as error:
                record["kv_error"] = str(error)
            record["process_tree_rss_kib"] = process_tree_rss_kib(args.server_pid)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
