from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


running = True


def stop(*_: object) -> None:
    global running
    running = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--metrics-url", action="append", default=[])
    return parser.parse_args()


def gpu_samples(gpu_ids: set[int]) -> list[dict[str, float | int]]:
    fields = (
        "index,memory.used,memory.total,utilization.gpu,utilization.memory,"
        "power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory"
    )
    result = subprocess.run(
        (
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ),
        text=True,
        capture_output=True,
        timeout=5,
        check=True,
    )
    samples = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        gpu_id = int(values[0])
        if gpu_id not in gpu_ids:
            continue
        samples.append(
            {
                "gpu_id": gpu_id,
                "memory_used_mib": float(values[1]),
                "memory_total_mib": float(values[2]),
                "gpu_util_percent": float(values[3]),
                "memory_util_percent": float(values[4]),
                "power_watts": float(values[5]),
                "temperature_c": float(values[6]),
                "sm_clock_mhz": float(values[7]),
                "memory_clock_mhz": float(values[8]),
            }
        )
    return samples


def relevant_metrics(url: str) -> dict[str, float]:
    with urllib.request.urlopen(url, timeout=3) as response:
        body = response.read().decode()
    wanted = (
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:gpu_cache_usage_perc",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:time_to_first_token_seconds_count",
        "vllm:inter_token_latency_seconds_sum",
        "vllm:inter_token_latency_seconds_count",
        "vllm:request_time_per_output_token_seconds_sum",
        "vllm:request_time_per_output_token_seconds_count",
        "vllm:e2e_request_latency_seconds_sum",
        "vllm:e2e_request_latency_seconds_count",
        "vllm:request_queue_time_seconds_sum",
        "vllm:request_queue_time_seconds_count",
        "vllm:request_prefill_time_seconds_sum",
        "vllm:request_prefill_time_seconds_count",
        "vllm:request_decode_time_seconds_sum",
        "vllm:request_decode_time_seconds_count",
    )
    metrics: dict[str, float] = {}
    for line in body.splitlines():
        if line.startswith("#") or not line:
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name not in wanted:
            continue
        try:
            metrics[line.rsplit(" ", 1)[0]] = float(line.rsplit(" ", 1)[1])
        except ValueError:
            continue
    return metrics


def main() -> None:
    args = parse_args()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    gpu_ids = {int(value) for value in args.gpu_ids.split(",") if value}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        while running:
            before = time.monotonic()
            record: dict[str, object] = {
                "wall_time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
            }
            try:
                record["gpus"] = gpu_samples(gpu_ids)
            except Exception as error:
                record["gpu_error"] = repr(error)
            metrics = {}
            for url in args.metrics_url:
                try:
                    metrics[url] = relevant_metrics(url)
                except Exception as error:
                    metrics[url] = {"error": repr(error)}
            record["metrics"] = metrics
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            elapsed = time.monotonic() - before
            time.sleep(max(0, args.interval - elapsed))


if __name__ == "__main__":
    main()
