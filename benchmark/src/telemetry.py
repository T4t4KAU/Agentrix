from __future__ import annotations

import argparse
import json
import math
import re
import signal
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable


NVIDIA_SMI_FIELDS = (
    "index",
    "uuid",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
)
PROMETHEUS_METRICS = {
    "vllm:kv_cache_usage_perc",
    "vllm:kv_offload_cpu_cache_usage_perc",
    "vllm:kv_offload_cpu_cache_occupancy_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
}


def read_gpu_samples(gpu_ids: set[str] | None = None) -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(NVIDIA_SMI_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    samples = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(NVIDIA_SMI_FIELDS):
            continue
        index, uuid, compute, memory, used, total = values
        if gpu_ids and index not in gpu_ids and uuid not in gpu_ids:
            continue
        samples.append(
            {
                "index": index,
                "uuid": uuid,
                "compute_utilization_percent": _number_or_none(compute),
                "memory_bandwidth_utilization_percent": _number_or_none(memory),
                "memory_used_mib": _number_or_none(used),
                "memory_total_mib": _number_or_none(total),
            }
        )
    return samples


def parse_prometheus(text: str) -> dict[str, list[float]]:
    metrics: dict[str, list[float]] = {}
    pattern = re.compile(
        r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
        r"(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)$"
    )
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if not match or match.group("name") not in PROMETHEUS_METRICS:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if math.isfinite(value):
            metrics.setdefault(match.group("name"), []).append(value)
    return metrics


def read_prometheus(url: str) -> dict[str, list[float]]:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return parse_prometheus(response.read().decode("utf-8", errors="replace"))
    except OSError:
        return {}


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    raw_sample_count = len(samples)
    samples = _request_window(samples)
    gpu_fields = (
        "compute_utilization_percent",
        "memory_bandwidth_utilization_percent",
        "memory_used_mib",
    )
    gpu_values: dict[str, list[float]] = {field: [] for field in gpu_fields}
    per_gpu: dict[str, dict[str, list[float]]] = {}
    prometheus_samples: dict[str, list[float]] = {
        metric: [] for metric in PROMETHEUS_METRICS
    }

    for sample in samples:
        for gpu in sample.get("gpus", []):
            gpu_id = str(gpu.get("index", "unknown"))
            fields = per_gpu.setdefault(gpu_id, {field: [] for field in gpu_fields})
            for field in gpu_fields:
                value = gpu.get(field)
                if isinstance(value, (int, float)):
                    gpu_values[field].append(float(value))
                    fields[field].append(float(value))
        current_prometheus: dict[str, list[float]] = {
            metric: [] for metric in PROMETHEUS_METRICS
        }
        for endpoint in sample.get("vllm", []):
            for metric, values in endpoint.get("metrics", {}).items():
                if metric in current_prometheus:
                    current_prometheus[metric].extend(float(value) for value in values)
        for metric, values in current_prometheus.items():
            if not values:
                continue
            if metric.endswith("_perc"):
                prometheus_samples[metric].append(100 * statistics.fmean(values))
            else:
                prometheus_samples[metric].append(sum(values))

    aggregate = {
        field: _distribution(values) for field, values in gpu_values.items() if values
    }
    aggregate.update(
        {
            _prometheus_summary_name(metric): _distribution(values)
            for metric, values in prometheus_samples.items()
            if values
        }
    )
    return {
        "sample_count": len(samples),
        "raw_sample_count": raw_sample_count,
        "aggregate": aggregate,
        "gpus": {
            gpu_id: {
                field: _distribution(values)
                for field, values in fields.items()
                if values
            }
            for gpu_id, fields in sorted(per_gpu.items())
        },
    }


def _request_window(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active_indices = [
        index for index, sample in enumerate(samples) if _has_queued_request(sample)
    ]
    if not active_indices:
        return samples
    return samples[active_indices[0] : active_indices[-1] + 1]


def _has_queued_request(sample: dict[str, Any]) -> bool:
    for endpoint in sample.get("vllm", []):
        metrics = endpoint.get("metrics", {})
        for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
            if any(float(value) > 0 for value in metrics.get(name, [])):
                return True
    return False


def monitor(
    output: Path,
    gpu_ids: set[str] | None,
    metrics_urls: list[str],
    interval_seconds: float,
) -> None:
    running = True

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    samples: list[dict[str, Any]] = []
    while running:
        started = time.monotonic()
        samples.append(
            {
                "timestamp": time.time(),
                "gpus": read_gpu_samples(gpu_ids),
                "vllm": [
                    {"url": url, "metrics": read_prometheus(url)}
                    for url in metrics_urls
                ],
            }
        )
        remaining = interval_seconds - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)

    payload = {
        "interval_seconds": interval_seconds,
        "gpu_ids": sorted(gpu_ids or []),
        "metrics_urls": metrics_urls,
        "summary": summarize_samples(samples),
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _number_or_none(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _distribution(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "max": max(ordered),
    }


def _percentile(ordered: list[float], percentile: float) -> float:
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _prometheus_summary_name(metric: str) -> str:
    return {
        "vllm:kv_cache_usage_perc": "gpu_kv_cache_usage_percent",
        "vllm:kv_offload_cpu_cache_usage_perc": "cpu_kv_cache_usage_percent",
        "vllm:kv_offload_cpu_cache_occupancy_perc": ("cpu_kv_cache_occupancy_percent"),
        "vllm:num_requests_running": "requests_running",
        "vllm:num_requests_waiting": "requests_waiting",
    }[metric]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--metrics-url", action="append", default=[])
    parser.add_argument("--interval-seconds", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")
    gpu_ids = {value.strip() for value in args.gpu_ids.split(",") if value.strip()}
    monitor(args.output, gpu_ids or None, args.metrics_url, args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
