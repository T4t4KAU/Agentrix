from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_samples(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def cache_capacity_tokens(path: Path) -> list[int]:
    if not path.exists():
        return []
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("vllm:cache_config_info"):
            continue
        match = re.search(r'kv_cache_size_tokens="(\d+)"', line)
        if match:
            values.append(int(match.group(1)))
    return values


def summarize_memory(
    samples: list[dict[str, Any]], capacities: list[int]
) -> dict[str, Any]:
    if not samples:
        raise ValueError("resource sample stream is empty")

    gpu_ids = sorted(
        {int(gpu["index"]) for sample in samples for gpu in sample.get("gpus", [])}
    )
    gpu_peak = {
        str(index): max(
            (
                float(gpu["memory_used_mib"])
                for sample in samples
                for gpu in sample.get("gpus", [])
                if int(gpu["index"]) == index
            ),
            default=0.0,
        )
        for index in gpu_ids
    }
    aggregate_gpu = [
        sum(float(gpu["memory_used_mib"]) for gpu in sample.get("gpus", []))
        for sample in samples
        if sample.get("gpus")
    ]
    aggregate_gpu_total = [
        sum(float(gpu["memory_total_mib"]) for gpu in sample.get("gpus", []))
        for sample in samples
        if sample.get("gpus")
    ]
    memory_controller = [
        float(gpu["memory_controller_utilization_percent"])
        for sample in samples
        for gpu in sample.get("gpus", [])
    ]
    server_rss = [float(sample["process_tree_rss_kib"]) / 1024 for sample in samples]
    application_rss = [
        float(sample.get("application_tree_rss_kib", 0)) / 1024 for sample in samples
    ]
    kv_rows = [sample["kv_usage"] for sample in samples if sample.get("kv_usage")]
    kv_width = max((len(row) for row in kv_rows), default=0)
    per_engine_kv_peak = [
        max((float(row[index]) for row in kv_rows if index < len(row)), default=0.0)
        for index in range(kv_width)
    ]
    aggregate_kv_usage = [
        sum(float(value) for value in row) / len(row) for row in kv_rows if row
    ]
    live_token_rows = [
        sum(float(usage) * capacity for usage, capacity in zip(row, capacities))
        for row in kv_rows
        if len(row) == len(capacities)
    ]
    started = float(samples[0]["time"])
    ended = float(samples[-1]["time"])
    warm_gpu = min(aggregate_gpu, default=0.0)
    warm_server_rss = min(server_rss, default=0.0)
    return {
        "sample_count": len(samples),
        "sampled_duration_seconds": max(0.0, ended - started),
        "gpu": {
            "ids": gpu_ids,
            "peak_used_mib_per_gpu": gpu_peak,
            "peak_aggregate_used_mib": max(aggregate_gpu, default=0.0),
            "warm_min_aggregate_used_mib": warm_gpu,
            "peak_delta_from_warm_mib": max(aggregate_gpu, default=0.0) - warm_gpu,
            "aggregate_total_mib": max(aggregate_gpu_total, default=0.0),
            "peak_memory_controller_utilization_percent": max(
                memory_controller, default=0.0
            ),
        },
        "server_process_tree": {
            "peak_rss_mib": max(server_rss, default=0.0),
            "warm_min_rss_mib": warm_server_rss,
            "peak_delta_from_warm_mib": max(server_rss, default=0.0) - warm_server_rss,
        },
        "application_process_tree": {
            "peak_rss_mib": max(application_rss, default=0.0),
        },
        "kv_cache": {
            "capacity_tokens_per_engine": capacities,
            "aggregate_capacity_tokens": sum(capacities),
            "peak_usage_percent_per_engine": [
                100 * value for value in per_engine_kv_peak
            ],
            "peak_aggregate_usage_percent": 100 * max(aggregate_kv_usage, default=0.0),
            "peak_aggregate_live_tokens": round(max(live_token_rows, default=0.0)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize coding-agent GPU, KV, server, and application memory"
    )
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_memory(
        load_samples(args.samples), cache_capacity_tokens(args.metrics)
    )
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
