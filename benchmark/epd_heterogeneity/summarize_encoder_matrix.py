from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


FILE_PATTERN = re.compile(r"(?P<bucket>.+)_c(?P<concurrency>\d+)_r(?P<repeat>\d+)_requests")
EVENT_PATTERN = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?"
    r"\[(?P<request_id>epd-encoder-[^]]+)] (?P<message>.*)$"
)
METRICS_ENDPOINTS = {
    "encoder": "http://127.0.0.1:19534/metrics",
    "pd": "http://127.0.0.1:19535/metrics",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--proxy-log", type=Path, required=True)
    parser.add_argument("--direct-encoder", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[position]


def parse_proxy_events(path: Path) -> dict[str, dict[str, datetime]]:
    events: dict[str, dict[str, datetime]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        match = EVENT_PATTERN.match(line)
        if not match:
            continue
        timestamp = datetime.strptime(match["time"], "%Y-%m-%d %H:%M:%S,%f")
        message = match["message"]
        request = events[match["request_id"]]
        if "completed with status" in message:
            request["proxy_finish"] = timestamp
        elif message.startswith("POST "):
            request["proxy_start"] = timestamp
        elif message.startswith("All ") and "encoder requests completed" in message:
            request["encoder_done"] = timestamp
        elif message.startswith("Forwarding to decode"):
            request["pd_start"] = timestamp
    return events


def elapsed_ms(event: dict[str, datetime], start: str, finish: str) -> float | None:
    if start not in event or finish not in event:
        return None
    return (event[finish] - event[start]).total_seconds() * 1000


def gpu_stats(
    telemetry: list[dict[str, object]],
    start_ns: int,
    finish_ns: int,
    gpu_id: int,
) -> dict[str, float]:
    samples = [
        gpu
        for row in telemetry
        if start_ns <= int(row["monotonic_ns"]) <= finish_ns
        for gpu in row.get("gpus", [])  # type: ignore[union-attr]
        if gpu["gpu_id"] == gpu_id
    ]
    result = {}
    for field, label in (
        ("gpu_util_percent", "util"),
        ("memory_util_percent", "memory_util"),
        ("power_watts", "power_watts"),
        ("memory_used_mib", "memory_used_mib"),
    ):
        values = [float(sample[field]) for sample in samples]
        result[f"gpu{gpu_id}_{label}_mean"] = statistics.mean(values)
        result[f"gpu{gpu_id}_{label}_peak"] = max(values)
    result[f"gpu{gpu_id}_samples"] = float(len(samples))
    return result


def serving_load_stats(
    telemetry: list[dict[str, object]],
    start_ns: int,
    finish_ns: int,
) -> dict[str, float]:
    result = {}
    for stage, url in METRICS_ENDPOINTS.items():
        for metric, label in (
            ("vllm:num_requests_running", "running"),
            ("vllm:num_requests_waiting", "waiting"),
            ("vllm:kv_cache_usage_perc", "kv_cache_usage"),
        ):
            values = []
            for row in telemetry:
                if not start_ns <= int(row["monotonic_ns"]) <= finish_ns:
                    continue
                metrics = row.get("metrics", {})
                if not isinstance(metrics, dict):
                    continue
                snapshot = metrics.get(url, {})
                if not isinstance(snapshot, dict):
                    continue
                values.extend(
                    float(value)
                    for key, value in snapshot.items()
                    if key.startswith(metric) and isinstance(value, (int, float))
                )
            result[f"{stage}_{label}_mean"] = (
                statistics.mean(values) if values else 0
            )
            result[f"{stage}_{label}_peak"] = max(values, default=0)
    return result


def repeat_rows(
    results_dir: Path,
    events: dict[str, dict[str, datetime]],
    direct_encoder: bool,
) -> list[dict[str, object]]:
    rows = []
    for request_path in sorted(results_dir.glob("*_requests.jsonl")):
        match = FILE_PATTERN.fullmatch(request_path.stem)
        if match is None:
            continue
        requests = [
            json.loads(line)
            for line in request_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        telemetry_path = request_path.with_name(
            request_path.name.replace("_requests.jsonl", "_telemetry.jsonl")
        )
        telemetry = [
            json.loads(line)
            for line in telemetry_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        start_ns = min(int(item["client_start_monotonic_ns"]) for item in requests)
        finish_ns = max(int(item["client_finish_monotonic_ns"]) for item in requests)
        duration_seconds = (finish_ns - start_ns) / 1e9
        client = [float(item["client_latency_ms"]) for item in requests]
        encoder = []
        downstream = []
        for item in requests:
            event = events.get(item["request_id"], {})
            encoder_ms = elapsed_ms(event, "proxy_start", "encoder_done")
            downstream_ms = elapsed_ms(event, "pd_start", "proxy_finish")
            if encoder_ms is not None:
                encoder.append(encoder_ms)
            if downstream_ms is not None:
                downstream.append(downstream_ms)
        if direct_encoder:
            encoder = client.copy()
        visual_tokens = sum(int(item["expected_visual_tokens"]) for item in requests)
        row: dict[str, object] = {
            **match.groupdict(),
            "requests": len(requests),
            "errors": sum(item["error"] is not None for item in requests),
            "visual_tokens_per_request": requests[0]["expected_visual_tokens"],
            "wall_seconds": duration_seconds,
            "visual_tokens_per_second": visual_tokens / duration_seconds,
            "client_mean_ms": statistics.mean(client),
            "client_p50_ms": percentile(client, 0.50),
            "client_p95_ms": percentile(client, 0.95),
            "encoder_path_mean_ms": statistics.mean(encoder) if encoder else "",
            "encoder_path_p95_ms": percentile(encoder, 0.95) if encoder else "",
            "downstream_mean_ms": statistics.mean(downstream) if downstream else "",
        }
        row.update(gpu_stats(telemetry, start_ns, finish_ns, 0))
        row.update(gpu_stats(telemetry, start_ns, finish_ns, 1))
        row.update(serving_load_stats(telemetry, start_ns, finish_ns))
        rows.append(row)
    return rows


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["bucket"]), str(row["concurrency"]))].append(row)
    aggregate = []
    fields = (
        "visual_tokens_per_second",
        "client_mean_ms",
        "client_p95_ms",
        "encoder_path_mean_ms",
        "gpu0_util_mean",
        "gpu0_util_peak",
        "gpu0_memory_util_mean",
        "gpu0_memory_util_peak",
        "gpu0_power_watts_mean",
        "gpu0_power_watts_peak",
        "gpu0_memory_used_mib_mean",
        "gpu0_memory_used_mib_peak",
        "gpu1_util_mean",
        "gpu1_util_peak",
        "gpu1_power_watts_mean",
        "gpu1_power_watts_peak",
        "gpu1_memory_used_mib_mean",
        "gpu1_memory_used_mib_peak",
        "encoder_running_mean",
        "encoder_running_peak",
        "encoder_waiting_mean",
        "encoder_waiting_peak",
        "pd_running_mean",
        "pd_running_peak",
        "pd_waiting_mean",
        "pd_waiting_peak",
    )
    for (bucket, concurrency), group in sorted(grouped.items()):
        row: dict[str, object] = {
            "bucket": bucket,
            "concurrency": concurrency,
            "repeats": len(group),
            "errors": sum(int(item["errors"]) for item in group),
            "visual_tokens_per_request": group[0]["visual_tokens_per_request"],
        }
        for field in fields:
            values = [float(item[field]) for item in group]
            row[f"{field}_mean"] = statistics.mean(values)
            row[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0
        aggregate.append(row)
    return aggregate


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    events = parse_proxy_events(args.proxy_log)
    repeats = repeat_rows(args.results_dir, events, args.direct_encoder)
    if not repeats:
        raise ValueError(f"no encoder matrix results found in {args.results_dir}")
    write_csv(args.results_dir / "repeat_summary.csv", repeats)
    write_csv(args.results_dir / "aggregate_summary.csv", aggregate_rows(repeats))
    print(f"summarized {len(repeats)} repeats in {args.results_dir}")


if __name__ == "__main__":
    main()
