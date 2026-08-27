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


FILE_PATTERN = re.compile(r"sparse_p(?P<probability>\d+\.\d+)_r(?P<repeat>\d+)_requests")
EVENT_PATTERN = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?"
    r"\[(?P<request_id>epd-session-[^]]+)] (?P<message>.*)$"
)
METRICS_ENDPOINTS = {
    "encoder": "http://127.0.0.1:19534/metrics",
    "pd": "http://127.0.0.1:19535/metrics",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--proxy-log", type=Path, required=True)
    parser.add_argument("--max-repeat", type=int)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def proxy_events(path: Path) -> dict[str, dict[str, datetime]]:
    events: dict[str, dict[str, datetime]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        match = EVENT_PATTERN.match(line)
        if not match:
            continue
        timestamp = datetime.strptime(match["time"], "%Y-%m-%d %H:%M:%S,%f")
        message = match["message"]
        request = events[match["request_id"]]
        if "completed with status" in message:
            request["finish"] = timestamp
        elif message.startswith("POST "):
            request["start"] = timestamp
        elif message.startswith("All ") and "encoder requests completed" in message:
            request["encoder_done"] = timestamp
        elif message.startswith("Forwarding to decode"):
            request["pd_start"] = timestamp
    return events


def elapsed_ms(event: dict[str, datetime], start: str, finish: str) -> float | None:
    if start not in event or finish not in event:
        return None
    return (event[finish] - event[start]).total_seconds() * 1000


def gpu_samples(
    telemetry: list[dict[str, object]], start_ns: int, finish_ns: int
) -> dict[int, list[dict[str, float]]]:
    result: dict[int, list[dict[str, float]]] = {0: [], 1: []}
    for row in telemetry:
        if not start_ns <= int(row["monotonic_ns"]) <= finish_ns:
            continue
        for gpu in row.get("gpus", []):  # type: ignore[union-attr]
            result[int(gpu["gpu_id"])].append(gpu)
    return result


def serving_load(
    telemetry: list[dict[str, object]], start_ns: int, finish_ns: int
) -> dict[str, float]:
    output = {}
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
                snapshot = metrics.get(url, {}) if isinstance(metrics, dict) else {}
                if not isinstance(snapshot, dict):
                    continue
                values.extend(
                    float(value)
                    for key, value in snapshot.items()
                    if key.startswith(metric) and isinstance(value, (int, float))
                )
            output[f"{stage}_{label}_mean"] = statistics.mean(values) if values else 0
            output[f"{stage}_{label}_peak"] = max(values, default=0)
    return output


def repeat_rows(
    results_dir: Path, events: dict[str, dict[str, datetime]]
) -> list[dict[str, object]]:
    output = []
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
        visual_requests = [item for item in requests if item["image_path"]]
        text_requests = [item for item in requests if not item["image_path"]]
        samples = gpu_samples(telemetry, start_ns, finish_ns)
        e_util = [sample["gpu_util_percent"] for sample in samples[0]]
        pd_util = [sample["gpu_util_percent"] for sample in samples[1]]
        pair_count = min(len(e_util), len(pd_util))
        pairs = list(zip(e_util[:pair_count], pd_util[:pair_count]))
        e_mean = statistics.mean(e_util)
        pd_mean = statistics.mean(pd_util)
        denominator = math.sqrt(
            sum((value - e_mean) ** 2 for value in e_util[:pair_count])
            * sum((value - pd_mean) ** 2 for value in pd_util[:pair_count])
        )
        correlation = (
            sum((e - e_mean) * (pd - pd_mean) for e, pd in pairs) / denominator
            if denominator
            else 0
        )
        encoder_path = []
        for item in visual_requests:
            value = elapsed_ms(events.get(item["request_id"], {}), "start", "encoder_done")
            if value is not None:
                encoder_path.append(value)
        row: dict[str, object] = {
            **match.groupdict(),
            "requests": len(requests),
            "visual_requests": len(visual_requests),
            "actual_visual_probability": len(visual_requests) / len(requests),
            "errors": sum(item["error"] is not None for item in requests),
            "wall_seconds": (finish_ns - start_ns) / 1e9,
            "latency_mean_ms": statistics.mean(
                float(item["client_latency_ms"]) for item in requests
            ),
            "latency_p95_ms": percentile(
                [float(item["client_latency_ms"]) for item in requests], 0.95
            ),
            "visual_latency_mean_ms": statistics.mean(
                float(item["client_latency_ms"]) for item in visual_requests
            ),
            "text_latency_mean_ms": statistics.mean(
                float(item["client_latency_ms"]) for item in text_requests
            )
            if text_requests
            else "",
            "encoder_path_mean_ms": statistics.mean(encoder_path),
            "gpu_util_correlation": correlation,
            "e_idle_pd_busy_ratio": sum(e < 20 and pd > 60 for e, pd in pairs)
            / pair_count,
            "e_busy_ratio": sum(value >= 20 for value in e_util) / len(e_util),
        }
        for gpu_id in (0, 1):
            for field, label in (
                ("gpu_util_percent", "util"),
                ("memory_util_percent", "memory_util"),
                ("memory_used_mib", "memory_used_mib"),
                ("power_watts", "power_watts"),
            ):
                values = [float(item[field]) for item in samples[gpu_id]]
                row[f"gpu{gpu_id}_{label}_mean"] = statistics.mean(values)
                row[f"gpu{gpu_id}_{label}_peak"] = max(values)
        row.update(serving_load(telemetry, start_ns, finish_ns))
        output.append(row)
    return output


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["probability"])].append(row)
    fields = [
        field
        for field in rows[0]
        if field
        not in {
            "probability",
            "repeat",
            "requests",
            "visual_requests",
            "errors",
        }
    ]
    output = []
    for probability, group in sorted(grouped.items(), key=lambda item: float(item[0])):
        row: dict[str, object] = {
            "probability": probability,
            "repeats": len(group),
            "requests_per_repeat": group[0]["requests"],
            "errors": sum(int(item["errors"]) for item in group),
        }
        for field in fields:
            values = [float(item[field]) for item in group if item[field] != ""]
            row[f"{field}_mean"] = statistics.mean(values) if values else ""
            row[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repeats = repeat_rows(args.results_dir, proxy_events(args.proxy_log))
    if args.max_repeat is not None:
        repeats = [
            row for row in repeats if int(str(row["repeat"])) <= args.max_repeat
        ]
    if not repeats:
        raise ValueError(f"no temporal matrix results found in {args.results_dir}")
    write_csv(args.results_dir / "repeat_summary.csv", repeats)
    write_csv(args.results_dir / "aggregate_summary.csv", aggregate_rows(repeats))
    print(f"summarized {len(repeats)} repeats in {args.results_dir}")


if __name__ == "__main__":
    main()
