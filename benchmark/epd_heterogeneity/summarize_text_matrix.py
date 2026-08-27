from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


FILE_PATTERN = re.compile(
    r"(?P<workload>prefill|decode)_i(?P<input_tokens>\d+)_o(?P<output_tokens>\d+)"
    r"_c(?P<concurrency>\d+)_r(?P<repeat>\d+)_requests"
)
METRICS_URL = "http://127.0.0.1:19535/metrics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def metric_value(metrics: dict[str, float], name: str) -> float | None:
    values = [value for key, value in metrics.items() if key.startswith(name)]
    return sum(values) if values else None


def histogram_delta_mean(
    telemetry: list[dict[str, object]], name: str
) -> float | None:
    snapshots = [
        row.get("metrics", {}).get(METRICS_URL, {})  # type: ignore[union-attr]
        for row in telemetry
    ]
    snapshots = [item for item in snapshots if isinstance(item, dict)]
    if len(snapshots) < 2:
        return None
    first_sum = metric_value(snapshots[0], f"{name}_sum")
    last_sum = metric_value(snapshots[-1], f"{name}_sum")
    first_count = metric_value(snapshots[0], f"{name}_count")
    last_count = metric_value(snapshots[-1], f"{name}_count")
    if None in (first_sum, last_sum, first_count, last_count):
        return None
    count = float(last_count) - float(first_count)
    if count <= 0:
        return None
    return (float(last_sum) - float(first_sum)) / count


def gpu_stats(
    telemetry: list[dict[str, object]], start_ns: int, finish_ns: int, gpu_id: int
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
        result[f"gpu{gpu_id}_{label}_mean"] = (
            statistics.mean(values) if values else ""
        )
        result[f"gpu{gpu_id}_{label}_peak"] = max(values) if values else ""
    return result


def pd_load_stats(
    telemetry: list[dict[str, object]], start_ns: int, finish_ns: int
) -> dict[str, float]:
    result = {}
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
            snapshot = metrics.get(METRICS_URL, {})
            if not isinstance(snapshot, dict):
                continue
            values.extend(
                float(value)
                for key, value in snapshot.items()
                if key.startswith(metric) and isinstance(value, (int, float))
            )
        result[f"pd_{label}_mean"] = statistics.mean(values) if values else ""
        result[f"pd_{label}_peak"] = max(values) if values else ""
    return result


def repeat_rows(results_dir: Path) -> list[dict[str, object]]:
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
        duration = (finish_ns - start_ns) / 1e9
        client = [float(item["client_latency_ms"]) for item in requests]
        workload = match["workload"]
        usage_field = "prompt_tokens" if workload == "prefill" else "completion_tokens"
        measured_tokens = sum(int(item["usage"][usage_field]) for item in requests)
        row: dict[str, object] = {
            **match.groupdict(),
            "requests": len(requests),
            "errors": sum(item["error"] is not None for item in requests),
            "wall_seconds": duration,
            "tokens_per_second": measured_tokens / duration,
            "client_mean_ms": statistics.mean(client),
            "client_p50_ms": percentile(client, 0.50),
            "client_p95_ms": percentile(client, 0.95),
        }
        for metric, label in (
            ("vllm:time_to_first_token_seconds", "ttft_seconds"),
            ("vllm:request_time_per_output_token_seconds", "tpot_seconds"),
            ("vllm:request_queue_time_seconds", "queue_seconds"),
            ("vllm:request_prefill_time_seconds", "prefill_seconds"),
            ("vllm:request_decode_time_seconds", "decode_seconds"),
        ):
            value = histogram_delta_mean(telemetry, metric)
            row[label] = "" if value is None else value
        row.update(gpu_stats(telemetry, start_ns, finish_ns, 0))
        row.update(gpu_stats(telemetry, start_ns, finish_ns, 1))
        row.update(pd_load_stats(telemetry, start_ns, finish_ns))
        rows.append(row)
    return rows


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(
            str(row[field])
            for field in ("workload", "input_tokens", "output_tokens", "concurrency")
        )
        grouped[key].append(row)
    fields = (
        "tokens_per_second",
        "client_mean_ms",
        "client_p95_ms",
        "ttft_seconds",
        "tpot_seconds",
        "queue_seconds",
        "prefill_seconds",
        "decode_seconds",
        "gpu0_util_mean",
        "gpu0_util_peak",
        "gpu1_util_mean",
        "gpu1_util_peak",
        "gpu1_memory_util_mean",
        "gpu1_memory_util_peak",
        "gpu1_power_watts_mean",
        "gpu1_power_watts_peak",
        "gpu0_memory_used_mib_mean",
        "gpu0_memory_used_mib_peak",
        "gpu1_memory_used_mib_mean",
        "gpu1_memory_used_mib_peak",
        "pd_running_mean",
        "pd_running_peak",
        "pd_waiting_mean",
        "pd_waiting_peak",
        "pd_kv_cache_usage_mean",
        "pd_kv_cache_usage_peak",
    )
    output = []
    for key, group in sorted(grouped.items()):
        row: dict[str, object] = {
            name: value
            for name, value in zip(
                ("workload", "input_tokens", "output_tokens", "concurrency"), key
            )
        }
        row["repeats"] = len(group)
        row["errors"] = sum(int(item["errors"]) for item in group)
        for field in fields:
            values = [float(item[field]) for item in group if item[field] != ""]
            row[f"{field}_mean"] = statistics.mean(values) if values else ""
            row[f"{field}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0 if values else ""
            )
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
    repeats = repeat_rows(args.results_dir)
    if not repeats:
        raise ValueError(f"no text matrix results found in {args.results_dir}")
    write_csv(args.results_dir / "repeat_summary.csv", repeats)
    write_csv(args.results_dir / "aggregate_summary.csv", aggregate_rows(repeats))
    print(f"summarized {len(repeats)} repeats in {args.results_dir}")


if __name__ == "__main__":
    main()
