from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


EVENT_PATTERN = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*?"
    r"\[(?P<request_id>epd-[^]]+)] (?P<message>.*)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-dir", type=Path, required=True)
    parser.add_argument("--proxy-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def proxy_events(path: Path) -> dict[str, dict[str, datetime]]:
    events: dict[str, dict[str, datetime]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        match = EVENT_PATTERN.match(line)
        if not match:
            continue
        timestamp = datetime.strptime(match["time"], "%Y-%m-%d %H:%M:%S,%f")
        message = match["message"]
        if "completed with status" in message:
            events[match["request_id"]]["proxy_finish"] = timestamp
        elif message.startswith("POST "):
            events[match["request_id"]]["proxy_start"] = timestamp
        elif message.startswith("All ") and "encoder requests completed" in message:
            events[match["request_id"]]["encoder_done"] = timestamp
        elif message.startswith("Forwarding to decode"):
            events[match["request_id"]]["pd_start"] = timestamp
    return events


def duration_ms(start: datetime | None, finish: datetime | None) -> float | None:
    if start is None or finish is None:
        return None
    return (finish - start).total_seconds() * 1000


def main() -> None:
    args = parse_args()
    events = proxy_events(args.proxy_log)
    rows = []
    for request_path in sorted(args.requests_dir.glob("*_requests.jsonl")):
        name = request_path.name.removesuffix("_requests.jsonl")
        requests = [
            json.loads(line)
            for line in request_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        client_latencies = [float(item["client_latency_ms"]) for item in requests]
        encoder_latencies = []
        downstream_latencies = []
        for item in requests:
            event = events.get(item["request_id"], {})
            encoder = duration_ms(event.get("proxy_start"), event.get("encoder_done"))
            downstream = duration_ms(event.get("pd_start"), event.get("proxy_finish"))
            if encoder is not None:
                encoder_latencies.append(encoder)
            if downstream is not None:
                downstream_latencies.append(downstream)
        rows.append(
            {
                "name": name,
                "requests": len(requests),
                "errors": sum(item["error"] is not None for item in requests),
                "visual_tokens_per_request": requests[0]["expected_visual_tokens"],
                "client_mean_ms": statistics.mean(client_latencies),
                "client_p95_ms": percentile(client_latencies, 0.95),
                "encoder_path_mean_ms": (
                    statistics.mean(encoder_latencies) if encoder_latencies else ""
                ),
                "encoder_path_p95_ms": (
                    percentile(encoder_latencies, 0.95) if encoder_latencies else ""
                ),
                "downstream_mean_ms": (
                    statistics.mean(downstream_latencies) if downstream_latencies else ""
                ),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} configurations to {args.output}")


if __name__ == "__main__":
    main()
