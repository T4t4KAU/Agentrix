"""Summarize engine iteration logs on the server; elapsed time is host-observed."""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ITERATION = re.compile(
    r"EngineCore_DP(\d+).*Iteration\((\d+)\): "
    r"(\d+) context requests, (\d+) context tokens, "
    r"(\d+) generation requests, (\d+) generation tokens, "
    r"iteration elapsed time: ([\d.]+) ms"
)


def profiling_window(benchmark_log):
    matches = re.findall(
        r"PhaseRecordsStats\(phase=CreditPhase\.PROFILING.*?"
        r"\bstart_ns=(\d+), sent_end_ns=(\d+)",
        benchmark_log.read_text(),
    )
    if len(matches) != 1:
        raise ValueError("Expected one completed official profiling phase")
    start, end = (int(value) / 1e9 for value in matches[0])
    if end <= start:
        raise ValueError("Invalid official profiling window")
    return start, end


def summarize(path, window=None):
    samples = defaultdict(lambda: defaultdict(list))
    seen = set()
    for line in path.read_text().splitlines():
        match = ITERATION.search(line)
        if not match:
            continue
        if window is not None:
            stamp = re.search(r"\b(\d{2}-\d{2} \d{2}:\d{2}:\d{2})\b", line)
            if stamp is None:
                raise ValueError("Iteration log lacks its UTC timestamp")
            years = {datetime.fromtimestamp(t, timezone.utc).year for t in window}
            timestamps = [
                datetime.strptime(f"{year}-{stamp[1]}", "%Y-%m-%d %H:%M:%S")
                .replace(tzinfo=timezone.utc)
                .timestamp()
                for year in years
            ]
            if not any(window[0] <= t < window[1] for t in timestamps):
                continue
        rank, iteration, context, context_tokens, decode, decode_tokens, elapsed = (
            match.groups()
        )
        identity = (rank, iteration)
        if identity in seen:
            raise ValueError(f"Repeated engine iteration: {identity}")
        seen.add(identity)
        context, decode = int(context), int(decode)
        kind = (
            "mixed"
            if context and decode
            else "prefill"
            if context
            else "decode"
            if decode
            else "empty"
        )
        samples[rank][kind].append(
            (float(elapsed), context + decode, int(context_tokens), int(decode_tokens))
        )
    if not samples:
        raise ValueError("No iteration details; start with --batch-diagnostics")
    ranks = {}
    for rank, groups in samples.items():
        count = sum(len(rows) for rows in groups.values())
        ranks[rank] = {}
        for kind, rows in groups.items():
            times = sorted(row[0] for row in rows)
            batches = defaultdict(int)
            for _, requests, _, _ in rows:
                batches[requests] += 1
            ranks[rank][kind] = {
                "iterations": len(rows),
                "iteration_fraction": len(rows) / count,
                "host_elapsed_ms_mean": sum(times) / len(times),
                "host_elapsed_ms_p90": times[(9 * len(times) - 1) // 10],
                "context_tokens": sum(row[2] for row in rows),
                "generation_tokens": sum(row[3] for row in rows),
                "request_count_histogram": dict(sorted(batches.items())),
            }
    return {
        "kind": "batch_iteration_diagnostic",
        "profiling_window_utc": (
            [datetime.fromtimestamp(t, timezone.utc).isoformat() for t in window]
            if window is not None
            else None
        ),
        "notes": [
            (
                "Filtered to the official request-sending window; log timestamps have one-second precision."
                if window is not None
                else "Includes all logged iterations, including warmup and any later probes."
            ),
            "Context/generation classification follows vLLM compute_iteration_details.",
            "Elapsed time is the engine's host-observed wait/processing interval; async overlap means it is not device execution time.",
            "Decode phase alone does not prove graph replay; check cudagraph metrics too.",
        ],
        "ranks": ranks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server_log", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmark-log", type=Path)
    args = parser.parse_args()
    window = profiling_window(args.benchmark_log) if args.benchmark_log else None
    result = summarize(args.server_log, window)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["ranks"], indent=2))
