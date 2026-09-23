"""Summarize server-side Ascend profiler CSVs without copying large traces."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def summarize(path):
    workers = []
    for output in sorted(path.glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT")):
        kernels = read_csv(output / "kernel_details.csv")
        intervals = sorted(
            (
                float(row["Start Time(us)"]),
                float(row["Start Time(us)"]) + float(row["Duration(us)"]),
            )
            for row in kernels
        )
        if not intervals:
            raise ValueError(f"No device kernels in {output}")
        busy = 0.0
        start, end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start > end:
                busy += end - start
                start = next_start
            end = max(end, next_end)
        busy += end - start
        span = max(end for _, end in intervals) - intervals[0][0]
        host = defaultdict(lambda: {"calls": 0, "self_us": 0.0})
        for row in read_csv(output / "operator_details.csv"):
            host[row["Name"]]["calls"] += 1
            host[row["Name"]]["self_us"] += float(row["Host Self Duration(us)"])
        ops = read_csv(output / "op_statistic.csv")
        apis = read_csv(output / "api_statistic.csv")
        workers.append(
            {
                "worker": output.parent.name,
                "kernel_count": len(kernels),
                "device_kernel_span_ms": span / 1000,
                "device_busy_union_ms": busy / 1000,
                "device_busy_fraction": busy / span,
                "top_device_ops": sorted(
                    ops, key=lambda r: float(r["Total Time(us)"]), reverse=True
                )[:8],
                "top_host_ops": sorted(
                    host.items(), key=lambda r: r[1]["self_us"], reverse=True
                )[:8],
                "graph_apis": [
                    row
                    for row in apis
                    if any(
                        key in row["API Name"].lower()
                        for key in ("mdl", "model", "graph", "replay")
                    )
                ],
            }
        )
    if not workers:
        raise ValueError("Run torch_npu.profiler.profiler.analyse on the traces first")
    return {
        "kind": "diagnostic_profile_summary",
        "notes": "Profiler adds overhead. Device busy is the union of recorded kernel intervals, not hardware utilization or an official benchmark score.",
        "workers": workers,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.traces)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    for worker in summary["workers"]:
        print(
            worker["worker"],
            "device busy fraction:",
            round(worker["device_busy_fraction"], 4),
        )
