#!/usr/bin/env python3
"""Aggregate attention-kernel counters from an Nsight Compute raw CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


METRICS = (
    "dram__bytes_op_read.sum",
    "dram__sectors_op_read.sum",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_read_lookup_hit.sum",
    "lts__t_sectors_op_read_lookup_miss.sum",
    "smsp__inst_executed_pipe_tensor.sum",
    "smsp__inst_executed_pipe_fma.sum",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--output-tokens", required=True, type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def number(value: str) -> float:
    return float(value) if value else 0.0


def main() -> None:
    args = parse_args()
    totals = Counter[str]()
    launch_counts = []
    kernel_counts = []
    kernel_times_ns = []
    for csv_path in args.csv:
        with csv_path.open(newline="") as source:
            reader = csv.DictReader(source)
            next(reader)  # Nsight emits a units row directly below the header.
            kernels = Counter[str]()
            kernel_time_ns = 0.0
            for row in reader:
                kernel_name = row.get("Kernel Name", "")
                if not kernel_name:
                    continue
                kernels[kernel_name] += 1
                kernel_time_ns += number(row.get("gpu__time_duration.sum", ""))
                for metric in METRICS:
                    totals[metric] += number(row.get(metric, ""))
        launch_counts.append(sum(kernels.values()))
        kernel_counts.append(len(kernels))
        kernel_times_ns.append(kernel_time_ns)

    if len(set(launch_counts)) != 1:
        raise RuntimeError(f"metric runs captured different launches: {launch_counts}")
    if not launch_counts or launch_counts[0] == 0:
        raise RuntimeError("no attention kernels were captured")

    dram_bytes = totals["dram__bytes_op_read.sum"]
    l2_sectors = totals["lts__t_sectors_op_read.sum"]
    l2_hits = totals["lts__t_sectors_op_read_lookup_hit.sum"]
    l2_misses = totals["lts__t_sectors_op_read_lookup_miss.sum"]
    l2_lookups = l2_hits + l2_misses
    result = {
        "attention_backend": args.backend,
        "kernel_launches": launch_counts[0],
        "kernel_specializations": max(kernel_counts),
        "kernel_time_ns": kernel_times_ns[0],
        "output_token_count": args.output_tokens,
        "dram_read_bytes": dram_bytes,
        "dram_read_sectors": totals["dram__sectors_op_read.sum"],
        "dram_read_bytes_per_output_token": dram_bytes / args.output_tokens,
        "l2_read_sectors": l2_sectors,
        "l2_read_bytes": l2_sectors * 32,
        "l2_read_hit_sectors": l2_hits,
        "l2_read_miss_sectors": l2_misses,
        "l2_read_hit_rate_percent": 100 * l2_hits / l2_lookups,
        "tensor_pipe_warp_instructions": totals["smsp__inst_executed_pipe_tensor.sum"],
        "fma_pipe_warp_instructions": totals["smsp__inst_executed_pipe_fma.sum"],
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.write_text(encoded)


if __name__ == "__main__":
    main()
