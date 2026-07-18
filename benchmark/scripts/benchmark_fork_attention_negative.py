#!/usr/bin/env python3
"""Locate small-workload regimes where forced ForkAttention loses to Flash."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

from fork_attention_operator_ncu import make_flash, make_fork, make_inputs


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefixes", default="16,32,64,128,256,512,1024,2048")
    parser.add_argument("--queries", default="1,2,4,8,16,32")
    parser.add_argument("--suffixes", default="128")
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def elapsed_us(run, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def benchmark_cell(
    prefix: int,
    queries: int,
    suffix: int,
    warmups: int,
    iterations: int,
    repeats: int,
) -> dict[str, float | int]:
    operator_args = SimpleNamespace(
        prefix_tokens=prefix,
        private_suffix_tokens=suffix,
        prefix_chunk_tokens=0,
        branches=queries,
        block_size=16,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
    )
    q, k_cache, v_cache, block_table, seq_lens, boxes = make_inputs(operator_args)
    flash, flash_output = make_flash(q, k_cache, v_cache, block_table, seq_lens)
    fork, fork_output = make_fork(q, k_cache, v_cache, boxes, 16)

    flash()
    fork()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        fork_output, flash_output.view_as(q), atol=2e-2, rtol=2e-2
    )
    for _ in range(warmups):
        flash()
        fork()
    torch.cuda.synchronize()

    flash_samples = []
    fork_samples = []
    for repeat in range(repeats):
        # Alternate order so temperature or clock drift cannot systematically
        # favor either backend.
        if repeat % 2:
            fork_samples.append(elapsed_us(fork, iterations))
            flash_samples.append(elapsed_us(flash, iterations))
        else:
            flash_samples.append(elapsed_us(flash, iterations))
            fork_samples.append(elapsed_us(fork, iterations))

    flash_median = statistics.median(flash_samples)
    fork_median = statistics.median(fork_samples)
    return {
        "prefix_tokens": prefix,
        "queries": queries,
        "private_suffix_tokens": suffix,
        "flash_median_us": flash_median,
        "fork_median_us": fork_median,
        "speedup": flash_median / fork_median,
        "fork_overhead_percent": 100 * (fork_median / flash_median - 1),
        "flash_range_percent": 100
        * (max(flash_samples) - min(flash_samples))
        / flash_median,
        "fork_range_percent": 100
        * (max(fork_samples) - min(fork_samples))
        / fork_median,
        "shared_chunks": sum(len(item[0]) > 1 for item in boxes),
        "max_splits": max(item[2] for item in boxes) + 1,
        "iterations": iterations,
        "repeats": repeats,
    }


def main() -> None:
    args = parse_args()
    prefixes = parse_ints(args.prefixes)
    queries = parse_ints(args.queries)
    suffixes = parse_ints(args.suffixes)
    rows = []
    for suffix in suffixes:
        for prefix in prefixes:
            for query_count in queries:
                row = benchmark_cell(
                    prefix,
                    query_count,
                    suffix,
                    args.warmups,
                    args.iterations,
                    args.repeats,
                )
                rows.append(row)
                print(
                    f"prefix={prefix:5d} q={query_count:2d} suffix={suffix:4d} "
                    f"Flash={row['flash_median_us']:.2f} us "
                    f"Fork={row['fork_median_us']:.2f} us "
                    f"speedup={row['speedup']:.3f}x",
                    flush=True,
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
