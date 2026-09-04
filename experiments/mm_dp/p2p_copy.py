#!/usr/bin/env python3
"""Measure GPU-to-GPU tensor copy time for vision embedding-sized payloads."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--src", type=int, default=0)
    parser.add_argument("--dst", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    print("peer_access", torch.cuda.can_device_access_peer(args.src, args.dst))
    rows = []
    for tokens in args.tokens:
        src = torch.empty((tokens, args.hidden_size), dtype=torch.bfloat16, device=f"cuda:{args.src}")
        dst = torch.empty_like(src, device=f"cuda:{args.dst}")
        for _ in range(5):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize(args.src)
        torch.cuda.synchronize(args.dst)
        samples = []
        for _ in range(args.repetitions):
            torch.cuda.synchronize(args.src)
            torch.cuda.synchronize(args.dst)
            start = time.perf_counter()
            dst.copy_(src, non_blocking=True)
            torch.cuda.synchronize(args.dst)
            samples.append((time.perf_counter() - start) * 1000)
        rows.append({
            "tokens": tokens,
            "hidden_size": args.hidden_size,
            "dtype": "bfloat16",
            "bytes": src.numel() * src.element_size(),
            "median_ms": sorted(samples)[len(samples) // 2],
            "min_ms": min(samples),
        })
        del src, dst
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
