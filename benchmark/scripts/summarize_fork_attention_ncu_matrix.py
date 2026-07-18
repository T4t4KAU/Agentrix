#!/usr/bin/env python3
"""Combine ForkAttention NCU matrix cell summaries into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


CELL_RE = re.compile(r"p(?P<prefix>\d+)_b(?P<branches>\d+)$")
PLAN_RE = re.compile(r"shared_chunks=(?P<chunks>\d+) max_splits=(?P<splits>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def reduction(control: float, candidate: float) -> float:
    return 100 * (1 - candidate / control)


def main() -> None:
    args = parse_args()
    rows = []
    for cell in args.root.glob("p*_b*"):
        match = CELL_RE.fullmatch(cell.name)
        if match is None:
            continue
        flash_path = cell / "flash_attn_summary.json"
        fork_path = cell / "fork_attn_summary.json"
        if not flash_path.exists() or not fork_path.exists():
            continue
        flash = json.loads(flash_path.read_text())
        fork = json.loads(fork_path.read_text())
        plan_log = (cell / "fork_attn_dram.log").read_text()
        plan_match = PLAN_RE.search(plan_log)
        if plan_match is None:
            raise RuntimeError(f"missing Fork plan in {cell}")
        rows.append(
            {
                "prefix_tokens": int(match.group("prefix")),
                "queries": int(match.group("branches")),
                "shared_chunks": int(plan_match.group("chunks")),
                "max_splits": int(plan_match.group("splits")),
                "flash_kernel_us": flash["kernel_time_ns"] / 1000,
                "fork_kernel_us": fork["kernel_time_ns"] / 1000,
                "kernel_speedup": flash["kernel_time_ns"] / fork["kernel_time_ns"],
                "flash_dram_bytes": flash["dram_read_bytes"],
                "fork_dram_bytes": fork["dram_read_bytes"],
                "flash_dram_sectors": flash["dram_read_sectors"],
                "fork_dram_sectors": fork["dram_read_sectors"],
                "dram_reduction_percent": reduction(
                    flash["dram_read_bytes"], fork["dram_read_bytes"]
                ),
                "flash_dram_bytes_per_token": flash["dram_read_bytes_per_output_token"],
                "fork_dram_bytes_per_token": fork["dram_read_bytes_per_output_token"],
                "flash_l2_read_bytes": flash["l2_read_bytes"],
                "fork_l2_read_bytes": fork["l2_read_bytes"],
                "flash_l2_read_sectors": flash["l2_read_sectors"],
                "fork_l2_read_sectors": fork["l2_read_sectors"],
                "flash_l2_hit_sectors": flash["l2_read_hit_sectors"],
                "fork_l2_hit_sectors": fork["l2_read_hit_sectors"],
                "flash_l2_miss_sectors": flash["l2_read_miss_sectors"],
                "fork_l2_miss_sectors": fork["l2_read_miss_sectors"],
                "l2_reduction_percent": reduction(
                    flash["l2_read_bytes"], fork["l2_read_bytes"]
                ),
                "flash_l2_hit_percent": flash["l2_read_hit_rate_percent"],
                "fork_l2_hit_percent": fork["l2_read_hit_rate_percent"],
                "flash_tensor_pipe_warp_instructions": flash[
                    "tensor_pipe_warp_instructions"
                ],
                "fork_tensor_pipe_warp_instructions": fork[
                    "tensor_pipe_warp_instructions"
                ],
                "flash_fma_pipe_warp_instructions": flash["fma_pipe_warp_instructions"],
                "fork_fma_pipe_warp_instructions": fork["fma_pipe_warp_instructions"],
                "tensor_instruction_reduction_percent": reduction(
                    flash["tensor_pipe_warp_instructions"],
                    fork["tensor_pipe_warp_instructions"],
                ),
                "fma_instruction_reduction_percent": reduction(
                    flash["fma_pipe_warp_instructions"],
                    fork["fma_pipe_warp_instructions"],
                ),
                "flash_kernel_launches": flash["kernel_launches"],
                "fork_kernel_launches": fork["kernel_launches"],
                "flash_kernel_specializations": flash["kernel_specializations"],
                "fork_kernel_specializations": fork["kernel_specializations"],
            }
        )
    rows.sort(key=lambda row: (row["prefix_tokens"], row["queries"]))
    if not rows:
        raise RuntimeError(f"no complete matrix cells under {args.root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} matrix rows to {args.output}")


if __name__ == "__main__":
    main()
