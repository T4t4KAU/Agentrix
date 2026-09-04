#!/usr/bin/env python3
"""Measure bounded KV placement planning and optional queue application."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.kv_placement import KVPlacementPlanner
from vllm.v1.core.kv_residency import KVResidencyIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=8192)
    parser.add_argument("--scan-budget", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--active", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def build_index(num_blocks: int) -> KVResidencyIndex:
    index = KVResidencyIndex(
        num_blocks + 1,
        null_block_id=0,
        warm_seconds=0.0,
        cooling_seconds=0.0,
        aging_budget=num_blocks * 2,
        now_ns=1_000_000_000,
    )
    for block_id in range(1, num_blocks + 1):
        index.on_allocated(block_id, 1)
        index.on_cache_inserted(block_id)
        index.on_released(block_id, 0)
    index.advance(now_ns=1_000_000_000)
    return index


def build_pool(num_blocks: int) -> BlockPool:
    pool = BlockPool(
        num_gpu_blocks=num_blocks + 1,
        enable_caching=True,
        hash_block_size=1,
    )
    for block in pool.blocks[1:]:
        block.set_block_hash(
            make_block_hash_with_group_id(
                BlockHash(str(block.block_id).encode()),
                0,
            ),
            num_tokens=1,
        )
    return pool


def run_trial(
    *,
    enable_planner: bool,
    num_blocks: int,
    scan_budget: int,
    rounds: int,
    active: bool,
) -> tuple[int, int]:
    index = build_index(num_blocks)
    pool = build_pool(num_blocks) if active else None
    planner = KVPlacementPlanner(scan_budget, active=active) if enable_planner else None

    started_ns = time.perf_counter_ns()
    for _ in range(rounds):
        index.advance(now_ns=index.now_ns)
        if planner is not None:
            if pool is None:
                planner.plan(index, requested_blocks=scan_budget)
            else:
                if not planner.plan_and_apply(index, scan_budget, pool):
                    raise RuntimeError("active placement plan could not be applied")
    elapsed_ns = time.perf_counter_ns() - started_ns
    total_scanned = planner.snapshot().scanned_blocks if planner is not None else 0
    return elapsed_ns, total_scanned


def sample_step_latencies(
    *,
    num_blocks: int,
    scan_budget: int,
    rounds: int,
    active: bool,
) -> list[int]:
    index = build_index(num_blocks)
    pool = build_pool(num_blocks) if active else None
    planner = KVPlacementPlanner(scan_budget, active=active)
    samples = []
    for _ in range(rounds):
        started_ns = time.perf_counter_ns()
        index.advance(now_ns=index.now_ns)
        if pool is None:
            planner.plan(index, requested_blocks=scan_budget)
        elif not planner.plan_and_apply(index, scan_budget, pool):
            raise RuntimeError("active placement plan could not be applied")
        samples.append(time.perf_counter_ns() - started_ns)
    return samples


def percentile(samples: list[int], fraction: float) -> int:
    ordered = sorted(samples)
    return ordered[int((len(ordered) - 1) * fraction)]


def main() -> None:
    args = parse_args()
    if min(args.blocks, args.scan_budget, args.rounds, args.repeats) <= 0:
        raise ValueError("all numeric arguments must be positive")
    if args.scan_budget > args.blocks:
        raise ValueError("scan-budget cannot exceed blocks")

    baseline_samples: list[int] = []
    planner_samples: list[int] = []
    total_scanned = 0
    for repeat in range(args.repeats):
        modes = (False, True) if repeat % 2 == 0 else (True, False)
        for enable_planner in modes:
            elapsed_ns, scanned = run_trial(
                enable_planner=enable_planner,
                num_blocks=args.blocks,
                scan_budget=args.scan_budget,
                rounds=args.rounds,
                active=args.active,
            )
            if enable_planner:
                planner_samples.append(elapsed_ns)
                total_scanned = scanned
            else:
                baseline_samples.append(elapsed_ns)

    baseline_ns = statistics.median(baseline_samples)
    planner_ns = statistics.median(planner_samples)
    delta_ns = planner_ns - baseline_ns
    step_samples = sample_step_latencies(
        num_blocks=args.blocks,
        scan_budget=args.scan_budget,
        rounds=args.rounds,
        active=args.active,
    )
    result = {
        "configuration": {
            "blocks": args.blocks,
            "scan_budget": args.scan_budget,
            "rounds": args.rounds,
            "repeats": args.repeats,
            "active": args.active,
        },
        "baseline": {
            "median_ms": baseline_ns / 1e6,
            "median_ns_per_step": baseline_ns / args.rounds,
            "samples_ms": [sample / 1e6 for sample in baseline_samples],
        },
        "planner": {
            "median_ms": planner_ns / 1e6,
            "median_ns_per_step": planner_ns / args.rounds,
            "samples_ms": [sample / 1e6 for sample in planner_samples],
            "scanned_blocks": total_scanned,
        },
        "overhead": {
            "median_ms": delta_ns / 1e6,
            "ns_per_step": delta_ns / args.rounds,
            "ns_per_scanned_block": delta_ns / total_scanned,
        },
        "planner_step_latency_us": {
            "p50": percentile(step_samples, 0.50) / 1e3,
            "p95": percentile(step_samples, 0.95) / 1e3,
            "p99": percentile(step_samples, 0.99) / 1e3,
            "max": max(step_samples) / 1e3,
        },
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
