#!/usr/bin/env python3
"""Measure bounded scheduler work for proactive LMCache backup planning."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

from lmcache.integration.vllm.proactive_backup import ProactiveBackupScheduler


class FakeResidency:
    """Minimal residency tracker for deterministic planner measurements."""

    def __init__(self, num_blocks: int, mode: str) -> None:
        shared = mode == "shared"
        lower_copy = mode == "backed_up"
        self.status = [[1, True, shared, lower_copy, False] for _ in range(num_blocks)]
        self.occupancy = 1.0

    def backup_metadata(self, block_id: int) -> tuple[int, bool, bool, bool, bool]:
        generation, idle, shared, lower_copy, inflight = self.status[block_id]
        return generation, idle, shared, lower_copy, inflight

    def cache_occupancy(self) -> float:
        return self.occupancy

    def start_backup(
        self,
        block_id: int,
        generation: int,
        tier: object,
        operation_id: int,
    ) -> bool:
        del tier, operation_id
        status = self.status[block_id]
        if status[0] != generation or status[4]:
            return False
        status[4] = True
        return True

    def finish_backup(
        self,
        block_id: int,
        generation: int,
        tier: object,
        operation_id: int,
        *,
        success: bool,
    ) -> bool:
        del tier, operation_id
        status = self.status[block_id]
        if status[0] != generation or not status[4]:
            return False
        status[3] = success
        status[4] = False
        return True

    def drop_backup(
        self,
        block_id: int,
        generation: int,
        tier: object,
    ) -> bool:
        del tier
        status = self.status[block_id]
        if status[0] != generation:
            return False
        status[3] = False
        return True


class FakeBlockPool:
    """Constant-time pinning surface used by the coordinator."""

    def __init__(self, num_blocks: int) -> None:
        self.blocks: Sequence[object] = tuple(range(num_blocks))

    def pin(self, blocks: Sequence[object]) -> None:
        del blocks

    def unpin(self, blocks: Sequence[object]) -> None:
        del blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5000)
    parser.add_argument("--scan-budget", type=int, default=32)
    parser.add_argument("--batch-blocks", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(samples: list[int], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[int((len(ordered) - 1) * fraction)] / 1e3


def sample_mode(args: argparse.Namespace, mode: str) -> dict[str, float]:
    blocks_per_chunk = args.chunk_size // args.block_size
    num_blocks = args.scan_budget * blocks_per_chunk
    chunks = [
        (
            chunk_index,
            tuple(
                range(
                    chunk_index * blocks_per_chunk,
                    (chunk_index + 1) * blocks_per_chunk,
                )
            ),
        )
        for chunk_index in range(args.scan_budget)
    ]
    samples = []
    for _ in range(args.rounds):
        residency = FakeResidency(num_blocks, mode)
        scheduler = ProactiveBackupScheduler(
            FakeBlockPool(num_blocks),
            residency,
            block_size=args.block_size,
            chunk_size=args.chunk_size,
            high_watermark=0.8,
            scan_budget=args.scan_budget,
            batch_blocks=args.batch_blocks,
            max_inflight_blocks=num_blocks,
            worker_count=1,
            cpu_tier="cpu",
        )
        if mode == "low_pressure_registration":
            residency.occupancy = 0.5
            started_ns = time.perf_counter_ns()
            accepted = scheduler.register_request("request", chunks)
            samples.append(time.perf_counter_ns() - started_ns)
            if accepted != 0:
                raise RuntimeError("profiling setup accepted a low-pressure request")
            continue
        accepted = scheduler.register_request("request", chunks)
        if accepted != args.scan_budget:
            raise RuntimeError("profiling setup rejected candidates")
        if mode == "pressure_drop":
            residency.occupancy = 0.5
        started_ns = time.perf_counter_ns()
        scheduler.plan()
        samples.append(time.perf_counter_ns() - started_ns)
    return {
        "p50_us": percentile(samples, 0.50),
        "p95_us": percentile(samples, 0.95),
        "p99_us": percentile(samples, 0.99),
        "max_us": max(samples) / 1e3,
    }


def main() -> None:
    args = parse_args()
    numeric = (
        args.rounds,
        args.scan_budget,
        args.batch_blocks,
        args.block_size,
        args.chunk_size,
    )
    if min(numeric) <= 0:
        raise ValueError("all numeric arguments must be positive")
    if args.chunk_size % args.block_size != 0:
        raise ValueError("chunk-size must be a multiple of block-size")
    if args.batch_blocks < args.chunk_size // args.block_size:
        raise ValueError("batch-blocks must hold at least one chunk")

    result = {
        "configuration": {
            "rounds": args.rounds,
            "scan_budget": args.scan_budget,
            "batch_blocks": args.batch_blocks,
            "block_size": args.block_size,
            "chunk_size": args.chunk_size,
        },
        "latency": {
            mode: sample_mode(args, mode)
            for mode in (
                "ready",
                "shared",
                "backed_up",
                "low_pressure_registration",
                "pressure_drop",
            )
        },
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
