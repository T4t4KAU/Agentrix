#!/usr/bin/env python3
"""Microbenchmark the BlockPool observer used by KV residency shadow mode."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.kv_residency import KVResidencyIndex


class CallbackCountingObserver:
    """Count timed lifecycle callbacks in a separate untimed trial."""

    def __init__(self, delegate: KVResidencyIndex) -> None:
        self.delegate = delegate
        self.count = 0

    def on_allocated(self, block_id: int, ref_count: int) -> int:
        self.count += 1
        return self.delegate.on_allocated(block_id, ref_count)

    def on_cache_inserted(self, block_id: int) -> None:
        self.count += 1
        self.delegate.on_cache_inserted(block_id)

    def on_cache_hit(self, block_id: int, ref_count: int) -> None:
        self.count += 1
        self.delegate.on_cache_hit(block_id, ref_count)

    def on_pinned(self, block_id: int, ref_count: int) -> None:
        self.count += 1
        self.delegate.on_pinned(block_id, ref_count)

    def on_released(self, block_id: int, ref_count: int) -> None:
        self.count += 1
        self.delegate.on_released(block_id, ref_count)

    def on_unpinned(self, block_id: int, ref_count: int) -> None:
        self.count += 1
        self.delegate.on_unpinned(block_id, ref_count)

    def on_cache_removed(
        self,
        block_id: int,
        num_hashes: int,
        *,
        evicted: bool = False,
    ) -> None:
        self.count += 1
        self.delegate.on_cache_removed(block_id, num_hashes, evicted=evicted)

    def on_step(self) -> None:
        self.delegate.on_step()

    def reset(self) -> None:
        self.delegate.reset()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run_trial(
    *,
    enable_shadow: bool,
    num_blocks: int,
    batch_size: int,
    rounds: int,
    count_callbacks: bool = False,
) -> tuple[int, dict[str, int] | None, int]:
    pool = BlockPool(
        num_gpu_blocks=num_blocks + 1,
        enable_caching=True,
        hash_block_size=16,
    )
    index = None
    if enable_shadow:
        index = KVResidencyIndex(
            pool.num_gpu_blocks,
            null_block_id=pool.null_block.block_id,
        )
        observer = CallbackCountingObserver(index) if count_callbacks else index
        pool.set_observer(observer)

    block_hashes = [
        make_block_hash_with_group_id(
            BlockHash(sequence.to_bytes(32, byteorder="big")),
            0,
        )
        for sequence in range(rounds * batch_size)
    ]

    started_ns = time.perf_counter_ns()
    key_offset = 0
    for _ in range(rounds):
        blocks = pool.get_new_blocks(batch_size)
        for position, block in enumerate(blocks):
            pool._insert_block_hash(
                block_hashes[key_offset + position],
                block,
                num_tokens=16,
            )
        key_offset += batch_size
        pool.free_blocks(reversed(blocks))
        pool.touch(blocks)
        pool.free_blocks(reversed(blocks))
        if index is not None:
            index.on_step()
    elapsed_ns = time.perf_counter_ns() - started_ns

    if index is None:
        return elapsed_ns, None, 0
    index.check_consistency()
    stats = index.snapshot()
    callback_count = observer.count if count_callbacks else 0
    return (
        elapsed_ns,
        {
            "allocations": stats.allocations,
            "cache_hits": stats.cache_hits,
            "cache_insertions": stats.cache_insertions,
            "cache_removals": stats.cache_removals,
            "shared_blocks": stats.shared_blocks,
        },
        callback_count,
    )


def main() -> None:
    args = parse_args()
    if args.blocks <= 0 or args.batch_size <= 0 or args.rounds <= 0:
        raise ValueError("blocks, batch-size, and rounds must be positive")
    if args.batch_size > args.blocks:
        raise ValueError("batch-size cannot exceed blocks")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")

    init_none_hash(sha256)
    # Alternate warm-ups to reduce one-sided CPU frequency and import effects.
    run_trial(
        enable_shadow=False,
        num_blocks=args.blocks,
        batch_size=args.batch_size,
        rounds=max(1, args.rounds // 10),
    )
    run_trial(
        enable_shadow=True,
        num_blocks=args.blocks,
        batch_size=args.batch_size,
        rounds=max(1, args.rounds // 10),
    )

    baseline_samples: list[int] = []
    shadow_samples: list[int] = []
    shadow_stats = None
    for repeat in range(args.repeats):
        modes = (False, True) if repeat % 2 == 0 else (True, False)
        for enable_shadow in modes:
            elapsed_ns, stats, _ = run_trial(
                enable_shadow=enable_shadow,
                num_blocks=args.blocks,
                batch_size=args.batch_size,
                rounds=args.rounds,
            )
            if enable_shadow:
                shadow_samples.append(elapsed_ns)
                shadow_stats = stats
            else:
                baseline_samples.append(elapsed_ns)
    _, counted_stats, observer_callbacks = run_trial(
        enable_shadow=True,
        num_blocks=args.blocks,
        batch_size=args.batch_size,
        rounds=args.rounds,
        count_callbacks=True,
    )
    if counted_stats != shadow_stats:
        raise RuntimeError("callback-counting trial did not reproduce timed workload")
    baseline_ns = statistics.median(baseline_samples)
    shadow_ns = statistics.median(shadow_samples)
    blocks_processed = args.batch_size * args.rounds
    delta_ns = shadow_ns - baseline_ns
    result = {
        "configuration": {
            "blocks": args.blocks,
            "batch_size": args.batch_size,
            "rounds": args.rounds,
            "repeats": args.repeats,
            "blocks_processed_per_trial": blocks_processed,
            "observer_callbacks_per_trial": observer_callbacks,
            "observer_callbacks_per_block": observer_callbacks / blocks_processed,
        },
        "baseline": {
            "median_ms": baseline_ns / 1e6,
            "median_ns_per_block": baseline_ns / blocks_processed,
            "samples_ms": [sample / 1e6 for sample in baseline_samples],
        },
        "shadow": {
            "median_ms": shadow_ns / 1e6,
            "median_ns_per_block": shadow_ns / blocks_processed,
            "samples_ms": [sample / 1e6 for sample in shadow_samples],
            "stats": shadow_stats,
        },
        "overhead": {
            "median_ms": delta_ns / 1e6,
            "ns_per_block": delta_ns / blocks_processed,
            "percent": (delta_ns / baseline_ns) * 100,
            "ns_per_callback": delta_ns / observer_callbacks,
        },
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
