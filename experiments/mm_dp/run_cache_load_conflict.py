#!/usr/bin/env python3
"""Sweep hot-rank prefill load against cold-rank vision recomputation."""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_experiment import image_content, make_victim_prompt, stream_chat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--loads", type=int, nargs="+", default=[0, 2, 4, 6])
    parser.add_argument(
        "--workloads", nargs="+", default=["small", "medium", "large", "xlarge"]
    )
    parser.add_argument("--background-head-start-ms", type=float, default=40)
    parser.add_argument("--variant-start", type=int, default=11)
    args = parser.parse_args()
    background_prompt = make_victim_prompt(args.model, 4096)
    rows: list[dict[str, object]] = []

    stream_chat(8000, "Warm up. Reply with one word.")
    stream_chat(8001, "Warm up. Reply with one word.")

    for workload in args.workloads:
        cell = 0
        for load in args.loads:
            for repetition in range(args.repetitions):
                variant = args.variant_start + cell
                path = next(args.assets.glob(f"{workload}_*_v{variant}.jpg"))
                cell += 1

                # Establish processor + encoder locality on rank 0 only.
                stream_chat(8000, image_content(path))

                max_workers = load + 2
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    backgrounds = [
                        pool.submit(stream_chat, 8000, background_prompt)
                        for _ in range(load)
                    ]
                    if load:
                        time.sleep(args.background_head_start_ms / 1000)
                    hot_future = pool.submit(stream_chat, 8000, image_content(path))
                    cold_future = pool.submit(stream_chat, 8001, image_content(path))
                    hot = hot_future.result()
                    cold = cold_future.result()
                    background_results = [future.result() for future in backgrounds]

                for destination, measurement in (("hot_busy_rank0", hot), ("cold_idle_rank1", cold)):
                    ttft, total, usage = measurement
                    rows.append({
                        "workload": workload,
                        "background_load": load,
                        "repetition": repetition,
                        "destination": destination,
                        "rank": 0 if destination == "hot_busy_rank0" else 1,
                        "ttft_ms": ttft,
                        "total_ms": total,
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "image_path": str(path),
                    })
                print(
                    workload, load, repetition,
                    f"hot={hot[0]:.1f}ms", f"cold={cold[0]:.1f}ms",
                    f"delta={hot[0] - cold[0]:.1f}ms",
                    f"bg={[round(x[0], 1) for x in background_results]}",
                    flush=True,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
