#!/usr/bin/env python3
"""E/P interference test with head starts calibrated to CPU preprocessing."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from run_experiment import concurrent_pair, make_victim_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--variant-start", type=int, default=19)
    parser.add_argument(
        "--workloads", nargs="+", default=["small", "medium", "large", "xlarge"]
    )
    args = parser.parse_args()
    head_start_ms = {"small": 10.0, "medium": 30.0, "large": 150.0, "xlarge": 400.0}
    victim = make_victim_prompt(args.model, 4096)
    rows: list[dict[str, object]] = []

    # Text-only warmup and baseline.
    from run_experiment import stream_chat
    stream_chat(8000, "Warm up. Reply with one word.")
    for repetition in range(3):
        ttft, total, usage = stream_chat(8000, victim)
        rows.append({
            "workload": "text", "case": "baseline", "repetition": repetition,
            "ttft_ms": ttft, "total_ms": total,
            "prompt_tokens": usage.get("prompt_tokens"), "render_ms": 0,
            "image_path": "",
        })

    for workload in args.workloads:
        for repetition in range(args.repetitions):
            same_variant = args.variant_start + repetition
            diff_variant = args.variant_start + args.repetitions + repetition
            same_path = next(args.assets.glob(f"{workload}_*_v{same_variant}.jpg"))
            diff_path = next(args.assets.glob(f"{workload}_*_v{diff_variant}.jpg"))

            victim_result, attacker_result = concurrent_pair(
                victim, same_path, attacker_rank=0,
                delay_ms=head_start_ms[workload],
            )
            diff_victim_result, diff_attacker_result = concurrent_pair(
                victim, diff_path, attacker_rank=1,
                delay_ms=head_start_ms[workload],
            )
            for case, result, render_ms, path in (
                ("same_rank_victim", victim_result, 0, same_path),
                ("same_rank_attacker", attacker_result, 0, same_path),
                ("different_rank_victim", diff_victim_result, 0, diff_path),
                ("different_rank_attacker", diff_attacker_result, 0, diff_path),
            ):
                ttft, total, usage = result
                rows.append({
                    "workload": workload, "case": case, "repetition": repetition,
                    "ttft_ms": ttft, "total_ms": total,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "render_ms": render_ms, "image_path": str(path),
                })
            print(
                workload, repetition,
                f"same_victim={victim_result[0]:.1f}ms",
                f"different_victim={diff_victim_result[0]:.1f}ms",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    baseline = statistics.median(
        float(row["ttft_ms"]) for row in rows if row["case"] == "baseline"
    )
    print("baseline_ms", baseline)
    for workload in args.workloads:
        values = [
            float(row["ttft_ms"]) for row in rows
            if row["workload"] == workload and row["case"] == "same_rank_victim"
        ]
        print(workload, "same_rank_slowdown", statistics.median(values) / baseline)


if __name__ == "__main__":
    main()
