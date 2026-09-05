#!/usr/bin/env python3
"""Summarize every completed trial without selecting the fastest repeat."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    trials = []
    groups = []
    for arm in sorted(args.root.iterdir()):
        if not arm.is_dir():
            continue
        per_case = {}
        for path in sorted(arm.glob("*/summary.json")):
            row = json.loads(path.read_text())
            row["arm"] = arm.name
            row["preemptions"] = sum(
                value
                for key, value in row["counter_deltas"].items()
                if "preemptions_total" in key
            )
            row["local_compute_tokens"] = sum(
                value
                for key, value in row["counter_deltas"].items()
                if 'source="local_compute"' in key
            )
            requests = json.loads((path.parent / "requests.json").read_text())
            followups = [r for r in requests if r["stage"] == "turn" and r["turn"] > 0]
            row["followup_ttft_p50_ms"] = (
                statistics.median(r["ttft_ms"] for r in followups)
                if followups
                else None
            )
            row["cached_token_fraction"] = row["cached_tokens"] / row["prompt_tokens"]
            sample_path = path.parent / "metrics_samples.json"
            samples = (
                json.loads(sample_path.read_text()) if sample_path.exists() else []
            )
            for label, metric in (
                ("peak_running", "num_requests_running"),
                ("peak_waiting", "num_requests_waiting"),
            ):
                row[label] = max(
                    (
                        sum(v for k, v in sample["values"].items() if metric in k)
                        for sample in samples
                    ),
                    default=None,
                )
            trials.append(row)
            per_case.setdefault(row["case"], []).append(row)
        for case, rows in per_case.items():
            record = {"arm": arm.name, "case": case, "repeats": len(rows)}
            for key in (
                "wall_s",
                "branch_wall_s",
                "output_tps",
                "branch_output_tps",
                "ttft_p50_ms",
                "ttft_p95_ms",
                "tpot_p50_ms",
                "cached_token_fraction",
                "preemptions",
                "local_compute_tokens",
                "followup_ttft_p50_ms",
                "peak_running",
                "peak_waiting",
            ):
                values = [r[key] for r in rows if r[key] is not None]
                record[key] = statistics.median(values) if values else None
            record["wall_min_s"] = min(r["wall_s"] for r in rows)
            record["wall_max_s"] = max(r["wall_s"] for r in rows)
            groups.append(record)
    (args.root / "all_trials.json").write_text(json.dumps(trials, indent=2))
    (args.root / "aggregate.json").write_text(json.dumps(groups, indent=2))
    if groups:
        with (args.root / "aggregate.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(groups[0]))
            writer.writeheader()
            writer.writerows(groups)
    lines = [
        "# Agent scenario measurements",
        "",
        "All completed trials; median values. Screening and profiled runs are explicitly separate arms.",
        "",
        "| Arm | Case | N | Wall median (range), s | Output tok/s | Preemptions |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in groups:
        lines.append(
            f"| {row['arm']} | {row['case']} | {row['repeats']} | "
            f"{row['wall_s']:.2f} ({row['wall_min_s']:.2f}–{row['wall_max_s']:.2f}) | "
            f"{row['output_tps']:.2f} | {row['preemptions']:.0f} |"
        )
    (args.root / "all_measurements.md").write_text("\n".join(lines) + "\n")
    print(f"Summarized {len(trials)} trials across {len(groups)} arm/case groups")


if __name__ == "__main__":
    main()
