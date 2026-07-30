#!/usr/bin/env python3
"""Plot paired live-Agent tool activity and KV-cache timelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_arm(axis: Any, payload: dict[str, Any], title: str) -> None:
    samples = payload["kv_samples"]
    origin = min((event["time_ms"] for event in payload["events"]), default=0)
    x = [(sample["time_ms"] - origin) / 1000 for sample in samples]
    y = [100 * sample["usage"] for sample in samples]
    axis.plot(x, y, color="#285f8f", linewidth=1.8, label="GPU KV usage")

    events = payload["events"]
    starts: dict[int, dict[str, Any]] = {}
    for item in events:
        if item["kind"] == "tool_start":
            starts[item["round"]] = item
        elif item["kind"] == "tool_end" and item["round"] in starts:
            start = starts[item["round"]]
            left = (start["time_ms"] - origin) / 1000
            right = (item["time_ms"] - origin) / 1000
            axis.axvspan(left, right, color="#edb458", alpha=0.23)
            axis.text(
                (left + right) / 2,
                axis.get_ylim()[1] * 0.82,
                f"{start['tool']} wait",
                ha="center",
                va="top",
                fontsize=8,
            )
    for item in events:
        if item["kind"] == "agent_start":
            axis.axvline(
                (item["time_ms"] - origin) / 1000,
                color="#5b8c5a",
                linewidth=0.8,
                alpha=0.6,
            )
        elif item["kind"] == "trim_result":
            at = (item["time_ms"] - origin) / 1000
            axis.axvline(at, color="#b23a48", linestyle="--", linewidth=1.5)
            axis.scatter([at], [0], color="#b23a48", marker="v", zorder=5)

    axis.set_title(title)
    axis.set_ylabel("KV cache usage (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.set_ylim(bottom=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ttl", type=Path, required=True)
    parser.add_argument("--predicted", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load(args.baseline)
    ttl = load(args.ttl)
    predicted = load(args.predicted) if args.predicted else None
    rows = 3 if predicted else 2
    figure, axes = plt.subplots(rows, 1, figsize=(12, 3.15 * rows), sharex=False)
    plot_arm(axes[0], baseline, "Baseline: live LangGraph agent")
    plot_arm(
        axes[1],
        ttl,
        f"Tool-KV TTL: {ttl['ttl_ms']:.0f} ms (red dashed = trim)",
    )
    if predicted is not None:
        predictions = [
            item["ttl_ms"]
            for item in predicted["events"]
            if item["kind"] == "ttl_prediction"
        ]
        label = (
            f"Predicted TTL: {predictions[0]:.0f} ms"
            if predictions
            else "Predicted TTL"
        )
        plot_arm(axes[2], predicted, f"{label} (red dashed = trim)")
    axes[-1].set_xlabel("Experiment time (s)")
    figure.suptitle(
        "Agent workflow aligned with GPU KV-cache lifetime\n"
        f"Qwen3-14B, real search/read tools, {ttl['tool_delay_ms']:.0f} ms tool wait"
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
