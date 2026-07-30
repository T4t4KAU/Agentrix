#!/usr/bin/env python3
"""Summarize and plot a matched live HotpotQA OpenAI-API backend pair."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

STAGES = (
    ("planner", "Planner", "#4c78a8"),
    ("tool_select", "Branch tool selection", "#f58518"),
    ("tool", "Tool execution", "#eeca3b"),
    ("branch_reflect", "Branch reflection", "#54a24b"),
    ("reduce", "Reducer", "#b279a2"),
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def prometheus_value(path: Path, name: str, source: str | None = None) -> float:
    text = path.read_text(encoding="utf-8")
    pattern = rf"^{re.escape(name)}(?:\{{([^}}]*)\}})?\s+(\S+)$"
    total = 0.0
    for match in re.finditer(pattern, text, re.MULTILINE):
        labels = match.group(1) or ""
        if source is None or f'source="{source}"' in labels:
            total += float(match.group(2))
    return total


def prometheus_delta(
    root: Path,
    arm: str,
    name: str,
    source: str | None = None,
) -> float:
    after = root / arm / "metrics.prom"
    before = root / arm / "metrics_before.prom"
    if not after.exists():
        return 0.0
    return prometheus_value(after, name, source) - (
        prometheus_value(before, name, source) if before.exists() else 0.0
    )


def load_memory(
    root: Path,
    arm: str,
    wall_s: float,
) -> tuple[list[tuple[float, float, float, float, float]], float]:
    samples: list[tuple[float, float, float, float, float]] = []
    with (root / arm / "memory_samples.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            try:
                samples.append(
                    (
                        float(row["unix_s"]),
                        float(row["vllm:kv_cache_usage_perc"]),
                        float(row["gpu_used_mib"]),
                        float(row["vllm:num_requests_running"] or 0),
                        float(row["vllm:num_requests_waiting"] or 0),
                    )
                )
            except (TypeError, ValueError):
                continue
    first_timestamp = samples[0][0]
    aligned = [
        (timestamp - first_timestamp, kv, gpu, running, waiting)
        for timestamp, kv, gpu, running, waiting in samples
        if timestamp - first_timestamp <= wall_s
    ]
    if aligned[-1][0] < wall_s:
        _, kv, gpu, running, waiting = aligned[-1]
        aligned.append((wall_s, kv, gpu, running, waiting))
    auc = sum(
        (left[1] + right[1]) / 2 * (right[0] - left[0])
        for left, right in pairwise(aligned)
    )
    return aligned, auc


def summarize(root: Path, arm: str, title: str) -> tuple[dict[str, Any], Any]:
    payload = json.loads((root / arm / "run.json").read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    wall_s = metadata["wall_ms"] / 1000
    llm_events = [event for event in payload["events"] if event["kind"] == "llm"]
    memory, kv_auc = load_memory(root, arm, wall_s)
    row: dict[str, Any] = {
        "arm": arm,
        "title": title,
        "wall_s": wall_s,
        "cases": len(payload["outputs"]),
        "branches": sum(
            output.get("branch_count", 0) for output in payload["outputs"]
        ),
        "events": len(payload["events"]),
        "prompt_compaction": metadata["prompt_compaction"],
        "prompt_tokens": sum(
            event["usage"].get("prompt_tokens", 0) or 0 for event in llm_events
        ),
        "completion_tokens": sum(
            event["usage"].get("completion_tokens", 0) or 0 for event in llm_events
        ),
        "response_chars": sum(
            len(event.get("response", {}).get("content") or "")
            for event in llm_events
        ),
        "compaction_saved_chars": metadata["prompt_compaction_report"][
            "saved_chars"
        ],
        "kv_auc_usage_s": kv_auc,
        "time_avg_kv_usage": kv_auc / wall_s,
        "peak_kv_usage": max(sample[1] for sample in memory),
        "peak_gpu_mib": max(sample[2] for sample in memory),
        "max_running": max(sample[3] for sample in memory),
        "max_waiting": max(sample[4] for sample in memory),
        "cached_prompt_tokens": prometheus_delta(
            root, arm, "vllm:prompt_tokens_cached_total"
        ),
        "computed_prompt_tokens": prometheus_delta(
            root,
            arm,
            "vllm:prompt_tokens_by_source_total",
            "local_compute",
        ),
        "fork_observed_steps": prometheus_delta(
            root, arm, "vllm:fork_attention_observed_steps_total"
        ),
        "fork_active_steps": prometheus_delta(
            root, arm, "vllm:fork_attention_active_steps_total"
        ),
        "fork_shared_ctas": prometheus_delta(
            root, arm, "vllm:fork_attention_shared_ctas_total"
        ),
        "fork_singleton_ctas": prometheus_delta(
            root, arm, "vllm:fork_attention_singleton_ctas_total"
        ),
    }
    row["total_tokens"] = row["prompt_tokens"] + row["completion_tokens"]
    row["total_tokens_per_s"] = row["total_tokens"] / wall_s
    for stage in ("planner", "tool_select", "branch_reflect", "reduce"):
        events = [event for event in llm_events if event["stage"] == stage]
        latencies = [event["latency_ms"] for event in events]
        row[f"{stage}_count"] = len(events)
        row[f"{stage}_mean_latency_ms"] = statistics.fmean(latencies)
        row[f"{stage}_p95_latency_ms"] = percentile(latencies, 0.95)
        row[f"{stage}_prompt_tokens"] = sum(
            event["usage"].get("prompt_tokens", 0) or 0 for event in events
        )
        row[f"{stage}_completion_tokens"] = sum(
            event["usage"].get("completion_tokens", 0) or 0 for event in events
        )
    return row, (payload, memory)


def activity_counts(events: list[dict[str, Any]], grid: np.ndarray) -> dict[str, Any]:
    counts = {stage: np.zeros_like(grid) for stage, _, _ in STAGES}
    for event in events:
        if event.get("kind") not in {"llm", "tool"}:
            continue
        stage = "tool" if event["kind"] == "tool" else event.get("stage")
        if stage not in counts:
            continue
        start = event["started_ms"] / 1000
        end = start + event["latency_ms"] / 1000
        counts[stage] += (grid >= start) & (grid <= end)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    arms = (
        ("baseline", "FlashAttention baseline — compaction off"),
        ("forkattention", "ForkAttention — compaction on"),
    )
    summaries = []
    plot_data = []
    for arm, title in arms:
        summary, data = summarize(args.root, arm, title)
        summaries.append(summary)
        plot_data.append(data)

    fields = [key for key in summaries[0] if key != "title"]
    with (args.root / "systems_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in summaries)

    baseline, fork = summaries
    comparison = {
        "speedup": baseline["wall_s"] / fork["wall_s"],
        "wall_reduction_fraction": 1 - fork["wall_s"] / baseline["wall_s"],
        "total_token_throughput_speedup": (
            fork["total_tokens_per_s"] / baseline["total_tokens_per_s"]
        ),
        "prompt_token_reduction_fraction": (
            1 - fork["prompt_tokens"] / baseline["prompt_tokens"]
        ),
        "completion_token_reduction_fraction": (
            1 - fork["completion_tokens"] / baseline["completion_tokens"]
        ),
        "kv_auc_reduction_fraction": (
            1 - fork["kv_auc_usage_s"] / baseline["kv_auc_usage_s"]
        ),
        "peak_kv_reduction_fraction": (
            1 - fork["peak_kv_usage"] / baseline["peak_kv_usage"]
        ),
        "tool_select_latency_reduction_fraction": (
            1
            - fork["tool_select_mean_latency_ms"]
            / baseline["tool_select_mean_latency_ms"]
        ),
        "reflect_latency_reduction_fraction": (
            1
            - fork["branch_reflect_mean_latency_ms"]
            / baseline["branch_reflect_mean_latency_ms"]
        ),
    }
    (args.root / "systems_summary.json").write_text(
        json.dumps({"runs": summaries, "comparison": comparison}, indent=2),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(16, 8),
        sharex="col",
        gridspec_kw={"height_ratios": (1, 1.15)},
        constrained_layout=True,
    )
    for column, ((payload, memory), summary) in enumerate(
        zip(plot_data, summaries)
    ):
        wall_s = summary["wall_s"]
        grid = np.linspace(0, wall_s, max(800, int(wall_s * 2)))
        counts = activity_counts(payload["events"], grid)
        axes[0, column].stackplot(
            grid,
            *(counts[stage] for stage, _, _ in STAGES),
            labels=[label for _, label, _ in STAGES],
            colors=[color for _, _, color in STAGES],
            alpha=0.85,
        )
        axes[0, column].set_title(summary["title"], loc="left", fontweight="bold")
        axes[0, column].set_ylabel("Outstanding Agent operations")
        axes[0, column].grid(axis="y", alpha=0.2)
        axes[1, column].plot(
            [sample[0] for sample in memory],
            [100 * sample[1] for sample in memory],
            color="#2f4b7c",
            linewidth=1.5,
        )
        axes[1, column].set_xlim(0, wall_s)
        axes[1, column].set_ylim(0, 100)
        axes[1, column].set_xlabel("Live OpenAI-API LangGraph workflow time (s)")
        axes[1, column].set_ylabel("GPU KV usage (%)")
        axes[1, column].grid(alpha=0.25)
        axes[1, column].text(
            0.01,
            0.96,
            (
                f"wall={wall_s:.1f}s  "
                f"KV AUC={summary['kv_auc_usage_s']:.1f} usage·s  "
                f"tok/s={summary['total_tokens_per_s']:,.0f}"
            ),
            transform=axes[1, column].transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=5)
    figure.savefig(args.root / "agent_kv_timeline.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"runs": summaries, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
