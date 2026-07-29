#!/usr/bin/env python3
"""Plot actual GPU/CPU KV placement and transfer events over Agent time."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT_LABEL = "Shared prefix"
PRESSURE_LABEL = "Cold pressure"
BRANCH_LABELS = tuple(f"Branch {index}" for index in range(8))
STACK_LABELS = (ROOT_LABEL, *BRANCH_LABELS, PRESSURE_LABEL)
COLORS = {
    ROOT_LABEL: "#f4b942",
    "Branch 0": "#4e79a7",
    "Branch 1": "#59a14f",
    "Branch 2": "#e15759",
    "Branch 3": "#b07aa1",
    "Branch 4": "#f28e2b",
    "Branch 5": "#76b7b2",
    "Branch 6": "#9c755f",
    "Branch 7": "#ff9da7",
    PRESSURE_LABEL: "#94a3b8",
}
EVENT_ROWS = {
    "cpu_load_complete": 0,
    "gpu_evict": 1,
    "cpu_store_complete": 2,
}
EVENT_LABELS = {
    "cpu_load_complete": "CPU→GPU Load",
    "gpu_evict": "GPU eviction",
    "cpu_store_complete": "GPU→CPU Store",
}
EVENT_MARKERS = {
    "cpu_load_complete": "^",
    "gpu_evict": "x",
    "cpu_store_complete": "v",
}


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def category_label(key: dict[str, Any]) -> str:
    if key["category"] == "shared_prefix":
        return ROOT_LABEL
    if key["category"] == "branch":
        return f"Branch {key['branch']}"
    return PRESSURE_LABEL


def apply_memory_event(
    event: dict[str, Any],
    gpu_blocks: dict[int, dict[str, Any]],
    cpu_keys: dict[str, dict[str, Any]],
) -> None:
    if event["event"] == "gpu_cache":
        for key in event["keys"]:
            gpu_blocks[int(key["block_id"])] = key
    elif event["event"] == "gpu_evict":
        for key in event["keys"]:
            gpu_blocks.pop(int(key["block_id"]), None)
    elif event["event"] == "cpu_store_complete":
        for key in event["keys"]:
            cpu_keys[str(key["key"])] = key
    elif event["event"] == "cpu_evict":
        for key in event["keys"]:
            cpu_keys.pop(str(key["key"]), None)


def state_counts(state: dict[Any, dict[str, Any]]) -> dict[str, int]:
    counts = Counter(category_label(key) for key in state.values())
    return {label: counts[label] for label in STACK_LABELS}


def memory_series(
    payload: dict[str, Any],
) -> tuple[list[float], dict[str, list[int]], dict[str, list[int]]]:
    gpu_blocks: dict[int, dict[str, Any]] = {}
    cpu_keys: dict[str, dict[str, Any]] = {}
    times = [0.0]
    gpu_values = {label: [0] for label in STACK_LABELS}
    cpu_values = {label: [0] for label in STACK_LABELS}

    for event in payload["memory_events"]:
        apply_memory_event(event, gpu_blocks, cpu_keys)
        times.append(float(event["elapsed_ms"]) / 1000)
        gpu_counts = state_counts(gpu_blocks)
        cpu_counts = state_counts(cpu_keys)
        for label in STACK_LABELS:
            gpu_values[label].append(gpu_counts[label])
            cpu_values[label].append(cpu_counts[label])

    end_time = float(payload["events"][-1]["elapsed_ms"]) / 1000
    times.append(end_time)
    for label in STACK_LABELS:
        gpu_values[label].append(gpu_values[label][-1])
        cpu_values[label].append(cpu_values[label][-1])
    return times, gpu_values, cpu_values


def event_time(payload: dict[str, Any], name: str) -> float:
    return next(
        float(event["elapsed_ms"]) / 1000
        for event in payload["events"]
        if event["event"] == name
    )


def annotate_agent_events(axes: list[Any], payload: dict[str, Any]) -> None:
    events = (
        ("root_hot_checkpoint", "root HOT"),
        ("fanout_dropped", "8→2 branches"),
        ("pressure_started", "pressure 1"),
        ("wave2_started", "re-fork"),
        (
            "all_shared_branches_finished",
            "root idle / eviction pressure",
        ),
        ("post_finish_pressure_completed", "pressure done"),
        ("revisit_started", "8-branch revisit"),
    )
    top_axis = axes[0]
    for index, (event_name, label) in enumerate(events):
        when = event_time(payload, event_name)
        for axis in axes:
            axis.axvline(when, color="#334155", linewidth=0.65, alpha=0.23)
        if event_name == "fanout_dropped":
            continue
        top_axis.annotate(
            label,
            xy=(when, 1),
            xycoords=("data", "axes fraction"),
            xytext=(when, 1.08 + 0.085 * (index % 2)),
            textcoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=8,
            arrowprops={
                "arrowstyle": "-",
                "color": "#64748b",
                "linewidth": 0.65,
            },
        )


def plot_event_raster(axis: Any, payload: dict[str, Any]) -> None:
    grouped: dict[tuple[int, str, str], int] = defaultdict(int)
    for event in payload["memory_events"]:
        event_name = event["event"]
        if event_name not in EVENT_ROWS:
            continue
        time_bin = round(float(event["elapsed_ms"]) / 20)  # 20 ms bins
        for key in event["keys"]:
            grouped[(time_bin, event_name, category_label(key))] += 1

    for (time_bin, event_name, label), count in grouped.items():
        axis.scatter(
            time_bin * 0.02,
            EVENT_ROWS[event_name],
            marker=EVENT_MARKERS[event_name],
            color=COLORS[label],
            s=18 + 8 * math.sqrt(count),
            linewidths=1.2,
            alpha=0.82 if label != PRESSURE_LABEL else 0.42,
        )
    axis.set_yticks(
        list(EVENT_ROWS.values()),
        [EVENT_LABELS[name] for name in EVENT_ROWS],
    )
    axis.set_ylim(-0.55, 2.55)
    axis.grid(axis="x", alpha=0.12)
    axis.tick_params(axis="x", labelbottom=False)


def plot_query_aggregation(axis: Any, payload: dict[str, Any]) -> None:
    events = payload.get("operator_events", ())
    times = [float(event["elapsed_ms"]) / 1000 for event in events]
    sizes = [int(event["max_aggregated_queries"]) for event in events]
    axis.vlines(times, 0, sizes, color="#7c3aed", linewidth=0.8, alpha=0.36)
    axis.scatter(times, sizes, color="#6d28d9", s=12, alpha=0.82)
    target = int(payload["metadata"]["branches"])
    axis.axhline(
        target,
        color="#6d28d9",
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
    )
    axis.set_ylim(-0.2, target + 0.8)
    axis.set_yticks(range(0, target + 1, 2))
    axis.set_ylabel("Queries in one\nshared Fork CTA")
    axis.grid(axis="y", alpha=0.18)
    axis.tick_params(axis="x", labelbottom=False)


def plot_timeline(
    payload: dict[str, Any],
    *,
    output: Path,
    optimized: bool,
    end_time: float,
    cpu_limit: float,
) -> None:
    times, gpu_values, cpu_values = memory_series(payload)
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={"height_ratios": (1.25, 3.1, 0.8, 3.1)},
        constrained_layout=True,
    )
    query_axis, gpu_axis, event_axis, cpu_axis = axes

    colors = [COLORS[label] for label in STACK_LABELS]
    gpu_axis.stackplot(
        times,
        *[gpu_values[label] for label in STACK_LABELS],
        colors=colors,
        labels=STACK_LABELS,
        step="post",
        alpha=0.9,
    )
    cpu_axis.stackplot(
        times,
        *[cpu_values[label] for label in STACK_LABELS],
        colors=colors,
        labels=STACK_LABELS,
        step="post",
        alpha=0.78,
    )

    pressure_start = event_time(payload, "post_finish_pressure_started")
    pressure_end = event_time(payload, "post_finish_pressure_completed")
    revisit_start = event_time(payload, "revisit_started")
    revisit_end = event_time(payload, "revisit_completed")
    for axis in axes:
        axis.axvspan(
            pressure_start,
            pressure_end,
            color="#fecaca",
            alpha=0.18,
        )
        axis.axvspan(
            revisit_start,
            revisit_end,
            color="#bbf7d0",
            alpha=0.24,
        )

    plot_query_aggregation(query_axis, payload)
    plot_event_raster(event_axis, payload)
    annotate_agent_events(list(axes), payload)

    gpu_axis.axhline(
        payload["metadata"]["num_gpu_blocks"] - 1,
        color="#0f172a",
        linestyle="--",
        linewidth=0.8,
        alpha=0.55,
    )
    gpu_axis.text(
        end_time,
        payload["metadata"]["num_gpu_blocks"] - 5,
        "GPU cache capacity: 511 usable blocks",
        ha="right",
        va="top",
        fontsize=8,
        color="#334155",
    )
    gpu_axis.set_ylim(0, payload["metadata"]["num_gpu_blocks"] * 1.05)
    cpu_axis.set_ylim(0, cpu_limit)
    gpu_axis.set_ylabel("GPU cached full KV blocks")
    cpu_axis.set_ylabel("CPU offload-resident KV blocks")
    cpu_axis.set_xlabel("Agent workflow timeline (seconds)")
    gpu_axis.grid(axis="y", alpha=0.18)
    cpu_axis.grid(axis="y", alpha=0.18)
    cpu_axis.set_xlim(0, end_time)

    mode = (
        "Lifecycle-aware eviction + hot-prefix query join"
        if optimized
        else "Baseline GPU LRU + FCFS admission"
    )
    summary = payload["summary"]
    revisit_operator = summary["revisit_operator"]
    figure.suptitle(
        (
            f"{mode}: actual KV placement and operator cohorts\n"
            f"CPU→GPU reload = {summary['load_bytes'] / 1024**2:.1f} MiB, "
            f"{summary['load_operations']} operations; revisit max cohort = "
            f"{revisit_operator['max_aggregated_queries']} queries"
        ),
        fontsize=15,
        fontweight="bold",
    )
    legend_handles = [
        Patch(facecolor=COLORS[label], label=label) for label in STACK_LABELS
    ]
    legend_handles.extend(
        [
            Line2D(
                [],
                [],
                color="#ef4444",
                linewidth=8,
                alpha=0.18,
                label="Eviction-pressure phase",
            ),
            Line2D(
                [],
                [],
                color="#22c55e",
                linewidth=8,
                alpha=0.24,
                label="Revisit phase",
            ),
        ]
    )
    figure.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=6,
        fontsize=8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def max_cpu_residency(payload: dict[str, Any]) -> int:
    _, _, cpu_values = memory_series(payload)
    return max(
        sum(cpu_values[label][index] for label in STACK_LABELS)
        for index in range(len(next(iter(cpu_values.values()))))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--optimized-output", type=Path, required=True)
    args = parser.parse_args()

    baseline = read(args.baseline)
    optimized = read(args.optimized)
    for payload in (baseline, optimized):
        if not payload.get("memory_events"):
            raise ValueError("input does not contain memory_events")

    end_time = max(
        float(payload["events"][-1]["elapsed_ms"]) / 1000
        for payload in (baseline, optimized)
    )
    cpu_limit = (
        max(max_cpu_residency(payload) for payload in (baseline, optimized)) * 1.08
    )
    plot_timeline(
        baseline,
        output=args.baseline_output,
        optimized=False,
        end_time=end_time,
        cpu_limit=cpu_limit,
    )
    plot_timeline(
        optimized,
        output=args.optimized_output,
        optimized=True,
        end_time=end_time,
        cpu_limit=cpu_limit,
    )
    print(f"wrote {args.baseline_output}")
    print(f"wrote {args.optimized_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
