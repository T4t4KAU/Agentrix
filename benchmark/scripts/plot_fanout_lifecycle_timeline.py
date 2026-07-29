#!/usr/bin/env python3
"""Plot side-by-side LRU and branch-aware KV lifecycle timelines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

EVENT_LABELS = {
    "wave1_running": "8 branches",
    "fanout_dropped": "fanout→2",
    "cooling_checkpoint": "COOLING",
    "wave2_started": "re-fork",
    "wave2_completed": "wave 2 done",
    "cold_checkpoint": "COLD",
    "pressure_started": "cold pressure",
    "revisit_started": "cold revisit",
}


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def series(rows: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    return (
        [float(row["elapsed_ms"]) / 1000 for row in rows],
        [float(row.get(key, 0)) for row in rows],
    )


def annotate_events(axis: Any, payload: dict[str, Any], *, labels: bool) -> None:
    top = axis.get_ylim()[1]
    for index, event in enumerate(payload["events"]):
        label = EVENT_LABELS.get(event["event"])
        if label is None:
            continue
        when = float(event["elapsed_ms"]) / 1000
        axis.axvline(when, color="#4b5563", alpha=0.22, linewidth=0.8)
        if not labels:
            continue
        axis.text(
            when,
            top * (0.98 - 0.08 * (index % 3)),
            label,
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
            color="#374151",
        )


def branch_series(
    payload: dict[str, Any],
) -> tuple[list[float], list[float]]:
    rows = [event for event in payload["events"] if "active_branches" in event]
    return series(rows, "active_branches")


def windowed_rate(
    rows: list[dict[str, Any]],
    *,
    key: str,
    end_seconds: float,
    window_seconds: float = 0.5,
    step_seconds: float = 0.1,
) -> tuple[list[float], list[float]]:
    row_times = [float(row["elapsed_ms"]) / 1000 for row in rows]
    row_values = [float(row.get(key, 0)) for row in rows]
    elapsed = []
    rates = []
    cursor = 0.0
    left = 0
    right = 0
    running_value = 0.0
    while cursor <= end_seconds + step_seconds:
        while right < len(rows) and row_times[right] <= cursor:
            running_value += row_values[right]
            right += 1
        while left < right and row_times[left] <= cursor - window_seconds:
            running_value -= row_values[left]
            left += 1
        elapsed.append(cursor)
        rates.append(running_value / window_seconds)
        cursor += step_seconds
    return elapsed, rates


def instantaneous_throughput(
    payload: dict[str, Any],
    *,
    window_seconds: float = 0.5,
    step_seconds: float = 0.1,
) -> tuple[list[float], list[float]]:
    samples = payload["token_samples"]
    if not samples:
        return [], []
    end = max(
        float(payload["events"][-1]["elapsed_ms"]) / 1000,
        float(samples[-1]["elapsed_ms"]) / 1000,
    )
    elapsed = []
    throughput = []
    cursor = 0.0
    left = 0
    right = 0
    running_tokens = 0
    token_times = [float(sample["elapsed_ms"]) / 1000 for sample in samples]
    token_counts = [int(sample["tokens"]) for sample in samples]
    while cursor <= end + step_seconds:
        while right < len(samples) and token_times[right] <= cursor:
            running_tokens += token_counts[right]
            right += 1
        while left < right and token_times[left] <= cursor - window_seconds:
            running_tokens -= token_counts[left]
            left += 1
        elapsed.append(cursor)
        throughput.append(running_tokens / window_seconds)
        cursor += step_seconds
    return elapsed, throughput


def event_time(payload: dict[str, Any], name: str) -> float | None:
    for row in payload["events"]:
        if row["event"] == name:
            return float(row["elapsed_ms"]) / 1000
    return None


def plot_panel(
    axes: list[Any],
    payload: dict[str, Any],
    title: str,
) -> None:
    (
        branch_axis,
        state_axis,
        eviction_axis,
        throughput_axis,
        gpu_axis,
    ) = axes
    lifecycle = payload["lifecycle_samples"]

    branch_x, branches = branch_series(payload)
    branch_axis.step(
        branch_x,
        branches,
        where="post",
        color="#0f766e",
        linewidth=1.6,
        label="active agent branches",
    )
    branch_axis.set_title(title)
    branch_axis.set_ylabel("Branches")
    branch_axis.set_ylim(0, max(1, payload["metadata"]["branches"]) + 1)
    branch_axis.grid(alpha=0.2)
    kv_axis = branch_axis.twinx()
    kv_x, usage = series(payload["kv_samples"], "usage")
    kv_axis.plot(
        kv_x,
        [value * 100 for value in usage],
        color="#111827",
        linewidth=1.0,
        alpha=0.75,
        label="logical GPU KV usage",
    )
    kv_axis.set_ylabel("GPU KV (%)")
    kv_axis.set_ylim(0, 105)
    lines = branch_axis.lines + kv_axis.lines
    branch_axis.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper right",
        fontsize=7,
    )
    annotate_events(branch_axis, payload, labels=True)

    x, hot = series(lifecycle, "hot_blocks")
    _, cooling = series(lifecycle, "cooling_blocks")
    _, cold = series(lifecycle, "cold_blocks")
    state_axis.stackplot(
        x,
        cold,
        cooling,
        hot,
        labels=("COLD", "COOLING", "HOT"),
        colors=("#60a5fa", "#fbbf24", "#ef4444"),
        alpha=0.82,
    )
    state_axis.set_ylabel("Lifecycle KV blocks")
    state_axis.legend(loc="upper right", fontsize=7)
    state_axis.grid(alpha=0.2)
    annotate_events(state_axis, payload, labels=False)

    end_seconds = float(payload["events"][-1]["elapsed_ms"]) / 1000
    eviction_x, cold_rate = windowed_rate(
        payload["cpu_events"],
        key="evicted_cold",
        end_seconds=end_seconds,
    )
    _, cooling_rate = windowed_rate(
        payload["cpu_events"],
        key="evicted_cooling",
        end_seconds=end_seconds,
    )
    _, hot_rate = windowed_rate(
        payload["cpu_events"],
        key="evicted_hot",
        end_seconds=end_seconds,
    )
    _, unobserved_rate = windowed_rate(
        payload["cpu_events"],
        key="evicted_unobserved",
        end_seconds=end_seconds,
    )
    eviction_axis.stackplot(
        eviction_x,
        cold_rate,
        cooling_rate,
        hot_rate,
        unobserved_rate,
        labels=("COLD", "COOLING", "HOT", "unobserved"),
        colors=("#60a5fa", "#fbbf24", "#ef4444", "#9ca3af"),
        alpha=0.8,
    )
    eviction_axis.set_ylabel("CPU evictions/s\n(500 ms window)")
    eviction_axis.set_ylim(bottom=0)
    eviction_axis.legend(loc="upper right", fontsize=7, ncol=2)
    eviction_axis.grid(alpha=0.2)
    annotate_events(eviction_axis, payload, labels=False)

    throughput_x, throughput = instantaneous_throughput(payload)
    throughput_axis.plot(
        throughput_x,
        throughput,
        color="#7c3aed",
        linewidth=1.5,
        label="output throughput (500 ms window)",
    )
    for phase, color in (("wave2", "#ede9fe"), ("revisit", "#f3e8ff")):
        started = event_time(payload, f"{phase}_started")
        completed = event_time(payload, f"{phase}_completed")
        if started is None or completed is None:
            continue
        throughput_axis.axvspan(
            started,
            completed,
            color=color,
            alpha=0.45,
            label=f"{phase} generation",
        )
    wave = payload["summary"]["wave2"]
    effective = payload["summary"]["wave2_effective"]
    pressure = payload["summary"]["pressure_interval"]
    revisit = payload["summary"]["revisit"]
    throughput_axis.text(
        0.02,
        0.95,
        (
            f"pressure goodput {pressure['goodput_tokens_per_second']:.1f} tok/s\n"
            f"re-fork goodput {effective['goodput_tokens_per_second']:.1f} tok/s; "
            f"{wave['branch_turns_per_second']:.1f} turns/s\n"
            f"P95 TTFT {wave['ttft_p95_ms']:.1f} ms; "
            f"cold revisit {revisit['branch_turns_per_second']:.1f} turns/s, "
            f"{revisit['ttft_p95_ms']:.1f} ms"
        ),
        transform=throughput_axis.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
    )
    throughput_axis.set_ylabel("Output tokens/s")
    throughput_axis.set_ylim(bottom=0)
    throughput_axis.grid(alpha=0.2)
    throughput_axis.legend(loc="upper right", fontsize=7)
    annotate_events(throughput_axis, payload, labels=False)

    gpu_x, gpu_util = series(payload["gpu_samples"], "gpu_utilization")
    _, memory_util = series(payload["gpu_samples"], "memory_utilization")
    gpu_axis.plot(
        gpu_x,
        gpu_util,
        color="#059669",
        linewidth=1.2,
        label="GPU SM utilization",
    )
    gpu_axis.plot(
        gpu_x,
        memory_util,
        color="#0284c7",
        linewidth=1.0,
        alpha=0.85,
        label="memory-controller utilization",
    )
    gpu_axis.set_ylabel("GPU utilization (%)")
    gpu_axis.set_ylim(0, 105)
    pcie_axis = gpu_axis.twinx()
    _, pcie_rx = series(payload["gpu_samples"], "pcie_rx_mib_s")
    _, pcie_tx = series(payload["gpu_samples"], "pcie_tx_mib_s")
    pcie_axis.plot(
        gpu_x,
        [received + transmitted for received, transmitted in zip(pcie_rx, pcie_tx)],
        color="#f97316",
        linewidth=0.9,
        alpha=0.7,
        label="PCIe RX+TX",
    )
    pcie_axis.set_ylabel("PCIe MiB/s")
    pcie_axis.set_ylim(bottom=0)
    gpu_lines = gpu_axis.lines + pcie_axis.lines
    gpu_axis.legend(
        gpu_lines,
        [line.get_label() for line in gpu_lines],
        loc="upper right",
        fontsize=7,
    )
    gpu_axis.set_xlabel("Timeline (seconds)")
    gpu_axis.grid(alpha=0.2)
    annotate_events(gpu_axis, payload, labels=False)


def flatten_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    return {
        "policy": payload["metadata"]["policy"],
        "peak_kv_usage": summary["peak_kv_usage"],
        "cpu_evicted_blocks": summary["cpu_evicted_blocks"],
        "cpu_evicted_hot": summary["cpu_evicted_hot"],
        "cpu_evicted_cooling": summary["cpu_evicted_cooling"],
        "cpu_evicted_cold": summary["cpu_evicted_cold"],
        "cpu_evicted_unobserved": summary["cpu_evicted_unobserved"],
        "cpu_evicted_unique_blocks": summary.get("cpu_evicted_unique_blocks", ""),
        "cpu_evicted_unique_hot": summary.get("cpu_evicted_unique_hot", ""),
        "cpu_evicted_unique_cooling": summary.get("cpu_evicted_unique_cooling", ""),
        "cpu_evicted_unique_cold": summary.get("cpu_evicted_unique_cold", ""),
        "max_hot_blocks": summary["max_hot_blocks"],
        "max_cooling_blocks": summary["max_cooling_blocks"],
        "max_cold_blocks": summary["max_cold_blocks"],
        "wave2_goodput_tokens_per_second": summary["wave2"][
            "goodput_tokens_per_second"
        ],
        "pressure_goodput_tokens_per_second": summary["pressure_interval"][
            "goodput_tokens_per_second"
        ],
        "wave2_effective_goodput_tokens_per_second": summary["wave2_effective"][
            "goodput_tokens_per_second"
        ],
        "wave2_elapsed_ms": summary["wave2"]["elapsed_ms"],
        "wave2_ttft_p50_ms": summary["wave2"]["ttft_p50_ms"],
        "wave2_ttft_p95_ms": summary["wave2"]["ttft_p95_ms"],
        "wave2_branch_turns_per_second": summary["wave2"]["branch_turns_per_second"],
        "revisit_goodput_tokens_per_second": summary["revisit"][
            "goodput_tokens_per_second"
        ],
        "revisit_ttft_p95_ms": summary["revisit"]["ttft_p95_ms"],
        "revisit_branch_turns_per_second": summary["revisit"][
            "branch_turns_per_second"
        ],
    }


def write_summary(
    path: Path, baseline: dict[str, Any], optimized: dict[str, Any]
) -> None:
    rows = [flatten_summary(baseline), flatten_summary(optimized)]
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = read(args.baseline)
    optimized = read(args.optimized)

    figure, axes = plt.subplots(
        5, 2, figsize=(17, 16), sharex="col", constrained_layout=True
    )
    plot_panel(
        [axes[row, 0] for row in range(5)],
        baseline,
        "Original mode: LRU",
    )
    plot_panel(
        [axes[row, 1] for row in range(5)],
        optimized,
        "Optimized mode: branch-aware cohort_lru",
    )
    figure.suptitle(
        "Branch-aware KV HOT / COOLING / COLD lifecycle and handling",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    write_summary(args.output.with_suffix(".csv"), baseline, optimized)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
