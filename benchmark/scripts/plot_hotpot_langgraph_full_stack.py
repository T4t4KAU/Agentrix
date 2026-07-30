#!/usr/bin/env python3
"""Summarize and plot the Qwen3-32B live HotpotQA full-stack experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

STAGES = (
    ("planner", "Planner", "#4c78a8"),
    ("tool_select", "Branch tool selection", "#f58518"),
    ("tool", "Tool execution/wait", "#eeca3b"),
    ("branch_reflect", "Branch reflection", "#54a24b"),
    ("reduce", "Reducer", "#b279a2"),
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def load_run(path: Path, arm: str, repeat: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    offset_ms = payload["metadata"]["engine_startup_s"] * 1000
    samples = [
        {**sample, "aligned_ms": sample["time_ms"] - offset_ms}
        for sample in payload["periodic_kv_samples"]
    ]
    auc = sum(
        (left["usage"] + right["usage"])
        / 2
        * (right["aligned_ms"] - left["aligned_ms"])
        / 1000
        for left, right in pairwise(samples)
    )
    llm = [event for event in payload["events"] if event["kind"] == "llm"]
    resumes = [
        event["ttft_ms"] for event in llm if event["stage"] == "branch_reflect"
    ]
    gpu_peak = max(
        gpu["used_mib"]
        for sample in payload["gpu_samples"]
        for gpu in sample["gpus"]
    )
    connector = payload["kv"]["connector_counters"]
    trimmer = payload["trimmer_stats"] or {}
    compaction = payload["metadata"]["prompt_compaction_report"]
    fork = payload["kv"]["fork_execution"]
    return {
        "arm": arm,
        "repeat": repeat,
        "path": path,
        "payload": payload,
        "samples": samples,
        "wall_s": payload["metadata"]["wall_s"],
        "kv_auc_usage_s": auc,
        "time_averaged_kv_usage": auc / payload["metadata"]["wall_s"],
        "peak_kv_usage": payload["kv"]["peak_usage_fraction"],
        "mean_kv_usage": payload["kv"]["mean_usage_fraction"],
        "peak_gpu_mib": gpu_peak,
        "mean_ttft_ms": statistics.fmean(event["ttft_ms"] for event in llm),
        "p95_ttft_ms": percentile([event["ttft_ms"] for event in llm], 0.95),
        "mean_resume_ttft_ms": statistics.fmean(resumes),
        "p95_resume_ttft_ms": percentile(resumes, 0.95),
        "answer_f1": payload["evaluation"]["f1"],
        "supporting_fact_f1": payload["evaluation"]["sp_f1"],
        "joint_f1": payload["evaluation"]["joint_f1"],
        "tool_call_valid_rate": payload["online_tool_call_valid_rate"],
        "fork_observed_steps": fork["observed_steps"],
        "fork_active_steps": fork["active_steps"],
        "fork_shared_ctas": fork["shared_ctas"],
        "fork_singleton_ctas": fork["singleton_ctas"],
        "trimmed_sessions": trimmer.get("trimmed_sessions", 0),
        "released_block_references": trimmer.get(
            "released_block_references", 0
        ),
        "ttl_fallbacks": trimmer.get("ttl_fallbacks", 0),
        "compaction_saved_chars": compaction["saved_chars"],
        "offload_store_bytes": connector.get("vllm:kv_offload_store_bytes", 0),
        "offload_load_bytes": connector.get("vllm:kv_offload_load_bytes", 0),
    }


def write_csv(runs: list[dict[str, Any]], output: Path) -> None:
    fields = [
        key
        for key in runs[0]
        if key not in {"path", "payload", "samples"}
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow({key: run[key] for key in fields})


def activity_counts(
    events: Iterable[dict[str, Any]], grid_s: np.ndarray
) -> dict[str, np.ndarray]:
    counts = {stage: np.zeros_like(grid_s) for stage, _, _ in STAGES}
    for event in events:
        if event.get("kind") not in {"llm", "tool"}:
            continue
        stage = event.get("stage")
        if event["kind"] == "tool":
            stage = "tool"
        if stage not in counts:
            continue
        start = event["started_ms"] / 1000
        end = start + event["latency_ms"] / 1000
        counts[stage] += (grid_s >= start) & (grid_s <= end)
    return counts


def plot_run(
    activity_axis: Any,
    kv_axis: Any,
    run: dict[str, Any],
    title: str,
) -> None:
    payload = run["payload"]
    duration = payload["metadata"]["wall_s"]
    grid = np.linspace(0, duration, max(400, int(duration * 10)))
    counts = activity_counts(payload["events"], grid)
    activity_axis.stackplot(
        grid,
        *(counts[stage] for stage, _, _ in STAGES),
        labels=[label for _, label, _ in STAGES],
        colors=[color for _, _, color in STAGES],
        alpha=0.85,
    )
    activity_axis.set_title(title, loc="left", fontweight="bold")
    activity_axis.set_ylabel("Outstanding\nAgent operations")
    activity_axis.set_xlim(0, duration)
    activity_axis.grid(axis="y", alpha=0.2)

    times = [sample["aligned_ms"] / 1000 for sample in run["samples"]]
    usages = [sample["usage"] * 100 for sample in run["samples"]]
    kv_axis.plot(times, usages, color="#2f4b7c", linewidth=1.8, label="GPU KV usage")
    trim_times = [
        event["started_ms"] / 1000
        for event in payload["events"]
        if event["kind"] == "kv_trim" and event.get("trimmed")
    ]
    if trim_times:
        kv_axis.vlines(
            trim_times,
            0,
            100,
            color="#d62728",
            alpha=0.18,
            linewidth=0.8,
            label="Successful predicted-TTL trim",
        )
    kv_axis.set_xlabel("Live LangGraph workflow time (s)")
    kv_axis.set_ylabel("GPU KV usage (%)")
    kv_axis.set_xlim(0, duration)
    kv_axis.set_ylim(0, 104)
    kv_axis.grid(alpha=0.25)
    kv_axis.text(
        0.01,
        0.96,
        (
            f"wall={run['wall_s']:.1f}s  "
            f"KV AUC={run['kv_auc_usage_s']:.1f} usage·s  "
            f"resume TTFT={run['mean_resume_ttft_ms']:.0f}ms"
        ),
        transform=kv_axis.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )


def plot(baseline: dict[str, Any], full: dict[str, Any], output: Path) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15, 7.5),
        sharex="col",
        gridspec_kw={"height_ratios": (1, 1.25)},
        constrained_layout=True,
    )
    plot_run(
        axes[0, 0],
        axes[1, 0],
        baseline,
        "Baseline — FlashAttention, no compaction/TTL/offload",
    )
    plot_run(
        axes[0, 1],
        axes[1, 1],
        full,
        "Agentrix full — ForkAttention + compaction + predicted TTL + offload",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=5)
    kv_handles, kv_labels = axes[1, 1].get_legend_handles_labels()
    figure.legend(kv_handles, kv_labels, loc="outside lower center", ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--representative-repeat", type=int, default=2)
    args = parser.parse_args()
    runs = [
        load_run(args.input_dir / f"{arm}_r{repeat}.json", arm, repeat)
        for arm in ("baseline", "full")
        for repeat in range(1, 4)
    ]
    write_csv(runs, args.csv)
    baseline = next(
        run
        for run in runs
        if run["arm"] == "baseline"
        and run["repeat"] == args.representative_repeat
    )
    full = next(
        run
        for run in runs
        if run["arm"] == "full"
        and run["repeat"] == args.representative_repeat
    )
    plot(baseline, full, args.figure)
    for arm in ("baseline", "full"):
        arm_runs = [run for run in runs if run["arm"] == arm]
        print(
            arm,
            {
                key: (
                    statistics.fmean(run[key] for run in arm_runs),
                    statistics.stdev(run[key] for run in arm_runs),
                )
                for key in (
                    "wall_s",
                    "kv_auc_usage_s",
                    "mean_resume_ttft_ms",
                    "p95_resume_ttft_ms",
                    "answer_f1",
                    "supporting_fact_f1",
                    "joint_f1",
                )
            },
        )


if __name__ == "__main__":
    main()
