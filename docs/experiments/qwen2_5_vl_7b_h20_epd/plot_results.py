from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
CONCURRENCY_COLORS = {1: "#4472C4", 4: "#70AD47", 8: "#ED7D31", 16: "#C00000"}


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def number(row: dict[str, str], field: str) -> float:
    return float(row[field])


def save(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_input_sensitivity() -> None:
    encoder = [
        row
        for row in read_csv("encoder_only_aggregate.csv")
        if int(row["concurrency"]) == 1
    ]
    prefill = [
        row
        for row in read_csv("text_aggregate.csv")
        if row["workload"] == "prefill" and int(row["concurrency"]) == 1
    ]
    decode = [
        row
        for row in read_csv("text_aggregate.csv")
        if row["workload"] == "decode"
        and int(row["concurrency"]) == 1
        and int(row["output_tokens"]) == 256
    ]
    series = [
        (
            "Encoder (visual tokens)",
            sorted(
                (
                    number(row, "visual_tokens_per_request"),
                    number(row, "client_mean_ms_mean"),
                )
                for row in encoder
            ),
        ),
        (
            "Prefill (input tokens)",
            sorted(
                (number(row, "input_tokens"), number(row, "client_mean_ms_mean"))
                for row in prefill
            ),
        ),
        (
            "Decode context (256 output tokens)",
            sorted(
                (number(row, "input_tokens"), number(row, "client_mean_ms_mean"))
                for row in decode
            ),
        ),
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for label, values in series:
        base_x, base_y = values[0]
        axis.plot(
            [x / base_x for x, _ in values],
            [y / base_y for _, y in values],
            marker="o",
            linewidth=2,
            label=label,
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xlabel("Normalized input length (shortest case = 1x)")
    axis.set_ylabel("Normalized mean latency (shortest case = 1x)")
    axis.set_title("Input-length sensitivity at concurrency 1")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    save(figure, "input_length_sensitivity.png")


def plot_encoder_scaling() -> None:
    rows = read_csv("encoder_only_aggregate.csv")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for concurrency in (1, 4, 16):
        selected = sorted(
            (
                number(row, "visual_tokens_per_request"),
                number(row, "visual_tokens_per_second_mean"),
                number(row, "client_mean_ms_mean"),
            )
            for row in rows
            if int(row["concurrency"]) == concurrency
        )
        color = CONCURRENCY_COLORS[concurrency]
        axes[0].plot(
            [item[0] for item in selected],
            [item[1] for item in selected],
            marker="o",
            color=color,
            label=f"c={concurrency}",
        )
        axes[1].plot(
            [item[0] for item in selected],
            [item[2] for item in selected],
            marker="o",
            color=color,
            label=f"c={concurrency}",
        )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("Visual tokens per request")
        axis.legend()
    axes[0].set_ylabel("Visual tokens/s")
    axes[0].set_title("Isolated Encoder throughput")
    axes[1].set_yscale("log", base=2)
    axes[1].set_ylabel("Mean latency (ms)")
    axes[1].set_title("Isolated Encoder latency")
    save(figure, "encoder_concurrency_scaling.png")


def plot_encoder_resources() -> None:
    rows = read_csv("encoder_only_aggregate.csv")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fields = [
        ("gpu0_util_mean_mean", "GPU utilization (%)"),
        ("gpu0_power_watts_mean_mean", "Power (W)"),
        ("gpu0_memory_used_mib_mean_mean", "Memory used (MiB)"),
    ]
    for concurrency in (1, 4, 16):
        selected = sorted(
            (
                number(row, "visual_tokens_per_request"),
                row,
            )
            for row in rows
            if int(row["concurrency"]) == concurrency
        )
        for axis, (field, label) in zip(axes, fields):
            axis.plot(
                [item[0] for item in selected],
                [number(item[1], field) for item in selected],
                marker="o",
                color=CONCURRENCY_COLORS[concurrency],
                label=f"c={concurrency}",
            )
            axis.set_xscale("log", base=2)
            axis.set_xlabel("Visual tokens per request")
            axis.set_ylabel(label)
            axis.grid(True, alpha=0.25)
    axes[0].set_title("Encoder GPU load")
    axes[1].set_title("Encoder power")
    axes[2].set_title("Allocator-retained memory")
    axes[0].legend()
    save(figure, "encoder_resource_usage.png")


def plot_temporal_load() -> None:
    rows = sorted(
        read_csv("temporal_aggregate.csv"),
        key=lambda row: number(row, "actual_visual_probability_mean"),
    )
    probability = [100 * number(row, "actual_visual_probability_mean") for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharex=True)
    axes[0, 0].plot(
        probability,
        [number(row, "latency_mean_ms_mean") for row in rows],
        marker="o",
        color="#4472C4",
    )
    axes[0, 0].set_ylabel("Mean latency (ms)")
    axes[0, 0].set_title("End-to-end latency")
    axes[0, 1].plot(
        probability,
        [number(row, "gpu0_util_mean_mean") for row in rows],
        marker="o",
        label="Encoder GPU",
    )
    axes[0, 1].plot(
        probability,
        [number(row, "gpu1_util_mean_mean") for row in rows],
        marker="o",
        label="Prefill/Decode GPU",
    )
    axes[0, 1].set_ylabel("Mean GPU utilization (%)")
    axes[0, 1].set_title("GPU load imbalance")
    axes[0, 1].legend()
    axes[1, 0].plot(
        probability,
        [100 * number(row, "e_idle_pd_busy_ratio_mean") for row in rows],
        marker="o",
        color="#C00000",
    )
    axes[1, 0].set_ylabel("Samples with E idle / PD busy (%)")
    axes[1, 0].set_title("Stage mismatch")
    axes[1, 1].plot(
        probability,
        [number(row, "wall_seconds_mean") for row in rows],
        marker="o",
        color="#70AD47",
    )
    axes[1, 1].set_ylabel("Trace wall time (s)")
    axes[1, 1].set_title("Trace completion time")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("Actual visual request ratio (%)")
    save(figure, "temporal_sparse_load.png")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_input_sensitivity()
    plot_encoder_scaling()
    plot_encoder_resources()
    plot_temporal_load()
    print(f"wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
