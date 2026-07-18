#!/usr/bin/env python3
"""Plot the ForkAttention NCU operator matrix from its aggregated CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_csv", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_matrix(path: Path) -> tuple[list[dict[str, float]], list[int], list[int]]:
    with path.open(newline="") as source:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(source)
        ]
    prefixes = sorted({int(row["prefix_tokens"]) for row in rows})
    queries = sorted({int(row["queries"]) for row in rows})
    expected = {(prefix, query) for prefix in prefixes for query in queries}
    observed = {(int(row["prefix_tokens"]), int(row["queries"])) for row in rows}
    if observed != expected:
        raise RuntimeError(
            f"matrix is incomplete: missing={sorted(expected - observed)}"
        )
    return rows, prefixes, queries


def grid(
    rows: list[dict[str, float]], prefixes: list[int], queries: list[int], key: str
) -> np.ndarray:
    indexed = {(int(row["prefix_tokens"]), int(row["queries"])): row for row in rows}
    return np.array(
        [[indexed[(prefix, query)][key] for query in queries] for prefix in prefixes]
    )


def annotate_heatmap(axis: plt.Axes, values: np.ndarray, fmt: str) -> None:
    midpoint = (float(np.nanmin(values)) + float(np.nanmax(values))) / 2
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                format(value, fmt),
                ha="center",
                va="center",
                color="white" if value > midpoint else "black",
                fontsize=8,
            )


def plot_heatmaps(
    rows: list[dict[str, float]], prefixes: list[int], queries: list[int], output: Path
) -> None:
    panels = [
        ("kernel_speedup", "Kernel speedup (Flash / Fork)", "viridis", ".2f", "x"),
        ("dram_reduction_percent", "DRAM read reduction", "RdYlGn", ".1f", "%"),
        ("l2_reduction_percent", "L2 read reduction", "viridis", ".1f", "%"),
        (
            "fma_instruction_reduction_percent",
            "FMA-pipe instruction reduction",
            "viridis",
            ".1f",
            "%",
        ),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for axis, (key, title, cmap, fmt, unit) in zip(axes.flat, panels, strict=True):
        values = grid(rows, prefixes, queries, key)
        limit = (
            max(abs(float(values.min())), abs(float(values.max())))
            if key.startswith("dram_")
            else None
        )
        image = axis.imshow(
            values,
            cmap=cmap,
            aspect="auto",
            vmin=-limit if limit is not None else None,
            vmax=limit if limit is not None else None,
        )
        annotate_heatmap(axis, values, fmt)
        axis.set_title(title)
        axis.set_xticks(range(len(queries)), queries)
        axis.set_yticks(
            range(len(prefixes)), [f"{prefix // 1024}K" for prefix in prefixes]
        )
        axis.set_xlabel("Simultaneous queries")
        axis.set_ylabel("Shared-prefix length")
        figure.colorbar(image, ax=axis, shrink=0.82, label=unit)
    figure.suptitle(
        "ForkAttention pure-operator NCU matrix (25 measured points)", fontsize=14
    )
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_trends(
    rows: list[dict[str, float]], prefixes: list[int], queries: list[int], output: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(prefixes)))
    x = np.arange(len(queries))
    for prefix, color in zip(prefixes, colors, strict=True):
        selected = sorted(
            (row for row in rows if int(row["prefix_tokens"]) == prefix),
            key=lambda row: row["queries"],
        )
        label = f"{prefix // 1024}K"
        axes[0, 0].plot(
            x,
            [row["kernel_speedup"] for row in selected],
            "o-",
            color=color,
            label=label,
        )
        axes[0, 1].plot(
            x,
            [row["dram_reduction_percent"] for row in selected],
            "o-",
            color=color,
            label=label,
        )
        axes[1, 0].plot(
            x,
            [row["l2_reduction_percent"] for row in selected],
            "o-",
            color=color,
            label=label,
        )
        axes[1, 1].plot(
            x,
            [row["fma_instruction_reduction_percent"] for row in selected],
            "o-",
            color=color,
            label=label,
        )

    ideal = [100 * (1 - 1 / query) for query in queries]
    axes[1, 0].plot(x, ideal, "k--", linewidth=1.5, label=r"Ideal $1-1/Q$")
    axes[0, 1].axhline(0, color="black", linewidth=1, linestyle="--")
    titles = [
        "Kernel speedup",
        "DRAM read reduction",
        "L2 read reduction",
        "FMA-pipe instruction reduction",
    ]
    ylabels = ["Flash / Fork (x)", "Reduction (%)", "Reduction (%)", "Reduction (%)"]
    for axis, title, ylabel in zip(axes.flat, titles, ylabels, strict=True):
        axis.set_title(title)
        axis.set_xlabel("Simultaneous queries")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, queries)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(title="Prefix", ncols=2)
    axes[1, 0].legend(title="Prefix", ncols=2)
    figure.suptitle(
        "ForkAttention scaling trends (markers are measured NCU cells)", fontsize=14
    )
    figure.savefig(output, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows, prefixes, queries = load_matrix(args.matrix_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_heatmaps(
        rows,
        prefixes,
        queries,
        args.output_dir / "forkattention_ncu_matrix_heatmaps.png",
    )
    plot_trends(
        rows, prefixes, queries, args.output_dir / "forkattention_ncu_matrix_trends.png"
    )
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
