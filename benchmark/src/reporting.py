from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from models import BenchmarkTrace


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_results(
    output_dir: Path,
    traces: list[BenchmarkTrace],
    results: list[dict[str, Any]],
    raw_api_results: Any | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "benchmark_trace.json", [trace.to_dict() for trace in traces])
    if raw_api_results is not None:
        write_json(output_dir / "raw_api_results.json", raw_api_results)

    rows = [
        {**result, **measured_metrics(trace, raw_api_results)}
        for trace, result in zip(traces, results, strict=True)
    ]
    csv_path = output_dir / "benchmark_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.md").write_text(render_summary(rows), encoding="utf-8")


def measured_metrics(
    trace: BenchmarkTrace,
    raw_api_results: Any | None = None,
) -> dict[str, int | float | str]:
    branch_latencies = [
        branch.latency_ms
        for branch in trace.branches
        if branch.latency_ms is not None
    ]
    if not branch_latencies:
        return {}

    common_latency_ms = None
    branch_phase_latency_ms = None
    total_latency_ms = None
    if isinstance(raw_api_results, dict):
        common = raw_api_results.get("common")
        if isinstance(common, dict):
            common_latency_ms = common.get("latency_ms")
        branch_phase_latency_ms = raw_api_results.get("branch_phase_latency_ms")
        total_latency_ms = raw_api_results.get("total_latency_ms")

    branch_max_latency_ms = max(branch_latencies)
    branch_output_tokens = sum(branch.decode_tokens for branch in trace.branches)
    measured_branch_phase_ms = _float_or_none(branch_phase_latency_ms)
    branch_phase_ms = measured_branch_phase_ms or branch_max_latency_ms
    branch_phase_source = (
        "measured_branch_phase"
        if measured_branch_phase_ms is not None
        else "max_branch_latency"
    )
    common_ms = _float_or_none(common_latency_ms)
    measured_total_ms = _float_or_none(total_latency_ms)
    if measured_total_ms is not None:
        case_wall_time_ms = measured_total_ms
        case_wall_time_source = "measured_total"
    elif common_ms is not None:
        case_wall_time_ms = common_ms + branch_phase_ms
        case_wall_time_source = "common_plus_branch_wall"
    else:
        case_wall_time_ms = branch_phase_ms
        case_wall_time_source = "branch_wall_only"
    input_tokens = [
        branch.input_tokens
        for branch in trace.branches
        if branch.input_tokens is not None
    ]

    metrics: dict[str, int | float | str] = {
        "branch_mean_latency_ms": statistics.fmean(branch_latencies),
        "branch_median_latency_ms": statistics.median(branch_latencies),
        "branch_min_latency_ms": min(branch_latencies),
        "branch_max_latency_ms": branch_max_latency_ms,
        "branch_phase_wall_ms": branch_phase_ms,
        "branch_phase_wall_source": branch_phase_source,
        "case_wall_time_ms": case_wall_time_ms,
        "case_wall_time_source": case_wall_time_source,
        "approx_wall_time_ms": case_wall_time_ms,
        "branch_total_output_tokens": branch_output_tokens,
        "branch_output_tokens_per_s": _tokens_per_second(
            branch_output_tokens, branch_phase_ms
        ),
        "end_to_end_output_tokens_per_s": _tokens_per_second(
            branch_output_tokens, case_wall_time_ms
        ),
    }
    if common_ms is not None:
        metrics["common_latency_ms"] = common_ms
    if input_tokens:
        metrics["branch_mean_input_tokens"] = statistics.fmean(input_tokens)
        metrics["branch_min_input_tokens"] = min(input_tokens)
        metrics["branch_max_input_tokens"] = max(input_tokens)
    return metrics


def render_summary(results: Iterable[dict[str, Any]]) -> str:
    rows = list(results)
    lines = ["# Agentrix Benchmark Summary", ""]
    if any("branch_mean_latency_ms" in row for row in rows):
        lines.extend(
            [
                "## Measured End-to-End Latency",
                "",
                "| Case | Prefix | Branches | Common ms | Branch mean ms "
                "| Branch p50 ms | Branch max ms | Branch wall ms "
                "| E2E wall ms | Branch tok/s | E2E tok/s |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            if "branch_mean_latency_ms" not in row:
                continue
            lines.append(
                f"| {row['case_id']} | {row['prefix_tokens']} "
                f"| {row['branch_count']} | {_format_number(row, 'common_latency_ms')} "
                f"| {_format_number(row, 'branch_mean_latency_ms')} "
                f"| {_format_number(row, 'branch_median_latency_ms')} "
                f"| {_format_number(row, 'branch_max_latency_ms')} "
                f"| {_format_number(row, 'branch_phase_wall_ms')} "
                f"| {_format_number(row, 'case_wall_time_ms')} "
                f"| {_format_number(row, 'branch_output_tokens_per_s')} "
                f"| {_format_number(row, 'end_to_end_output_tokens_per_s')} |"
            )
        lines.extend(
            [
                "",
                "> Branch wall is the measured concurrent branch phase when present; "
                "older results fall back to max branch latency. E2E wall is measured "
                "total case latency when present; older results fall back to common "
                "latency plus branch wall.",
                "",
            ]
        )

    lines.extend(
        [
            "## Logical Savings",
            "",
            "| Case | Prefix | Branches | Suffix mean | KV reduction "
            "| Tile reduction | Launch reduction |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['prefix_tokens']} | {row['branch_count']} "
            f"| {_float(row, 'suffix_mean'):.1f} "
            f"| {_float(row, 'kv_reduction'):.2f}x "
            f"| {_float(row, 'tile_reduction'):.2f}x "
            f"| {_float(row, 'launch_reduction'):.2f}x |"
        )
    lines.extend(
        [
            "",
            "> Logical savings are length-model estimates. Use the measured latency "
            "table above for end-to-end performance claims.",
            "",
        ]
    )
    return "\n".join(lines)


def _tokens_per_second(tokens: int, latency_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    return tokens / (latency_ms / 1000)


def _float(row: dict[str, Any], key: str) -> float:
    value = row[key]
    if isinstance(value, str):
        return float(value)
    return float(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _format_number(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value in (None, ""):
        return "-"
    return f"{float(value):.1f}"
