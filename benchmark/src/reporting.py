from __future__ import annotations

import csv
import json
import math
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
    write_json(
        output_dir / "benchmark_trace.json", [trace.to_dict() for trace in traces]
    )
    if raw_api_results is not None:
        write_json(output_dir / "raw_api_results.json", raw_api_results)

    rows = []
    for index, (trace, result) in enumerate(zip(traces, results, strict=True)):
        rows.append(
            {
                **result,
                **measured_metrics(trace, _select_raw_result(raw_api_results, index)),
            }
        )
    csv_path = output_dir / "benchmark_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.md").write_text(render_summary(rows), encoding="utf-8")


def _select_raw_result(raw_api_results: Any | None, index: int) -> Any | None:
    if isinstance(raw_api_results, list):
        if index < len(raw_api_results):
            return raw_api_results[index]
        return None
    return raw_api_results


def measured_metrics(
    trace: BenchmarkTrace,
    raw_api_results: Any | None = None,
) -> dict[str, int | float | str]:
    branch_latencies = [
        branch.latency_ms for branch in trace.branches if branch.latency_ms is not None
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
    branch_input_tokens = sum(branch.input_tokens or 0 for branch in trace.branches)
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
        "branch_p50_latency_ms": _percentile(branch_latencies, 50),
        "branch_p95_latency_ms": _percentile(branch_latencies, 95),
        "branch_p99_latency_ms": _percentile(branch_latencies, 99),
        "branch_min_latency_ms": min(branch_latencies),
        "branch_max_latency_ms": branch_max_latency_ms,
        "branch_phase_wall_ms": branch_phase_ms,
        "branch_phase_wall_source": branch_phase_source,
        "case_wall_time_ms": case_wall_time_ms,
        "case_wall_time_source": case_wall_time_source,
        "approx_wall_time_ms": case_wall_time_ms,
        "branch_total_output_tokens": branch_output_tokens,
        "branch_total_input_tokens": branch_input_tokens,
        "branch_requests_per_s": _tokens_per_second(
            len(branch_latencies), branch_phase_ms
        ),
        "branch_input_tokens_per_s": _tokens_per_second(
            branch_input_tokens, branch_phase_ms
        ),
        "branch_output_tokens_per_s": _tokens_per_second(
            branch_output_tokens, branch_phase_ms
        ),
        "branch_total_tokens_per_s": _tokens_per_second(
            branch_input_tokens + branch_output_tokens, branch_phase_ms
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
    ttfts = [branch.ttft_ms for branch in trace.branches if branch.ttft_ms is not None]
    if ttfts:
        metrics.update(_distribution_metrics("ttft_ms", ttfts))
    tpots = [branch.tpot_ms for branch in trace.branches if branch.tpot_ms is not None]
    if tpots:
        metrics.update(_distribution_metrics("tpot_ms", tpots))
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
                "| Branch p50 ms | Branch p95 ms | Branch p99 ms "
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
                f"| {_format_number(row, 'branch_p50_latency_ms')} "
                f"| {_format_number(row, 'branch_p95_latency_ms')} "
                f"| {_format_number(row, 'branch_p99_latency_ms')} "
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

    if any("ttft_ms_mean" in row for row in rows):
        lines.extend(
            [
                "## Streaming Latency",
                "",
                "| Case | TTFT mean ms | TTFT P50 | TTFT P95 | TTFT P99 "
                "| TPOT mean ms | TPOT P50 | TPOT P95 | TPOT P99 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            if "ttft_ms_mean" not in row:
                continue
            lines.append(
                f"| {row['case_id']} "
                f"| {_format_number(row, 'ttft_ms_mean')} "
                f"| {_format_number(row, 'ttft_ms_p50')} "
                f"| {_format_number(row, 'ttft_ms_p95')} "
                f"| {_format_number(row, 'ttft_ms_p99')} "
                f"| {_format_number(row, 'tpot_ms_mean')} "
                f"| {_format_number(row, 'tpot_ms_p50')} "
                f"| {_format_number(row, 'tpot_ms_p95')} "
                f"| {_format_number(row, 'tpot_ms_p99')} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Logical Savings",
            "",
            "| Case | Prefix | Branches | Suffix mean | KV saved tokens "
            "| KV saved GiB | KV saved | KV reduction "
            "| Tile reduction | Launch reduction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['prefix_tokens']} | {row['branch_count']} "
            f"| {_float(row, 'suffix_mean'):.1f} "
            f"| {int(_kv_tokens_saved(row))} "
            f"| {_format_number(row, 'kv_gib_saved', 3)} "
            f"| {_kv_reduction_percent(row):.1f}% "
            f"| {_float(row, 'kv_reduction'):.2f}x "
            f"| {_float(row, 'tile_reduction'):.2f}x "
            f"| {_float(row, 'launch_reduction'):.2f}x |"
        )
    if len(rows) > 1:
        total_baseline = sum(_float(row, "baseline_unique_kv") for row in rows)
        total_saved = sum(_kv_tokens_saved(row) for row in rows)
        total_bytes = sum(float(row.get("kv_bytes_saved", 0)) for row in rows)
        lines.extend(
            [
                "",
                f"**Total KV Cache saved:** {int(total_saved)} tokens, "
                f"{total_bytes / 1024**3:.3f} GiB, "
                f"{100 * total_saved / total_baseline:.1f}%.",
            ]
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


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sample")
    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _distribution_metrics(prefix: str, values: list[float]) -> dict[str, float]:
    return {
        f"{prefix}_mean": statistics.fmean(values),
        f"{prefix}_p50": _percentile(values, 50),
        f"{prefix}_p95": _percentile(values, 95),
        f"{prefix}_p99": _percentile(values, 99),
    }


def _float(row: dict[str, Any], key: str) -> float:
    value = row[key]
    if isinstance(value, str):
        return float(value)
    return float(value)


def _kv_tokens_saved(row: dict[str, Any]) -> float:
    if "kv_tokens_saved" in row:
        return _float(row, "kv_tokens_saved")
    return _float(row, "baseline_unique_kv") - _float(row, "monowire_unique_kv")


def _kv_reduction_percent(row: dict[str, Any]) -> float:
    if "kv_reduction_percent" in row:
        return _float(row, "kv_reduction_percent")
    baseline = _float(row, "baseline_unique_kv")
    return 100 * _kv_tokens_saved(row) / baseline if baseline else 0.0


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _format_number(row: dict[str, Any], key: str, digits: int = 1) -> str:
    value = row.get(key)
    if value in (None, ""):
        return "-"
    return f"{float(value):.{digits}f}"
