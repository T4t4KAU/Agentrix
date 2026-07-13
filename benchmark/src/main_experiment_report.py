from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from telemetry import summarize_samples


BASELINE_VARIANTS = {
    "flash_no_offload": "flash_no_offload",
    "fork_no_offload": "flash_no_offload",
    "flash_ordinary_offload": "flash_no_offload",
    "fork_ordinary_offload": "flash_ordinary_offload",
    "fork_optimized_offload": "fork_ordinary_offload",
    "flash_dp": "flash_dp",
    "fork_dp": "flash_dp",
    "fork_prefix_aware_dp": "flash_dp",
    "flash_tp": "flash_tp",
    "fork_tp_run1": "flash_tp",
    "fork_tp_run2": "flash_tp",
}


def collect_run(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    backend_name = str(manifest["attention_backend"]).lower()
    backend_root = manifest_path.parent / backend_name
    with (backend_root / "benchmark_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        result_rows = list(csv.DictReader(handle))
    raw_batches = json.loads(
        (backend_root / "raw_api_results.json").read_text(encoding="utf-8")
    )
    profile_path = backend_root / "server_profile.json"
    profile = (
        json.loads(profile_path.read_text(encoding="utf-8"))
        if profile_path.exists()
        else {}
    )

    common_requests = [
        case
        for batch in raw_batches
        for case in batch.get("common", {}).get("cases", [])
    ]
    branches = [branch for batch in raw_batches for branch in batch["branches"]]
    requests = [
        {
            **request,
            "latency_ms": request.get("request_latency_ms"),
        }
        for request in common_requests
    ] + branches
    latencies = [
        float(request["latency_ms"])
        for request in requests
        if request.get("latency_ms") is not None
    ]
    ttfts = [
        float(request["ttft_ms"])
        for request in requests
        if request.get("ttft_ms") is not None
    ]
    tpots = [
        float(request["tpot_ms"])
        for request in requests
        if request.get("tpot_ms") is not None
    ]
    branch_latencies = [float(branch["latency_ms"]) for branch in branches]
    branch_ttfts = [
        float(branch["ttft_ms"])
        for branch in branches
        if branch.get("ttft_ms") is not None
    ]
    branch_tpots = [
        float(branch["tpot_ms"])
        for branch in branches
        if branch.get("tpot_ms") is not None
    ]
    total_case_ms = sum(float(batch["total_latency_ms"]) for batch in raw_batches)
    branch_input_tokens = sum(int(branch["input_tokens"]) for branch in branches)
    branch_output_tokens = sum(int(branch["output_tokens"]) for branch in branches)
    common_input_tokens = sum(
        int(request["input_tokens"]) for request in common_requests
    )
    common_output_tokens = sum(
        int(request["output_tokens"]) for request in common_requests
    )
    total_input_tokens = common_input_tokens + branch_input_tokens
    total_output_tokens = common_output_tokens + branch_output_tokens
    baseline_kv = sum(float(row["baseline_unique_kv"]) for row in result_rows)
    shared_kv = sum(float(row["monowire_unique_kv"]) for row in result_rows)
    is_fork = manifest["attention_backend"] == "FORK_ATTN"
    logical_kv = shared_kv if is_fork else baseline_kv
    telemetry = _telemetry_summary(backend_root, profile)
    peak_kv_percent = _summary_value(telemetry, "gpu_kv_cache_usage_percent", "max")
    peak_cpu_kv_percent = _summary_value(
        telemetry,
        "cpu_kv_cache_occupancy_percent",
        "max",
    )
    capacity_gib = _gpu_kv_capacity_gib(backend_root, result_rows, profile)
    cpu_capacity_gib = (
        float(manifest.get("offload_cpu_gib", 0))
        if manifest.get("offload") != "none"
        else 0.0
    )
    estimated_peak_gpu_kv_gib = (
        capacity_gib * peak_kv_percent / 100 if peak_kv_percent is not None else None
    )
    estimated_peak_cpu_kv_gib = (
        0.0
        if cpu_capacity_gib == 0
        else (
            cpu_capacity_gib * peak_cpu_kv_percent / 100
            if peak_cpu_kv_percent is not None
            else None
        )
    )
    agreement_path = (
        manifest_path.parent / "agreement_vs_flash" / "output_agreement.json"
    )
    agreement = (
        json.loads(agreement_path.read_text(encoding="utf-8"))
        if agreement_path.exists()
        else {}
    )
    repeatability_path = (
        manifest_path.parent / "repeatability_vs_fork_run1" / "output_agreement.json"
    )
    repeatability = (
        json.loads(repeatability_path.read_text(encoding="utf-8"))
        if repeatability_path.exists()
        else {}
    )
    routing = _first_rank_mapping(profile, "fork_dp_prefix_routing")
    routed_requests = int(routing.get("requests", 0))
    affinity_routes = int(routing.get("affinity_routes", 0))
    return {
        **manifest,
        "agentrix_git_commit": manifest.get("agentrix_git_commit"),
        "agentrix_git_dirty": manifest.get("agentrix_git_dirty"),
        "vllm_git_commit": manifest.get("vllm_git_commit"),
        "vllm_git_dirty": manifest.get("vllm_git_dirty"),
        "num_gpu_blocks_override": manifest.get("num_gpu_blocks_override"),
        "use_flashinfer_sampler": manifest.get("use_flashinfer_sampler"),
        "prefix_aware_policy": manifest.get("prefix_aware_policy"),
        "fanout_admission_window": manifest.get("fanout_admission_window"),
        "offload_cpu_gib": manifest.get("offload_cpu_gib"),
        "batches": len(raw_batches),
        "requests": len(requests),
        "common_requests": len(common_requests),
        "branch_requests": len(branches),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "common_input_tokens": common_input_tokens,
        "common_output_tokens": common_output_tokens,
        "branch_input_tokens": branch_input_tokens,
        "branch_output_tokens": branch_output_tokens,
        "request_throughput_per_s": _rate(len(requests), total_case_ms),
        "input_throughput_tokens_per_s": _rate(total_input_tokens, total_case_ms),
        "output_throughput_tokens_per_s": _rate(total_output_tokens, total_case_ms),
        "total_throughput_tokens_per_s": _rate(
            total_input_tokens + total_output_tokens, total_case_ms
        ),
        "branch_output_throughput_tokens_per_s": _rate(
            branch_output_tokens, total_case_ms
        ),
        **_distribution("latency_ms", latencies),
        **_distribution("ttft_ms", ttfts),
        **_distribution("tpot_ms", tpots),
        **_distribution("branch_latency_ms", branch_latencies),
        **_distribution("branch_ttft_ms", branch_ttfts),
        **_distribution("branch_tpot_ms", branch_tpots),
        "logical_kv_read_tokens": logical_kv,
        "logical_kv_read_saved_tokens": baseline_kv - logical_kv,
        "logical_kv_read_reduction_percent": (
            100 * (baseline_kv - logical_kv) / baseline_kv if baseline_kv else 0
        ),
        "gpu_kv_cache_capacity_gib": capacity_gib,
        "peak_gpu_kv_cache_usage_percent": peak_kv_percent,
        "estimated_peak_gpu_kv_gib": estimated_peak_gpu_kv_gib,
        "cpu_kv_cache_capacity_gib": cpu_capacity_gib,
        "peak_cpu_kv_cache_occupancy_percent": peak_cpu_kv_percent,
        "estimated_peak_cpu_kv_gib": estimated_peak_cpu_kv_gib,
        "estimated_peak_total_kv_gib": (
            estimated_peak_gpu_kv_gib + estimated_peak_cpu_kv_gib
            if estimated_peak_gpu_kv_gib is not None
            and estimated_peak_cpu_kv_gib is not None
            else None
        ),
        "mean_gpu_compute_utilization_percent": _summary_value(
            telemetry, "compute_utilization_percent", "mean"
        ),
        "mean_gpu_memory_bandwidth_utilization_percent": _summary_value(
            telemetry, "memory_bandwidth_utilization_percent", "mean"
        ),
        "kv_offload_load_gib": float(profile.get("kv_offload_load_bytes", 0)) / 1024**3,
        "kv_offload_load_operations": int(profile.get("kv_offload_load_operations", 0)),
        "kv_offload_load_average_mib": float(
            profile.get("kv_offload_load_average_mib", 0)
        ),
        "kv_offload_store_gib": float(profile.get("kv_offload_store_bytes", 0))
        / 1024**3,
        "kv_offload_store_operations": int(
            profile.get("kv_offload_store_operations", 0)
        ),
        "kv_offload_store_average_mib": float(
            profile.get("kv_offload_store_average_mib", 0)
        ),
        "output_exact_match_percent": agreement.get("normalized_exact_match_percent"),
        "output_token_f1_percent": agreement.get("mean_token_f1_percent"),
        "output_text_similarity_percent": agreement.get("mean_text_similarity_percent"),
        "repeat_exact_match_percent": repeatability.get(
            "normalized_exact_match_percent"
        ),
        "repeat_token_f1_percent": repeatability.get("mean_token_f1_percent"),
        "repeat_text_similarity_percent": repeatability.get(
            "mean_text_similarity_percent"
        ),
        "dp_routed_requests": routed_requests,
        "dp_affinity_routes": affinity_routes,
        "dp_affinity_route_percent": (
            100 * affinity_routes / routed_requests if routed_requests else None
        ),
        "dp_average_route_us": routing.get("avg_route_us"),
        "dp_rank_routes": json.dumps(routing.get("rank_routes", [])),
    }


def annotate_baseline_comparisons(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["mode"],
            row["model_name"],
            row["dataset"],
            row["prefix_tokens"],
            row["branches"],
        )
        groups.setdefault(key, {})[str(row["variant"])] = row

    for variants in groups.values():
        for variant, row in variants.items():
            baseline_variant = BASELINE_VARIANTS.get(variant)
            baseline = variants.get(baseline_variant or "")
            row["baseline_variant"] = baseline_variant
            row["output_throughput_change_percent"] = _relative_change(
                row.get("output_throughput_tokens_per_s"),
                baseline.get("output_throughput_tokens_per_s") if baseline else None,
            )
            row["peak_gpu_kv_reduction_vs_baseline_percent"] = _reduction(
                row.get("estimated_peak_gpu_kv_gib"),
                baseline.get("estimated_peak_gpu_kv_gib") if baseline else None,
            )
            row["peak_total_kv_reduction_vs_baseline_percent"] = _reduction(
                row.get("estimated_peak_total_kv_gib"),
                baseline.get("estimated_peak_total_kv_gib") if baseline else None,
            )
            row["kv_offload_load_reduction_vs_baseline_percent"] = _reduction(
                row.get("kv_offload_load_gib"),
                baseline.get("kv_offload_load_gib") if baseline else None,
            )


def render_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Agentrix Main Experiment Results",
        "",
        "## Performance and Resource Metrics",
        "",
        "| Model | Dataset | Prefix | Branches | Variant | Output tok/s | "
        "TTFT P50/P95/P99 ms | TPOT P50/P95/P99 ms | "
        "Latency P50/P95/P99 ms | GPU compute | Memory BW | Peak GPU KV GiB (%) | Peak total KV GiB | "
        "Logical KV read reduction |",
        "|---|---|---:|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_name']} | {row['dataset']} | {row['prefix_tokens']} "
            f"| {row['branches']} | {row['variant']} "
            f"| {_format(row['output_throughput_tokens_per_s'])} "
            f"| {_triplet(row, 'ttft_ms')} | {_triplet(row, 'tpot_ms')} "
            f"| {_triplet(row, 'latency_ms')} "
            f"| {_format_percent(row['mean_gpu_compute_utilization_percent'])} "
            f"| {_format_percent(row['mean_gpu_memory_bandwidth_utilization_percent'])} "
            f"| {_peak_kv(row)} "
            f"| {_format(row['estimated_peak_total_kv_gib'])} "
            f"| {_format_percent(row['logical_kv_read_reduction_percent'])} |"
        )

    comparison_rows = [
        row
        for row in rows
        if row.get("baseline_variant")
        and row.get("baseline_variant") != row.get("variant")
    ]
    if comparison_rows:
        lines.extend(
            [
                "",
                "## Baseline Deltas",
                "",
                "| Model | Dataset | Prefix | Branches | Variant | Baseline | "
                "Output throughput change | Peak GPU KV reduction | Peak total KV reduction |",
                "|---|---|---:|---:|---|---|---:|---:|---:|",
            ]
        )
        for row in comparison_rows:
            lines.append(
                f"| {row['model_name']} | {row['dataset']} "
                f"| {row['prefix_tokens']} | {row['branches']} | {row['variant']} "
                f"| {row['baseline_variant']} "
                f"| {_format_percent(row['output_throughput_change_percent'])} "
                f"| {_format_percent(row['peak_gpu_kv_reduction_vs_baseline_percent'])} "
                f"| {_format_percent(row['peak_total_kv_reduction_vs_baseline_percent'])} |"
            )

    offload_rows = [row for row in rows if row.get("offload") != "none"]
    if offload_rows:
        lines.extend(
            [
                "",
                "## Offload Traffic",
                "",
                "| Model | Dataset | Prefix | Branches | Variant | Peak CPU KV GiB (%) | Load GiB | "
                "Load ops (avg MiB) | Store GiB | Store ops (avg MiB) | "
                "Load reduction vs baseline |",
                "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in offload_rows:
            lines.append(
                f"| {row['model_name']} | {row['dataset']} "
                f"| {row['prefix_tokens']} | {row['branches']} | {row['variant']} "
                f"| {_format(row['estimated_peak_cpu_kv_gib'])} "
                f"({_format_percent(row['peak_cpu_kv_cache_occupancy_percent'])}) "
                f"| {_format(row['kv_offload_load_gib'])} "
                f"| {row['kv_offload_load_operations']} "
                f"({_format(row['kv_offload_load_average_mib'])}) "
                f"| {_format(row['kv_offload_store_gib'])} "
                f"| {row['kv_offload_store_operations']} "
                f"({_format(row['kv_offload_store_average_mib'])}) "
                f"| {_format_percent(row['kv_offload_load_reduction_vs_baseline_percent'])} |"
            )

    agreement_rows = [
        row for row in rows if row.get("output_exact_match_percent") is not None
    ]
    if agreement_rows:
        lines.extend(
            [
                "",
                "## Accuracy Guardrail",
                "",
                "| Model | Dataset | Prefix | Branches | Variant | Exact match | "
                "Token F1 | Text similarity | Repeat exact match |",
                "|---|---|---:|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in agreement_rows:
            lines.append(
                f"| {row['model_name']} | {row['dataset']} "
                f"| {row['prefix_tokens']} | {row['branches']} | {row['variant']} "
                f"| {_format_percent(row['output_exact_match_percent'])} "
                f"| {_format_percent(row['output_token_f1_percent'])} "
                f"| {_format_percent(row['output_text_similarity_percent'])} "
                f"| {_format_percent(row['repeat_exact_match_percent'])} |"
            )

    routing_rows = [row for row in rows if row.get("dp_routed_requests")]
    if routing_rows:
        lines.extend(
            [
                "",
                "## Prefix-Aware DP Routing",
                "",
                "| Model | Dataset | Prefix | Branches | Requests | "
                "Affinity routes | Avg route us | Rank routes |",
                "|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in routing_rows:
            lines.append(
                f"| {row['model_name']} | {row['dataset']} "
                f"| {row['prefix_tokens']} | {row['branches']} "
                f"| {row['dp_routed_requests']} "
                f"| {_format_percent(row['dp_affinity_route_percent'])} "
                f"| {_format(row['dp_average_route_us'])} "
                f"| {row['dp_rank_routes']} |"
            )

    provenance_counts: dict[tuple[Any, ...], int] = {}
    for row in rows:
        key = (
            row.get("agentrix_git_commit"),
            row.get("agentrix_git_dirty"),
            row.get("vllm_git_commit"),
            row.get("vllm_git_dirty"),
            row.get("num_gpu_blocks_override"),
            row.get("use_flashinfer_sampler"),
            row.get("prefix_aware_policy"),
            row.get("fanout_admission_window"),
        )
        provenance_counts[key] = provenance_counts.get(key, 0) + 1
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "| Agentrix commit | Dirty | vLLM commit | Dirty | GPU blocks override | FlashInfer sampler | Prefix-aware policy | Admission window | Runs |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for provenance, run_count in sorted(
        provenance_counts.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    ):
        (
            agentrix_commit,
            agentrix_dirty,
            vllm_commit,
            vllm_dirty,
            block_override,
            use_flashinfer_sampler,
            prefix_aware_policy,
            fanout_admission_window,
        ) = provenance
        lines.append(
            f"| {_short_commit(agentrix_commit)} | {_format_bool(agentrix_dirty)} "
            f"| {_short_commit(vllm_commit)} | {_format_bool(vllm_dirty)} "
            f"| {block_override if block_override is not None else '-'} "
            f"| {_format_bool(use_flashinfer_sampler)} "
            f"| {_format_bool(prefix_aware_policy)} "
            f"| {fanout_admission_window if fanout_admission_window is not None else '-'} "
            f"| {run_count} |"
        )

    lines.extend(
        [
            "",
            "## Metric Notes",
            "",
            "- Memory BW is NVIDIA `utilization.memory`, a memory-controller activity proxy rather than measured HBM GB/s.",
            "- Peak GPU KV is sampled from `vllm:kv_cache_usage_perc` during the request phase.",
            "- Logical KV read reduction estimates repeated prefix KV read volume avoided by ForkAttention; it is not physical cache capacity.",
            "- Peak GPU KV reduction uses sampled physical GPU KV occupancy relative to the baseline named in the delta table.",
            "- Throughput and request latency include both common-analysis and branch requests; branch-only distributions remain in the CSV.",
            "- Accuracy is deterministic output agreement against FlashAttention, not environment-level task accuracy.",
            "- Repeat exact match compares Fork TP run 2 directly with Fork TP run 1.",
            "- Experimental KV reload rebalance is disabled for this experiment matrix.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _distribution(prefix: str, values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_p50": None,
            f"{prefix}_p95": None,
            f"{prefix}_p99": None,
        }
    return {
        f"{prefix}_mean": statistics.fmean(values),
        f"{prefix}_p50": _percentile(values, 50),
        f"{prefix}_p95": _percentile(values, 95),
        f"{prefix}_p99": _percentile(values, 99),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _rate(count: int, latency_ms: float) -> float:
    return count / (latency_ms / 1000) if latency_ms > 0 else 0


def _relative_change(value: Any, baseline: Any) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    return 100 * (float(value) - float(baseline)) / float(baseline)


def _reduction(value: Any, baseline: Any) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    return 100 * (float(baseline) - float(value)) / float(baseline)


def _summary_value(
    telemetry: dict[str, Any], metric: str, statistic: str
) -> float | None:
    summary = telemetry.get(metric)
    if not isinstance(summary, dict) or summary.get(statistic) is None:
        return None
    return float(summary[statistic])


def _telemetry_summary(
    backend_root: Path,
    profile: dict[str, Any],
) -> dict[str, Any]:
    telemetry_path = backend_root / "telemetry.json"
    if telemetry_path.exists():
        telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
        samples = telemetry.get("samples")
        if isinstance(samples, list):
            return summarize_samples(samples).get("aggregate", {})
    return profile.get("telemetry", {}).get("aggregate", {})


def _first_rank_mapping(profile: dict[str, Any], key: str) -> dict[str, Any]:
    for rank in profile.get("ranks", []):
        value = rank.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def _gpu_kv_capacity_gib(
    backend_root: Path,
    result_rows: list[dict[str, str]],
    profile: dict[str, Any],
) -> float:
    profile_capacity = float(profile.get("gpu_kv_cache_capacity_gib") or 0)
    if profile_capacity > 0:
        return profile_capacity

    kv_bytes_per_token = 0
    if result_rows:
        kv_bytes_per_token = int(float(result_rows[0].get("kv_bytes_per_token", 0)))
    capacity_tokens = 0
    if kv_bytes_per_token > 0:
        pattern = re.compile(r"GPU KV cache size:\s*([0-9,]+)\s*tokens")
        current_logs = [backend_root / "vllm_server.log"]
        current_logs.extend(sorted(backend_root.glob("vllm_server_rank[0-9]*.log")))
        for log_path in current_logs:
            if not log_path.exists():
                continue
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            capacity_tokens += sum(
                int(value.replace(",", "")) for value in pattern.findall(text)
            )
    if capacity_tokens > 0:
        return capacity_tokens * kv_bytes_per_token / 1024**3
    return 0.0


def _format(value: Any) -> str:
    return "-" if value is None else f"{float(value):.2f}"


def _format_percent(value: Any) -> str:
    return "-" if value is None else f"{float(value):.2f}%"


def _triplet(row: dict[str, Any], prefix: str) -> str:
    return "/".join(
        _format(row.get(f"{prefix}_{percentile}"))
        for percentile in ("p50", "p95", "p99")
    )


def _peak_kv(row: dict[str, Any]) -> str:
    gib = _format(row.get("estimated_peak_gpu_kv_gib"))
    percent = _format_percent(row.get("peak_gpu_kv_cache_usage_percent"))
    return f"{gib} ({percent})"


def _short_commit(value: Any) -> str:
    return str(value)[:12] if value else "-"


def _format_bool(value: Any) -> str:
    if value is None:
        return "-"
    return "yes" if bool(value) else "no"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifests = sorted(args.root.rglob("manifest.json"))
    if not manifests:
        raise SystemExit(f"no experiment manifests found under {args.root}")
    rows = [collect_run(path) for path in manifests]
    rows.sort(
        key=lambda row: (
            row["model_name"],
            row["dataset"],
            row["prefix_tokens"],
            row["branches"],
            row["variant"],
        )
    )
    annotate_baseline_comparisons(rows)
    write_csv(args.output.with_suffix(".csv"), rows)
    args.output.write_text(render_report(rows), encoding="utf-8")
    print(f"Wrote {len(rows)} experiment rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
