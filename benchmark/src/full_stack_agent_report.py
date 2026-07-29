"""Report the strict equal-GPU-KV full-stack Agentrix A/B."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from full_stack_agent import load_result, validate_equal_kv_metadata


def _peak_gpu_mib(payload: dict[str, Any]) -> tuple[float, float]:
    per_sample = []
    for sample in payload.get("gpu_memory_samples", []):
        values = [
            float(gpu["used_mib"])
            for gpu in sample.get("gpus", [])
            if isinstance(gpu.get("used_mib"), (int, float))
        ]
        if values:
            per_sample.append((max(values), sum(values)))
    if not per_sample:
        return 0.0, 0.0
    return max(item[0] for item in per_sample), max(item[1] for item in per_sample)


def _connector_bytes(payload: dict[str, Any], operation: str) -> float:
    counters = payload.get("kv", {}).get("connector_counters", {})
    return sum(
        float(value)
        for key, value in counters.items()
        if operation in key and "bytes" in key and isinstance(value, (int, float))
    )


def build_report(
    baseline: dict[str, Any], optimized: dict[str, Any]
) -> dict[str, Any]:
    fairness = validate_equal_kv_metadata(
        baseline["metadata"], optimized["metadata"]
    )
    baseline_by_id = {
        (row["case_id"], row["source_id"]): row
        for row in baseline["results"]
    }
    optimized_by_id = {
        (row["case_id"], row["source_id"]): row
        for row in optimized["results"]
    }
    common = sorted(baseline_by_id.keys() & optimized_by_id.keys())
    if not common:
        raise ValueError("A/B results did not contain paired tasks")
    paired_f1_delta = statistics.fmean(
        optimized_by_id[key]["f1"] - baseline_by_id[key]["f1"]
        for key in common
    )
    baseline_peak, baseline_sum = _peak_gpu_mib(baseline)
    optimized_peak, optimized_sum = _peak_gpu_mib(optimized)
    trimmer = optimized.get("trimmer_stats", {})
    router = optimized.get("router_stats", {})
    offload_store_bytes = _connector_bytes(optimized, "store")
    offload_load_bytes = _connector_bytes(optimized, "load")
    activation = {
        "forkattention": (
            optimized["metadata"].get("attention_backend") == "FORK_ATTN"
            and optimized.get("kv", {})
            .get("fork_execution", {})
            .get("active_steps", 0)
            > 0
        ),
        "router": bool(router.get("active")) and router.get("route_count", 0) > 0,
        "offload": offload_store_bytes > 0 or offload_load_bytes > 0,
        "trimmer": trimmer.get("trimmed_sessions", 0) > 0,
        "compaction": optimized.get("compaction_saved_chars", 0) > 0,
    }
    return {
        "schema_version": 1,
        "fairness": fairness,
        "paired_tasks": len(common),
        "activation": activation,
        "all_five_mechanisms_observed": all(activation.values()),
        "baseline": {
            "wall_seconds": baseline["wall_seconds"],
            "tasks_per_second": baseline["tasks_per_second"],
            "mean_f1": baseline["mean_f1"],
            "success_rate": baseline["success_rate"],
            "valid_citation_rate": baseline["valid_citation_rate"],
            "mean_task_latency_seconds": baseline["mean_task_latency_seconds"],
            "mean_turn_ttft_seconds": baseline["mean_turn_ttft_seconds"],
            "peak_single_gpu_mib": baseline_peak,
            "peak_gpu_sum_mib": baseline_sum,
        },
        "full_stack": {
            "wall_seconds": optimized["wall_seconds"],
            "tasks_per_second": optimized["tasks_per_second"],
            "mean_f1": optimized["mean_f1"],
            "success_rate": optimized["success_rate"],
            "valid_citation_rate": optimized["valid_citation_rate"],
            "mean_task_latency_seconds": optimized["mean_task_latency_seconds"],
            "mean_turn_ttft_seconds": optimized["mean_turn_ttft_seconds"],
            "peak_single_gpu_mib": optimized_peak,
            "peak_gpu_sum_mib": optimized_sum,
            "compaction_saved_chars": optimized.get("compaction_saved_chars", 0),
            "trimmed_sessions": trimmer.get("trimmed_sessions", 0),
            "released_block_references": trimmer.get(
                "released_block_references", 0
            ),
            "offload_store_bytes": offload_store_bytes,
            "offload_load_bytes": offload_load_bytes,
            "router": router,
        },
        "delta": {
            "throughput_speedup": (
                optimized["tasks_per_second"] / baseline["tasks_per_second"]
            ),
            "wall_speedup": baseline["wall_seconds"] / optimized["wall_seconds"],
            "mean_latency_speedup": (
                baseline["mean_task_latency_seconds"]
                / optimized["mean_task_latency_seconds"]
            ),
            "mean_ttft_speedup": (
                baseline["mean_turn_ttft_seconds"]
                / optimized["mean_turn_ttft_seconds"]
            ),
            "paired_f1_delta": paired_f1_delta,
            "success_rate_delta": (
                optimized["success_rate"] - baseline["success_rate"]
            ),
            "peak_single_gpu_mib_reduction": baseline_peak - optimized_peak,
            "peak_single_gpu_percent_reduction": (
                100 * (baseline_peak - optimized_peak) / baseline_peak
                if baseline_peak
                else 0
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    baseline = report["baseline"]
    full = report["full_stack"]
    delta = report["delta"]
    activation = report["activation"]
    lines = [
        "# Equal-GPU-KV Full-Stack Agent A/B",
        "",
        f"- Strict equal GPU KV capacity: `{report['fairness']['strict_equal_gpu_kv_capacity']}`",
        f"- GPU KV blocks: `{report['fairness']['num_gpu_blocks_override_per_rank']}` per rank, "
        f"`{report['fairness']['total_gpu_blocks']}` total",
        f"- Paired tasks: `{report['paired_tasks']}`",
        f"- All five mechanisms observed: `{report['all_five_mechanisms_observed']}`",
        "",
        "| Mechanism | Runtime evidence |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name} | {'active' if active else 'not observed'} |"
        for name, active in activation.items()
    )
    lines.extend(
        [
            "",
            "| Metric | Baseline | Full stack | Delta |",
            "|---|---:|---:|---:|",
            f"| Wall time (s) | {baseline['wall_seconds']:.3f} | "
            f"{full['wall_seconds']:.3f} | {delta['wall_speedup']:.3f}x |",
            f"| Tasks/s | {baseline['tasks_per_second']:.4f} | "
            f"{full['tasks_per_second']:.4f} | {delta['throughput_speedup']:.3f}x |",
            f"| Mean task latency (s) | {baseline['mean_task_latency_seconds']:.3f} | "
            f"{full['mean_task_latency_seconds']:.3f} | "
            f"{delta['mean_latency_speedup']:.3f}x |",
            f"| Mean turn TTFT (s) | {baseline['mean_turn_ttft_seconds']:.3f} | "
            f"{full['mean_turn_ttft_seconds']:.3f} | "
            f"{delta['mean_ttft_speedup']:.3f}x |",
            f"| Mean F1 | {baseline['mean_f1']:.4f} | {full['mean_f1']:.4f} | "
            f"{delta['paired_f1_delta']:+.4f} |",
            f"| Success rate | {baseline['success_rate']:.4f} | "
            f"{full['success_rate']:.4f} | {delta['success_rate_delta']:+.4f} |",
            f"| Peak single-GPU MiB | {baseline['peak_single_gpu_mib']:.0f} | "
            f"{full['peak_single_gpu_mib']:.0f} | "
            f"{delta['peak_single_gpu_percent_reduction']:+.2f}% |",
            "",
            "## Full-stack counters",
            "",
            f"- Compaction saved chars: `{full['compaction_saved_chars']}`",
            f"- Trimmer sessions: `{full['trimmed_sessions']}`",
            f"- Released KV block references: `{full['released_block_references']}`",
            f"- Offload store bytes: `{full['offload_store_bytes']:.0f}`",
            f"- Offload load bytes: `{full['offload_load_bytes']:.0f}`",
            f"- Router route count: `{full['router'].get('route_count', 0)}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(load_result(args.baseline), load_result(args.full))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
