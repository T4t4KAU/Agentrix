"""Report the strict equal-GPU-KV real tool-agent A/B."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from full_stack_agent import load_result, validate_equal_kv_metadata


def _peak_gpu_mib(payload: dict[str, Any]) -> tuple[float, float]:
    observations: list[tuple[float, float]] = []
    for sample in payload.get("gpu_memory_samples", []):
        values = [
            float(gpu["used_mib"])
            for gpu in sample.get("gpus", [])
            if isinstance(gpu.get("used_mib"), (int, float))
        ]
        if values:
            observations.append((max(values), sum(values)))
    if not observations:
        return 0.0, 0.0
    return (
        max(value[0] for value in observations),
        max(value[1] for value in observations),
    )


def build_report(
    baseline: dict[str, Any], agentrix: dict[str, Any]
) -> dict[str, Any]:
    fairness = validate_equal_kv_metadata(
        baseline["metadata"], agentrix["metadata"]
    )
    baseline_rows = {
        (row["case_id"], row["source_id"]): row
        for row in baseline["results"]
    }
    agentrix_rows = {
        (row["case_id"], row["source_id"]): row
        for row in agentrix["results"]
    }
    paired = sorted(baseline_rows.keys() & agentrix_rows.keys())
    if not paired:
        raise ValueError("A/B results contain no paired tasks")
    router = agentrix.get("router_stats", {})
    fork = agentrix.get("kv", {}).get("fork_execution", {})
    activation = {
        "forkattention_backend": (
            agentrix["metadata"].get("attention_backend") == "FORK_ATTN"
        ),
        "forkattention_shared_execution": fork.get("active_steps", 0) > 0,
        "prefix_router": bool(router.get("active"))
        and router.get("route_count", 0) > 0,
        "real_tool_loop": all(
            len(agentrix_rows[key].get("tool_events", []))
            == agentrix["metadata"]["tool_rounds"]
            for key in paired
        ),
        "prompt_compaction_disabled": (
            not agentrix["metadata"].get("compaction", False)
            and agentrix.get("compaction_saved_chars", 0) == 0
        ),
        "ttl_trimmer_disabled": not agentrix["metadata"].get("trimmer", False),
        "kv_offload_disabled": not agentrix["metadata"].get("offload", False),
    }
    baseline_peak, baseline_sum = _peak_gpu_mib(baseline)
    agentrix_peak, agentrix_sum = _peak_gpu_mib(agentrix)
    paired_f1_delta = statistics.fmean(
        agentrix_rows[key]["f1"] - baseline_rows[key]["f1"]
        for key in paired
    )
    exact_prediction_agreement = statistics.fmean(
        agentrix_rows[key]["prediction"].strip()
        == baseline_rows[key]["prediction"].strip()
        for key in paired
    )
    return {
        "schema_version": 1,
        "fairness": fairness,
        "paired_tasks": len(paired),
        "activation": activation,
        "core_agentrix_path_observed": all(
            activation[key]
            for key in (
                "forkattention_backend",
                "forkattention_shared_execution",
                "prefix_router",
                "real_tool_loop",
                "prompt_compaction_disabled",
                "ttl_trimmer_disabled",
                "kv_offload_disabled",
            )
        ),
        "baseline": {
            "wall_seconds": baseline["wall_seconds"],
            "tasks_per_second": baseline["tasks_per_second"],
            "mean_f1": baseline["mean_f1"],
            "success_rate": baseline["success_rate"],
            "agent_completion_rate": baseline["agent_completion_rate"],
            "valid_citation_rate": baseline["valid_citation_rate"],
            "tool_call_valid_rate": baseline["tool_call_valid_rate"],
            "mean_task_latency_seconds": baseline["mean_task_latency_seconds"],
            "mean_turn_ttft_seconds": baseline["mean_turn_ttft_seconds"],
            "peak_single_gpu_mib": baseline_peak,
            "peak_gpu_sum_mib": baseline_sum,
        },
        "agentrix": {
            "wall_seconds": agentrix["wall_seconds"],
            "tasks_per_second": agentrix["tasks_per_second"],
            "mean_f1": agentrix["mean_f1"],
            "success_rate": agentrix["success_rate"],
            "agent_completion_rate": agentrix["agent_completion_rate"],
            "valid_citation_rate": agentrix["valid_citation_rate"],
            "tool_call_valid_rate": agentrix["tool_call_valid_rate"],
            "mean_task_latency_seconds": agentrix["mean_task_latency_seconds"],
            "mean_turn_ttft_seconds": agentrix["mean_turn_ttft_seconds"],
            "peak_single_gpu_mib": agentrix_peak,
            "peak_gpu_sum_mib": agentrix_sum,
            "router": router,
            "fork_execution": fork,
        },
        "delta": {
            "throughput_speedup": (
                agentrix["tasks_per_second"] / baseline["tasks_per_second"]
            ),
            "wall_speedup": baseline["wall_seconds"] / agentrix["wall_seconds"],
            "mean_latency_speedup": (
                baseline["mean_task_latency_seconds"]
                / agentrix["mean_task_latency_seconds"]
            ),
            "mean_ttft_speedup": (
                baseline["mean_turn_ttft_seconds"]
                / agentrix["mean_turn_ttft_seconds"]
            ),
            "paired_f1_delta": paired_f1_delta,
            "success_rate_delta": (
                agentrix["success_rate"] - baseline["success_rate"]
            ),
            "agent_completion_rate_delta": (
                agentrix["agent_completion_rate"]
                - baseline["agent_completion_rate"]
            ),
            "valid_citation_rate_delta": (
                agentrix["valid_citation_rate"]
                - baseline["valid_citation_rate"]
            ),
            "exact_prediction_agreement": exact_prediction_agreement,
            "peak_single_gpu_mib_reduction": baseline_peak - agentrix_peak,
            "peak_single_gpu_percent_reduction": (
                100 * (baseline_peak - agentrix_peak) / baseline_peak
                if baseline_peak
                else 0
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    baseline = report["baseline"]
    agentrix = report["agentrix"]
    delta = report["delta"]
    lines = [
        "# Real Tool-Agent Equal-GPU-KV A/B",
        "",
        f"- Strict equal GPU KV capacity: `{report['fairness']['strict_equal_gpu_kv_capacity']}`",
        f"- GPU KV blocks: `{report['fairness']['num_gpu_blocks_override_per_rank']}` per rank",
        f"- Paired tasks: `{report['paired_tasks']}`",
        f"- Core Agentrix path observed: `{report['core_agentrix_path_observed']}`",
        "- Prompt Compaction, TTL Trimmer, and KV Offload are intentionally disabled in both arms.",
        "",
        "| Runtime check | Result |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name} | {'yes' if value else 'no'} |"
        for name, value in report["activation"].items()
    )
    lines.extend(
        [
            "",
            "| Metric | Flash ordinary DP | Agentrix | Delta |",
            "|---|---:|---:|---:|",
            f"| Wall time (s) | {baseline['wall_seconds']:.3f} | "
            f"{agentrix['wall_seconds']:.3f} | {delta['wall_speedup']:.3f}x |",
            f"| Tasks/s | {baseline['tasks_per_second']:.4f} | "
            f"{agentrix['tasks_per_second']:.4f} | "
            f"{delta['throughput_speedup']:.3f}x |",
            f"| Mean task latency (s) | {baseline['mean_task_latency_seconds']:.3f} | "
            f"{agentrix['mean_task_latency_seconds']:.3f} | "
            f"{delta['mean_latency_speedup']:.3f}x |",
            f"| Mean turn TTFT (s) | {baseline['mean_turn_ttft_seconds']:.3f} | "
            f"{agentrix['mean_turn_ttft_seconds']:.3f} | "
            f"{delta['mean_ttft_speedup']:.3f}x |",
            f"| Mean F1 | {baseline['mean_f1']:.4f} | "
            f"{agentrix['mean_f1']:.4f} | {delta['paired_f1_delta']:+.4f} |",
            f"| Success rate | {baseline['success_rate']:.4f} | "
            f"{agentrix['success_rate']:.4f} | "
            f"{delta['success_rate_delta']:+.4f} |",
            f"| Agent completion rate | {baseline['agent_completion_rate']:.4f} | "
            f"{agentrix['agent_completion_rate']:.4f} | "
            f"{delta['agent_completion_rate_delta']:+.4f} |",
            f"| Valid citation rate | {baseline['valid_citation_rate']:.4f} | "
            f"{agentrix['valid_citation_rate']:.4f} | "
            f"{delta['valid_citation_rate_delta']:+.4f} |",
            f"| Peak single-GPU MiB | {baseline['peak_single_gpu_mib']:.0f} | "
            f"{agentrix['peak_single_gpu_mib']:.0f} | "
            f"{delta['peak_single_gpu_percent_reduction']:+.2f}% |",
            "",
            f"- Router route count: `{agentrix['router'].get('route_count', 0)}`",
            f"- ForkAttention shared CTAs: `{agentrix['fork_execution'].get('shared_ctas', 0)}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--agentrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        load_result(args.baseline), load_result(args.agentrix)
    )
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
