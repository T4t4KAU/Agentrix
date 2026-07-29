from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any


def load_arm(root: Path) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*/run.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        task_id = str(payload["task_id"])
        if task_id in runs:
            raise ValueError(f"duplicate task in {root}: {task_id}")
        runs[task_id] = payload
    return runs


def test_passed(run: dict[str, Any], name: str) -> bool:
    result = run.get("score", {}).get("tests", {}).get(name)
    return bool(result) and result.get("returncode") == 0


def arm_summary(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = list(runs.values())
    total_wall_ms = sum(float(run.get("total_wall_time_ms", 0)) for run in values)
    resolved = sum(bool(run.get("resolved")) for run in values)
    requests = [
        request for run in values for request in run.get("requests", [])
    ]
    ttfts = [
        float(request["ttft_ms"])
        for request in requests
        if request.get("ttft_ms") is not None
    ]
    return {
        "tasks": len(values),
        "resolved": resolved,
        "resolved_rate": resolved / len(values) if values else 0.0,
        "public_test_pass_rate": (
            sum(test_passed(run, "public") for run in values) / len(values)
            if values
            else 0.0
        ),
        "hidden_test_pass_rate": (
            sum(test_passed(run, "hidden") for run in values) / len(values)
            if values
            else 0.0
        ),
        "invalid_patch_rate": (
            sum(
                not run.get("score", {}).get("scope_valid", False)
                or not run.get("score", {}).get("public_test_unchanged", False)
                for run in values
            )
            / len(values)
            if values
            else 0.0
        ),
        "total_wall_time_s": total_wall_ms / 1000,
        "resolved_tasks_per_hour": (
            resolved * 3_600_000 / total_wall_ms if total_wall_ms else 0.0
        ),
        "input_tokens": sum(
            int(request.get("input_tokens", 0)) for request in requests
        ),
        "output_tokens": sum(
            int(request.get("output_tokens", 0)) for request in requests
        ),
        "ttft_mean_ms": statistics.fmean(ttfts) if ttfts else None,
    }


def process_memory_summary(path: Path) -> dict[str, Any] | None:
    """Summarize aggregate NVML process memory while an experiment arm ran."""
    if not path.exists():
        return None
    totals: dict[str, float] = {}
    gpu_peaks: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) != 4:
                continue
            timestamp, gpu_uuid, _pid, used_mib = (value.strip() for value in row)
            try:
                used = float(used_mib)
            except ValueError:
                continue
            totals[timestamp] = totals.get(timestamp, 0.0) + used
            gpu_peaks[gpu_uuid] = max(gpu_peaks.get(gpu_uuid, 0.0), used)
    if not totals:
        return None
    return {
        "peak_aggregate_process_memory_mib": max(totals.values()),
        "peak_process_memory_by_gpu_mib": gpu_peaks,
        "sample_count": len(totals),
    }


def paired_bootstrap_interval(
    baseline: dict[str, dict[str, Any]],
    optimized: dict[str, dict[str, Any]],
    *,
    samples: int = 20_000,
    seed: int = 2026,
) -> tuple[float, float, float]:
    task_ids = sorted(set(baseline) & set(optimized))
    if not task_ids:
        raise ValueError("the two arms have no paired tasks")
    deltas = [
        float(bool(optimized[task_id]["resolved"]))
        - float(bool(baseline[task_id]["resolved"]))
        for task_id in task_ids
    ]
    observed = statistics.fmean(deltas)
    generator = random.Random(seed)
    estimates = sorted(
        statistics.fmean(generator.choice(deltas) for _ in deltas)
        for _ in range(samples)
    )
    lower = estimates[int(samples * 0.025)]
    upper = estimates[min(samples - 1, int(samples * 0.975))]
    return observed, lower, upper


def build_report(
    baseline: dict[str, dict[str, Any]],
    optimized: dict[str, dict[str, Any]],
    *,
    margin: float,
    minimum_resolved_rate: float = 0.5,
) -> tuple[dict[str, Any], str]:
    if set(baseline) != set(optimized):
        missing_baseline = sorted(set(optimized) - set(baseline))
        missing_optimized = sorted(set(baseline) - set(optimized))
        raise ValueError(
            "unpaired task sets: "
            f"missing baseline={missing_baseline}, "
            f"missing optimized={missing_optimized}"
        )
    baseline_summary = arm_summary(baseline)
    optimized_summary = arm_summary(optimized)
    delta, lower, upper = paired_bootstrap_interval(baseline, optimized)
    quality_floor_passed = (
        baseline_summary["resolved_rate"] >= minimum_resolved_rate
        and optimized_summary["resolved_rate"] >= minimum_resolved_rate
    )
    noninferior = lower >= -margin and quality_floor_passed
    payload = {
        "schema_version": 1,
        "task_count": len(baseline),
        "noninferiority_margin": margin,
        "minimum_resolved_rate": minimum_resolved_rate,
        "baseline": baseline_summary,
        "optimized": optimized_summary,
        "resolved_rate_delta": delta,
        "resolved_rate_delta_bootstrap_95ci": [lower, upper],
        "noninferior": noninferior,
        "quality_floor_passed": quality_floor_passed,
        "paired_tasks": [
            {
                "task_id": task_id,
                "baseline_resolved": bool(baseline[task_id]["resolved"]),
                "optimized_resolved": bool(optimized[task_id]["resolved"]),
            }
            for task_id in sorted(baseline)
        ],
    }
    rows = []
    for name, summary in (
        ("Flash ordinary DP", baseline_summary),
        ("Agentrix", optimized_summary),
    ):
        rows.append(
            f"| {name} | {summary['resolved']}/{summary['tasks']} "
            f"({summary['resolved_rate']:.1%}) | "
            f"{summary['hidden_test_pass_rate']:.1%} | "
            f"{summary['invalid_patch_rate']:.1%} | "
            f"{summary['resolved_tasks_per_hour']:.2f} | "
            f"{summary['ttft_mean_ms'] or 0:.2f} |"
        )
    markdown = "\n".join(
        [
            "# Executable Coding-Agent Quality A/B",
            "",
            "| Arm | Resolved tasks | Hidden-test pass | Invalid patch | "
            "Resolved tasks/hour | Mean TTFT (ms) |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"Paired resolved-rate delta: **{delta:+.1%}**, paired bootstrap "
            f"95% CI **[{lower:+.1%}, {upper:+.1%}]**.",
            "",
            f"Predeclared non-inferiority margin: **-{margin:.1%}**. "
            f"Result: **{'passed' if noninferior else 'not established'}**.",
            "",
            f"Absolute quality floor: **{minimum_resolved_rate:.1%} resolved** "
            f"for both arms. Result: "
            f"**{'passed' if quality_floor_passed else 'failed'}**.",
            "",
            "A task is resolved only when the patch changes an allowed source file, "
            "leaves the public test unchanged, builds successfully, and passes both "
            "the public and hidden executable tests.",
            "",
        ]
    )
    return payload, markdown


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize paired executable Coding-Agent quality runs"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--minimum-resolved-rate", type=float, default=0.5)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    baseline = load_arm(args.baseline)
    optimized = load_arm(args.optimized)
    payload, markdown = build_report(
        baseline,
        optimized,
        margin=args.margin,
        minimum_resolved_rate=args.minimum_resolved_rate,
    )
    baseline_memory = process_memory_summary(
        args.baseline / "gpu_process_memory.csv"
    )
    optimized_memory = process_memory_summary(
        args.optimized / "gpu_process_memory.csv"
    )
    payload["baseline"]["gpu_process_memory"] = baseline_memory
    payload["optimized"]["gpu_process_memory"] = optimized_memory
    for name, root, summary in (
        ("Flash ordinary DP", args.baseline, baseline_memory),
        ("Agentrix", args.optimized, optimized_memory),
    ):
        metadata_path = root / "arm_metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        payload[
            "baseline" if root == args.baseline else "optimized"
        ]["arm_metadata"] = metadata
        if summary:
            markdown += (
                f"\n{name} peak aggregate NVML process memory: "
                f"**{summary['peak_aggregate_process_memory_mib']:.0f} MiB** "
                f"with **{metadata.get('num_gpu_blocks', 'unknown')}** KV blocks "
                "per rank.\n"
            )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
