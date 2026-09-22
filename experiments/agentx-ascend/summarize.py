"""Summarize a completed official run, including drain cancellations from its log."""

import argparse
import json
import re
from pathlib import Path


def summarize(run_dir: Path) -> dict:
    report_path = run_dir / "artifacts/profile_export_aiperf.json"
    report = json.loads(report_path.read_text())
    log = (run_dir / "benchmark.log").read_text()
    completions = re.findall(
        r"Phase profiling \(profiling\) complete \| completed=(\d+), cancelled=(\d+), errors=(\d+)",
        log,
    )
    if len(completions) != 1:
        raise ValueError(
            "Expected exactly one completed profiling phase in benchmark.log"
        )
    completed, cancelled, errors = map(int, completions[0])
    if completed != report["request_count"]["avg"]:
        raise ValueError("Log and official report disagree on completed request count")
    rows = (
        json.loads(line)
        for line in (run_dir / "artifacts/profile_export.jsonl")
        .read_text()
        .splitlines()
    )
    child_records = sum(
        row["metadata"].get("agent_depth", 0) > 0
        for row in rows
        if row["metadata"]["benchmark_phase"] == "profiling"
        and not row["metadata"].get("was_cancelled", False)
    )
    manifest = json.loads((run_dir / "run-manifest.json").read_text())
    npus = manifest["tensor_parallel_size"] * manifest["data_parallel_size"]
    return {
        "submission_valid": report["metadata"]["submission_valid"],
        "was_cancelled": report["was_cancelled"],
        "completed_requests": completed,
        "subagent_requests": child_records,
        "request_errors": errors,
        "cancelled_at_drain_timeout": cancelled,
        "error_summary": report["error_summary"],
        "configured_duration_seconds": next(
            phase["duration"]
            for phase in report["input_config"]["phases"]
            if phase["kind"] == "profiling"
        ),
        "reported_benchmark_duration_seconds": report["benchmark_duration"]["avg"],
        "output_tokens_per_second": report["output_token_throughput"]["avg"],
        "total_tokens_per_npu_second_including_cached_inputs": report[
            "total_token_throughput"
        ]["avg"]
        / npus,
        "mean_output_tokens_per_second_per_user": report[
            "output_token_throughput_per_user"
        ]["avg"],
        "p90_output_tokens_per_second_per_user": report[
            "output_token_throughput_per_user"
        ]["p90"],
        "cache_read_percent": report["overall_usage_prompt_cache_read_pct"]["avg"],
        "ttft_ms": report["time_to_first_token"],
        "metric_duration_coverage": report["metadata"]["metric_duration_coverage"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = summarize(args.run_dir)
    output = json.dumps(result, indent=2) + "\n"
    (args.run_dir / "result-summary.json").write_text(output)
    print(output, end="")
