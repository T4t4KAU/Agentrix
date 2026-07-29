from __future__ import annotations

import pytest

from coding_quality_report import (
    arm_summary,
    build_report,
    process_memory_summary,
)


def run(task_id: str, resolved: bool, wall_ms: float = 1000) -> dict:
    return {
        "task_id": task_id,
        "resolved": resolved,
        "total_wall_time_ms": wall_ms,
        "score": {
            "scope_valid": True,
            "public_test_unchanged": True,
            "tests": {
                "public": {"returncode": 0 if resolved else 1},
                "hidden": {"returncode": 0 if resolved else 1},
            },
        },
        "requests": [
            {
                "input_tokens": 100,
                "output_tokens": 10,
                "ttft_ms": 25.0,
            }
        ],
    }


def test_arm_summary_reports_quality_adjusted_throughput() -> None:
    summary = arm_summary(
        {
            "a": run("a", True, 1000),
            "b": run("b", False, 1000),
        }
    )

    assert summary["resolved_rate"] == 0.5
    assert summary["hidden_test_pass_rate"] == 0.5
    assert summary["resolved_tasks_per_hour"] == 1800
    assert summary["input_tokens"] == 200


def test_build_report_passes_equal_quality_as_noninferior() -> None:
    baseline = {"a": run("a", True), "b": run("b", False)}
    optimized = {"a": run("a", True), "b": run("b", False)}

    payload, markdown = build_report(baseline, optimized, margin=0.05)

    assert payload["resolved_rate_delta"] == 0
    assert payload["noninferior"] is True
    assert "passed" in markdown


def test_build_report_rejects_unpaired_tasks() -> None:
    baseline = {"a": run("a", True)}
    optimized = {"b": run("b", True)}

    with pytest.raises(ValueError, match="unpaired task sets"):
        build_report(baseline, optimized, margin=0.05)


def test_build_report_rejects_vacuous_equal_failure() -> None:
    baseline = {"a": run("a", False), "b": run("b", False)}
    optimized = {"a": run("a", False), "b": run("b", False)}

    payload, _ = build_report(baseline, optimized, margin=0.05)

    assert payload["resolved_rate_delta"] == 0
    assert payload["quality_floor_passed"] is False
    assert payload["noninferior"] is False


def test_process_memory_summary_aggregates_by_timestamp(tmp_path) -> None:
    path = tmp_path / "gpu.csv"
    path.write_text(
        "2026/07/26 01:00:00.000, GPU-a, 1, 100\n"
        "2026/07/26 01:00:00.000, GPU-b, 2, 200\n"
        "2026/07/26 01:00:01.000, GPU-a, 1, 150\n"
        "2026/07/26 01:00:01.000, GPU-b, 2, 250\n"
    )

    summary = process_memory_summary(path)

    assert summary is not None
    assert summary["peak_aggregate_process_memory_mib"] == 400
    assert summary["peak_process_memory_by_gpu_mib"]["GPU-a"] == 150
