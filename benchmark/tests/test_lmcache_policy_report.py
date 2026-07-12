import csv
from pathlib import Path

import pytest

from lmcache_policy_report import parse_lmcache_log, read_results, render_report


FIELDNAMES = [
    "case_wall_time_ms",
    "branch_phase_wall_ms",
    "branch_total_output_tokens",
    "baseline_unique_kv",
    "monowire_unique_kv",
    "kv_bytes_per_token",
    "kv_bytes_saved",
]


def _write_run(root: Path, name: str, reload_tokens: int) -> None:
    run = root / name / "fork_attn"
    run.mkdir(parents=True)
    with (run / "benchmark_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for _ in range(2):
            writer.writerow(
                {
                    "case_wall_time_ms": 100,
                    "branch_phase_wall_ms": 50,
                    "branch_total_output_tokens": 20,
                    "baseline_unique_kv": 1000,
                    "monowire_unique_kv": 600,
                    "kv_bytes_per_token": 1024,
                    "kv_bytes_saved": 409600,
                }
            )
    (run / "vllm_server.log").write_text(
        f"Reqid: req-1, need to load: {reload_tokens}\n"
        f"Reqid: req-1, need to load: {reload_tokens}\n"
        "Retrieved 128 out of 256 required tokens.\n"
        "Stored 256 out of total 256 tokens.\n"
        "Memory allocation failed during disk load for key x\n",
        encoding="utf-8",
    )
    (root / name / "smoke_summary.txt").write_text(
        f"status=passed\ndisk_bytes={reload_tokens * 1024}\n",
        encoding="utf-8",
    )


def test_read_results_aggregates_cases(tmp_path: Path) -> None:
    _write_run(tmp_path, "baseline", 512)
    result = read_results(tmp_path / "baseline" / "fork_attn" / "benchmark_results.csv")
    assert result["case_wall_time_ms"] == 200
    assert result["end_to_end_output_tokens_per_s"] == 200
    assert result["kv_reduction_percent"] == pytest.approx(40)


def test_parse_lmcache_log_counts_reload_and_failures(tmp_path: Path) -> None:
    _write_run(tmp_path, "baseline", 512)
    result = parse_lmcache_log(tmp_path / "baseline" / "fork_attn" / "vllm_server.log")
    assert result == {
        "reload_demand_tokens": 512,
        "retrieved_tokens": 128,
        "stored_tokens": 256,
        "disk_load_allocation_failures": 1,
    }


def test_report_uses_default_lmcache_as_baseline(tmp_path: Path) -> None:
    _write_run(tmp_path, "baseline", 512)
    _write_run(tmp_path, "optimized", 256)
    report = render_report(tmp_path, "LRU", "FORK_AWARE")
    assert "Baseline: LMCache default `LRU` policy." in report
    assert "256 tokens, 0.000 GiB, 50.00%" in report
    assert "**Disk-tier footprint reduction:** 0.000 GiB, 50.00%." in report
    assert "| KV reload demand tokens | 512 | 256 | +50.00% |" in report
    assert "| Disk load allocation failures | 1 | 1 | +0.00% |" in report


def test_report_does_not_invent_percent_from_zero_baseline(tmp_path: Path) -> None:
    _write_run(tmp_path, "baseline", 512)
    _write_run(tmp_path, "optimized", 256)
    baseline_log = tmp_path / "baseline" / "fork_attn" / "vllm_server.log"
    baseline_log.write_text("Reqid: req-1, need to load: 512\n", encoding="utf-8")
    report = render_report(tmp_path, "LRU", "FORK_AWARE")
    assert "| Disk load allocation failures | 0 | 1 | n/a |" in report
