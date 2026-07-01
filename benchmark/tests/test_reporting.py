import csv

from models import BenchmarkTrace, BranchTrace
from reporting import write_results


def test_write_results_includes_measured_latency(tmp_path) -> None:
    trace = BenchmarkTrace(
        case_id="case",
        prefix_tokens=1024,
        branches=[
            BranchTrace(0, 16, 10, input_tokens=1040, latency_ms=100.0),
            BranchTrace(1, 32, 20, input_tokens=1056, latency_ms=200.0),
        ],
    )
    result = {
        "case_id": "case",
        "prefix_tokens": 1024,
        "branch_count": 2,
        "suffix_distribution": "observed",
        "suffix_mean": 24.0,
        "suffix_std": 8.0,
        "baseline_valid_qk": 2096,
        "monowire_valid_qk": 2096,
        "baseline_unique_kv": 2096,
        "monowire_unique_kv": 1072,
        "kv_reduction": 1.955,
        "baseline_tiles": 18,
        "monowire_tiles": 9,
        "tile_reduction": 2.0,
        "baseline_launches": 2,
        "monowire_launches": 1,
        "launch_reduction": 2.0,
    }
    raw = {
        "common": {"latency_ms": 50.0},
        "branch_phase_latency_ms": 225.0,
        "total_latency_ms": 300.0,
    }

    write_results(tmp_path, [trace], [result], raw)

    with (tmp_path / "benchmark_results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["common_latency_ms"] == "50.0"
    assert rows[0]["branch_mean_latency_ms"] == "150.0"
    assert rows[0]["branch_phase_wall_ms"] == "225.0"
    assert rows[0]["branch_phase_wall_source"] == "measured_branch_phase"
    assert rows[0]["case_wall_time_ms"] == "300.0"
    assert rows[0]["case_wall_time_source"] == "measured_total"
    assert rows[0]["approx_wall_time_ms"] == "300.0"
    assert rows[0]["branch_total_output_tokens"] == "30"
    assert float(rows[0]["branch_output_tokens_per_s"]) == 30 / 0.225

    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Measured End-to-End Latency" in summary
    assert "Logical Savings" in summary
