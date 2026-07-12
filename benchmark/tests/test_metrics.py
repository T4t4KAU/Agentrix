import pytest

from metrics import baseline_metrics, compare_trace, monowire_metrics
from models import BenchmarkTrace, BranchTrace


def test_reference_case() -> None:
    suffixes = [512] * 4
    baseline = baseline_metrics(8192, suffixes)
    monowire = monowire_metrics(8192, suffixes)
    assert baseline == {
        "valid_qk": 34816,
        "unique_kv_tokens": 34816,
        "scheduled_tiles": 272,
        "logical_launches": 4,
    }
    assert monowire == {
        "valid_qk": 34816,
        "unique_kv_tokens": 10240,
        "scheduled_tiles": 80,
        "logical_launches": 1,
    }


def test_valid_work_is_preserved() -> None:
    trace = BenchmarkTrace(
        "case", 16384, [BranchTrace(0, 1, 10), BranchTrace(1, 257, 10)]
    )
    result = compare_trace(trace)
    assert result["baseline_valid_qk"] == result["monowire_valid_qk"]
    assert result["launch_reduction"] == 2
    assert result["kv_tokens_saved"] == 16384
    assert result["kv_reduction_percent"] == pytest.approx(49.61, rel=1e-3)


def test_total_kv_bytes_saved() -> None:
    trace = BenchmarkTrace("case", 8192, [BranchTrace(i, 512, 1) for i in range(4)])
    result = compare_trace(trace, kv_bytes_per_token=114688)
    assert result["kv_tokens_saved"] == 24576
    assert result["kv_bytes_saved"] == 2818572288
    assert result["kv_gib_saved"] == pytest.approx(2.625)


@pytest.mark.parametrize("function", [baseline_metrics, monowire_metrics])
def test_invalid_arguments(function) -> None:
    with pytest.raises(ValueError):
        function(10, [], 128)
    with pytest.raises(ValueError):
        function(10, [1], 0)
