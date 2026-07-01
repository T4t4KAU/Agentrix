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


@pytest.mark.parametrize("function", [baseline_metrics, monowire_metrics])
def test_invalid_arguments(function) -> None:
    with pytest.raises(ValueError):
        function(10, [], 128)
    with pytest.raises(ValueError):
        function(10, [1], 0)
