import pytest

from longbench_qa_report import compare


def test_paired_report():
    left = {
        "mean_f1": 0.5,
        "wall_seconds": 10,
        "questions_per_second": 2,
        "mean_ttft_seconds": 4,
        "results": [
            {"source_id": "a", "f1": 0.5, "prediction": "X"}
        ],
    }
    right = {
        "mean_f1": 0.6,
        "wall_seconds": 5,
        "questions_per_second": 4,
        "mean_ttft_seconds": 2,
        "results": [
            {"source_id": "a", "f1": 0.6, "prediction": "x"}
        ],
    }
    result = compare(left, right)
    assert result["wall_speedup"] == 2
    assert result["prediction_exact_agreement"] == 1


def test_paired_report_rejects_different_request_sets():
    left = {
        "mean_f1": 0,
        "wall_seconds": 1,
        "questions_per_second": 1,
        "mean_ttft_seconds": 1,
        "results": [
            {"source_id": "left", "f1": 0, "prediction": ""}
        ],
    }
    right = {
        **left,
        "results": [
            {"source_id": "right", "f1": 0, "prediction": ""}
        ],
    }
    with pytest.raises(ValueError, match="request sets differ"):
        compare(left, right)
