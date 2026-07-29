from longbench_qa_report import compare

def test_paired_report():
    left = {"mean_f1": .5, "wall_seconds": 10, "questions_per_second": 2,
            "mean_ttft_seconds": 4, "results": [{"source_id": "a", "f1": .5, "prediction": "X"}]}
    right = {"mean_f1": .6, "wall_seconds": 5, "questions_per_second": 4,
             "mean_ttft_seconds": 2, "results": [{"source_id": "a", "f1": .6, "prediction": "x"}]}
    result = compare(left, right)
    assert result["wall_speedup"] == 2 and result["prediction_exact_agreement"] == 1
