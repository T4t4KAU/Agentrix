import pytest

from accuracy import compare_raw_results, normalize_text, token_f1


def test_normalize_and_token_f1() -> None:
    assert normalize_text(" Hello\n WORLD ") == "hello world"
    assert token_f1("one two two", "one two") == pytest.approx(0.8)


def test_compare_raw_results_aligns_batches_and_scopes() -> None:
    reference = [
        {
            "sample_start": 4,
            "common": {"cases": [{"case_index": 0, "text": "Shared answer"}]},
            "branches": [{"case_index": 0, "branch_id": 0, "text": "Final answer"}],
        }
    ]
    candidate = [
        {
            "sample_start": 4,
            "common": {"cases": [{"case_index": 0, "text": " shared  answer "}]},
            "branches": [{"case_index": 0, "branch_id": 0, "text": "Different answer"}],
        }
    ]

    result = compare_raw_results(reference, candidate)

    assert result["matched_outputs"] == 2
    assert result["by_scope"]["common"]["normalized_exact_match_percent"] == 100
    assert result["by_scope"]["branch"]["normalized_exact_match_percent"] == 0
    assert result["missing_from_candidate"] == []
