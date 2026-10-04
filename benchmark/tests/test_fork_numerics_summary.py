import copy
import hashlib
import json

import pytest
from summarize_fork_numerics import compare_numeric, compare_qa, numeric_rows, summarize
from test_fork_scale_summary import run


def numeric_run():
    result = run()
    result["config"]["trials"] = 3
    trial = result["shapes"][0]["trials"][0]
    for row in trial["rows"]:
        row.update(
            token_ids=[1] * 4,
            logprob_tokens=["token_id:1"] * 4,
            token_logprobs=[-0.1] * 4,
            top_logprobs=[{"token_id:1": -0.1, "token_id:2": -3}] * 4,
            output_token_sha256=hashlib.sha256(
                json.dumps([1] * 4).encode()
            ).hexdigest(),
        )
    result["shapes"][0]["trials"] = [
        dict(copy.deepcopy(trial), trial=i) for i in range(3)
    ]
    return result


def test_native_repeats_are_compared_without_attributing_them_to_fork():
    data = numeric_run()
    row = data["shapes"][0]["trials"][1]["rows"][0]
    row.update(
        token_ids=[2] * 4,
        logprob_tokens=["token_id:2"] * 4,
        token_logprobs=[-0.1] * 4,
        top_logprobs=[{"token_id:2": -0.1, "token_id:1": -3}] * 4,
        output_token_sha256=hashlib.sha256(json.dumps([2] * 4).encode()).hexdigest(),
    )
    group = compare_numeric(data)[0]
    assert group["pairs"] == 12  # Three pairs of trials, four requests per trial.
    assert group["equal_pairs"] == 10
    assert len(group["different_pairs"]) == 2
    assert "wall_seconds" not in group


@pytest.mark.parametrize("field", ["output_token_sha256", "logprob_tokens"])
def test_capture_digest_and_token_label_alignment_are_checked(field):
    data = numeric_run()
    row = data["shapes"][0]["trials"][0]["rows"][0]
    row[field] = "a" * 64 if field == "output_token_sha256" else ["token_id:2"] * 4
    with pytest.raises(ValueError):
        numeric_rows(data)


def qa_fixture():
    question = {"source_id": "q", "question": "Where?", "answers": ["Paris"]}
    case = {
        "case_id": "doc",
        "context_sha256": "abc",
        "context_tokens": 100,
        "questions": [question],
    }
    row = {
        **case,
        **question,
        "phase": "first",
        "prediction": "Paris",
        "f1": 1.0,
        "exact_match": 1.0,
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 128, "completion_tokens": 1},
    }
    report = {
        "case_count": 1,
        "question_count": 1,
        "prime_first_question": True,
        "question_waves": False,
        "coalesce_prefill": False,
        "cases_sha256": "frozen",
        "seed": 42,
        "model": "model",
        "document_routing": True,
        "wall_seconds": 10.0,
        "phase_seconds": {"first": 6.0, "followup": 4.0},
        "mean_f1": 1.0,
        "mean_exact_match": 1.0,
        "mean_ttft_seconds": 2.0,
        "mean_end_to_end_ttft_seconds": 3.0,
        "prompt_tokens": 128,
        "completion_tokens": 1,
        "results": [row],
    }
    return report, [case]


def test_faster_qa_with_lower_quality_fails_gate():
    a, cases = qa_fixture()
    b = copy.deepcopy(a)
    b.update(wall_seconds=5, mean_f1=0, mean_exact_match=0)
    b["results"][0].update(prediction="London", f1=0, exact_match=0)
    result = compare_qa(a, b, cases)
    assert result["wall_change_pct"] == -50
    assert result["quality_gate_passed"] is False
    assert result["equal_predictions"] == 0


def test_original_reference_and_reported_score_are_audited():
    a, cases = qa_fixture()
    b = copy.deepcopy(a)
    b["results"][0]["answers"] = ["London"]
    with pytest.raises(ValueError, match="QA source changed"):
        compare_qa(a, b, cases)
    b = copy.deepcopy(a)
    b["results"][0]["prediction"] = "London"
    with pytest.raises(ValueError, match="score mismatch"):
        compare_qa(a, b, cases)


def test_partial_matrix_cannot_be_reported_as_complete(tmp_path):
    (tmp_path / "validation-plan.json").write_text(
        json.dumps(
            {
                "services": ["native-0", "fork-0", "fork-1", "native-1"],
                "numeric_modes": ["concurrent", "batched"],
            }
        )
    )
    (tmp_path / "progress.json").write_text(
        json.dumps({"phase": "numerics", "completed": []})
    )
    with pytest.raises(ValueError, match="incomplete"):
        summarize(tmp_path)
