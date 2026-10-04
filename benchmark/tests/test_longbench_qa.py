import hashlib
import json

from longbench_qa import build_shared_document_cases, normalize_answer, score_answer


def test_official_style_qa_scoring():
    assert normalize_answer("The South West Ultras.") == "south west ultras"
    assert score_answer("South West Ultras", ["the South West Ultras."]) == {
        "exact_match": 1.0,
        "f1": 1.0,
    }


def test_selects_repeated_longest_context(tmp_path):
    source = tmp_path / "qasper.jsonl"
    rows = [
        {
            "context": "long " * 20,
            "input": f"q{i}",
            "answers": ["a"],
            "_id": f"x{i}",
            "dataset": "qasper",
        }
        for i in range(2)
    ] + [
        {
            "context": "short",
            "input": "q",
            "answers": ["a"],
            "_id": "z",
            "dataset": "qasper",
        }
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    cases = build_shared_document_cases([source], maximum_cases=1)
    assert len(cases) == 1 and len(cases[0]["questions"]) == 2
    assert (
        cases[0]["context_sha256"]
        == hashlib.sha256(("long " * 20).encode()).hexdigest()
    )


def test_context_and_question_filters_do_not_truncate_or_duplicate_inputs(tmp_path):
    source = tmp_path / "narrativeqa.jsonl"
    rows = [
        {
            "context": context,
            "input": f"q{i}",
            "answers": [f"a{i}"],
            "_id": f"{context}-{i}",
        }
        for context, count in (
            ("short", 4),
            ("long enough", 4),
            ("too few questions", 2),
        )
        for i in range(count)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    cases = build_shared_document_cases(
        [source],
        minimum_questions=4,
        maximum_questions=4,
        token_counter=len,
        minimum_context_tokens=10,
        maximum_context_tokens=30,
    )
    assert len(cases) == 1
    assert cases[0]["context"] == "long enough"
    assert [q["question"] for q in cases[0]["questions"]] == ["q0", "q1", "q2", "q3"]
    assert [q["answers"] for q in cases[0]["questions"]] == [
        ["a0"],
        ["a1"],
        ["a2"],
        ["a3"],
    ]


def test_unique_question_selection_uses_first_original_row_not_answer_quality(tmp_path):
    path = tmp_path / "narrativeqa.jsonl"
    rows = [
        {"context": "document", "input": question, "answers": [answer], "_id": str(i)}
        for i, (question, answer) in enumerate(
            [("q0", "first"), ("q0", "alternate"), ("q1", "last")]
        )
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert (
        build_shared_document_cases([path], unique_questions=True, minimum_questions=3)
        == []
    )
    cases = build_shared_document_cases(
        [path], unique_questions=True, minimum_questions=2
    )
    assert [q["source_id"] for q in cases[0]["questions"]] == ["0", "2"]
    assert cases[0]["questions"][0]["answers"] == ["first"]
