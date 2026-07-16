import json

import pytest

from hotpot import (
    HotpotCorpus,
    answer_scores,
    evaluate_prediction,
    evaluate_predictions,
    load_hotpot,
    stratified_sample,
    supporting_fact_scores,
)


def _record(index: int, question_type: str, level: str) -> dict:
    return {
        "_id": f"case-{index}",
        "question": f"Question {index}?",
        "answer": "The Alpha",
        "supporting_facts": [[f"Title {index}", 1]],
        "context": [
            [f"Title {index}", ["First sentence. ", "Second sentence."]],
            ["Shared", ["Shared paragraph."]],
        ],
        "type": question_type,
        "level": level,
    }


def test_loads_examples_and_builds_sentence_corpus(tmp_path) -> None:
    path = tmp_path / "hotpot.json"
    path.write_text(
        json.dumps([_record(1, "bridge", "hard"), _record(2, "comparison", "easy")]),
        encoding="utf-8",
    )

    examples = load_hotpot(path)
    assert examples[0].example_id == "case-1"
    assert examples[0].supporting_facts == (("Title 1", 1),)
    assert examples[0].context[0].text() == "First sentence. Second sentence."

    corpus = HotpotCorpus.from_examples(examples)
    assert len(corpus) == 3
    assert corpus.get_sentence("Title 1", 1).text == "Second sentence."
    assert len(corpus.sentences) == 5
    with pytest.raises(KeyError):
        corpus.get_sentence("Title 1", 9)


def test_loader_rejects_malformed_context(tmp_path) -> None:
    record = _record(1, "bridge", "hard")
    record["context"] = [["Title", "not a sentence list"]]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([record]), encoding="utf-8")

    with pytest.raises(ValueError, match=r"expected \[title"):
        load_hotpot(path)


def test_stratified_sampling_is_seeded_and_input_order_independent(tmp_path) -> None:
    records = [
        _record(index, question_type, level)
        for index, (question_type, level) in enumerate(
            [
                ("bridge", "hard"),
                ("bridge", "hard"),
                ("bridge", "easy"),
                ("bridge", "easy"),
                ("comparison", "hard"),
                ("comparison", "hard"),
                ("comparison", "easy"),
                ("comparison", "easy"),
            ]
        )
    ]
    path = tmp_path / "hotpot.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    examples = load_hotpot(path)

    first = stratified_sample(examples, 4, seed=17)
    second = stratified_sample(list(reversed(examples)), 4, seed=17)
    assert [example.id for example in first] == [example.id for example in second]
    assert len({(example.question_type, example.level) for example in first}) == 4
    assert [example.id for example in first] != [
        example.id for example in stratified_sample(examples, 4, seed=23)
    ]


def test_answer_and_supporting_fact_metrics_match_official_behavior() -> None:
    answer = answer_scores("alpha beta", "The alpha beta gamma")
    assert answer["em"] == 0.0
    assert answer["prec"] == 1.0
    assert answer["recall"] == pytest.approx(2 / 3)
    assert answer["f1"] == pytest.approx(0.8)
    assert answer_scores("yes", "no")["f1"] == 0.0

    supporting = supporting_fact_scores(
        [["A", 0], ["B", 1], ["B", 1]],
        [["A", 0], ["C", 2]],
    )
    assert supporting == {
        "em": 0.0,
        "f1": 0.5,
        "prec": 0.5,
        "recall": 0.5,
    }


def test_joint_and_macro_evaluation(tmp_path) -> None:
    path = tmp_path / "hotpot.json"
    path.write_text(
        json.dumps([_record(1, "bridge", "hard"), _record(2, "comparison", "easy")]),
        encoding="utf-8",
    )
    examples = load_hotpot(path)
    perfect = evaluate_prediction(
        "alpha", [["Title 1", 1]], "The Alpha", [["Title 1", 1]]
    )
    assert perfect["joint_em"] == 1.0
    assert perfect["joint_f1"] == 1.0

    # The second example is missing and therefore remains zero in the macro mean.
    metrics = evaluate_predictions(
        examples,
        {"answer": {"case-1": "alpha"}, "sp": {"case-1": [["Title 1", 1]]}},
    )
    assert metrics["em"] == 0.5
    assert metrics["sp_f1"] == 0.5
    assert metrics["joint_f1"] == 0.5
