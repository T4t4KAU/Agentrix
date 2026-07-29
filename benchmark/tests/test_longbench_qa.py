import hashlib, json
from longbench_qa import build_shared_document_cases, normalize_answer, score_answer

def test_official_style_qa_scoring():
    assert normalize_answer("The South West Ultras.") == "south west ultras"
    assert score_answer("South West Ultras", ["the South West Ultras."]) == {"exact_match": 1.0, "f1": 1.0}

def test_selects_repeated_longest_context(tmp_path):
    source = tmp_path / "qasper.jsonl"
    rows = [
        {"context": "long " * 20, "input": f"q{i}", "answers": ["a"], "_id": f"x{i}", "dataset": "qasper"}
        for i in range(2)
    ] + [{"context": "short", "input": "q", "answers": ["a"], "_id": "z", "dataset": "qasper"}]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    cases = build_shared_document_cases([source], maximum_cases=1)
    assert len(cases) == 1 and len(cases[0]["questions"]) == 2
    assert cases[0]["context_sha256"] == hashlib.sha256(("long " * 20).encode()).hexdigest()
