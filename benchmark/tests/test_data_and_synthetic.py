import json
import random

from data import load_records, record_to_prompt
from synthetic import sample_suffixes, traces_from_config


def test_load_and_render_swebench(tmp_path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text(
        json.dumps(
            {
                "repo": "org/repo",
                "base_commit": "abc",
                "instance_id": "issue-1",
                "problem_statement": "broken",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record = load_records("swebench", path)[0]
    assert "org/repo" in record_to_prompt("swebench", record)


def test_load_and_render_agentboard(tmp_path) -> None:
    prompt_dir = tmp_path / "agentboard" / "prompts" / "VanillaAgent"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "pddl.json").write_text(
        json.dumps(
            {
                "blockworld": {
                    "instruction": "Stack the blocks safely.",
                    "examples": ["pickup b1; stack b1 b2"],
                }
            }
        ),
        encoding="utf-8",
    )

    record = load_records("agentboard", tmp_path)[0]
    assert "Stack the blocks" in record_to_prompt("agentboard", record)


def test_load_and_render_appworld(tmp_path) -> None:
    prompt_dir = tmp_path / "experiments" / "prompts" / "function_calling_agent"
    prompt_dir.mkdir(parents=True)
    (prompt_dir / "demos.json").write_text(
        json.dumps([{"role": "user", "content": "Count my queued songs."}]),
        encoding="utf-8",
    )

    record = load_records("appworld", tmp_path)[0]
    assert "queued songs" in record_to_prompt("appworld", record)


def test_distributions_are_deterministic() -> None:
    left = sample_suffixes(8, "lognormal", 768, random.Random(7))
    right = sample_suffixes(8, "lognormal", 768, random.Random(7))
    assert left == right
    assert len(set(left)) > 1
    assert abs(sum(left) / len(left) - 768) < 1


def test_minimum_matrix_has_18_cases() -> None:
    traces = traces_from_config(
        {
            "seed": 1,
            "prefix_tokens": [8192, 16384, 32768],
            "branch_counts": [4, 8, 16],
            "suffix_distributions": ["uniform", "lognormal"],
            "suffix_mean": 768,
            "output_tokens": 256,
        }
    )
    assert len(traces) == 18
    assert len({trace.case_id for trace in traces}) == 18
