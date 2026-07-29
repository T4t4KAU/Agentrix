import pytest

from coding_agent_e2e_runner import (
    execute_action,
    finalization_blocker,
    parse_action,
    public_test_passed,
    select_case,
    summarize_request_metrics,
)
from coding_agent_tools import ToolError


def test_parse_action_accepts_plain_and_fenced_json() -> None:
    assert parse_action('{"action":"public_test"}') == {"action": "public_test"}
    assert parse_action('```json\n{"action":"final","summary":"done"}\n```')["action"] == "final"


def test_parse_action_rejects_non_action() -> None:
    with pytest.raises(ValueError):
        parse_action('{"summary":"missing"}')


def test_execute_action_rejects_unknown_tool() -> None:
    with pytest.raises(ToolError):
        execute_action(None, {"action": "shell"})  # type: ignore[arg-type]


def test_public_test_passed_requires_structured_zero_returncode() -> None:
    assert public_test_passed({"content": '{"returncode": 0}'})
    assert not public_test_passed({"content": '{"returncode": 1}'})
    assert not public_test_passed({"content": "build failed"})


def test_finalization_requires_patch_and_passing_public_test() -> None:
    assert finalization_blocker(
        patch_applied=False, passing_public_test=False
    )
    assert finalization_blocker(
        patch_applied=True, passing_public_test=False
    )
    assert (
        finalization_blocker(
            patch_applied=True, passing_public_test=True
        )
        is None
    )


def test_select_case_supports_oracle_task_id() -> None:
    cases = [
        {"case_id": "case-a", "oracle_task_id": "task-a"},
        {"case_id": "case-b", "oracle_task_id": "task-b"},
    ]

    assert select_case(cases, case_id=None, task_id="task-b") == cases[1]
    assert select_case(cases, case_id="case-a", task_id=None) == cases[0]


def test_select_case_requires_one_unique_selector() -> None:
    cases = [{"case_id": "case-a", "oracle_task_id": "task-a"}]

    with pytest.raises(ValueError):
        select_case(cases, case_id=None, task_id=None)
    with pytest.raises(ValueError):
        select_case(cases, case_id="case-a", task_id="task-a")
    with pytest.raises(ValueError):
        select_case(cases, case_id=None, task_id="missing")


def test_summarize_request_metrics() -> None:
    summary = summarize_request_metrics(
        [
            {
                "latency_ms": 100.0,
                "ttft_ms": 20.0,
                "input_tokens": 100,
                "output_tokens": 10,
            },
            {
                "latency_ms": 200.0,
                "ttft_ms": 40.0,
                "input_tokens": 200,
                "output_tokens": 20,
            },
        ]
    )

    assert summary == {
        "request_count": 2,
        "input_tokens": 300,
        "output_tokens": 30,
        "request_latency_mean_ms": 150.0,
        "ttft_mean_ms": 30.0,
    }
