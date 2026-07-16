import pytest

from coding_agent_e2e_runner import execute_action, parse_action
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
