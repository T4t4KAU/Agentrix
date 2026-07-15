from __future__ import annotations

import json

import pytest

from agentrix_application import (
    PromptSection,
    compact_json,
    compact_prompt_delta,
    compact_prompt_sections,
    deduplicate_tools,
)


def test_compactor_removes_only_empty_and_same_id_exact_duplicates() -> None:
    result = compact_prompt_sections(
        [
            PromptSection("policy", "keep exact whitespace  ", "Policy:"),
            PromptSection("empty", "  ", "Empty:"),
            PromptSection("policy", "keep exact whitespace  ", "Policy:"),
            PromptSection("second-policy", "keep exact whitespace  ", "Policy:"),
        ]
    )

    assert result.text.count("keep exact whitespace  ") == 2
    assert "Empty:" not in result.text
    assert result.report.removed_empty_sections == 1
    assert result.report.removed_duplicate_sections == 1
    assert result.report.saved_chars > 0


def test_delta_removes_exact_section_already_in_context() -> None:
    known = [PromptSection("rag:a", "same body", "Document a")]
    result = compact_prompt_delta(
        [
            PromptSection("rag:a", "same body", "Document a"),
            PromptSection("rag:b", "new body", "Document b"),
        ],
        known_sections=known,
    )

    assert result.text == "Document b\nnew body"
    assert result.report.removed_duplicate_sections == 1


def test_compactor_rejects_conflicting_segment_content() -> None:
    with pytest.raises(ValueError, match="conflicting content"):
        compact_prompt_delta(
            [PromptSection("policy", "second")],
            known_sections=[PromptSection("policy", "first")],
        )


def test_compact_json_is_deterministic_and_information_preserving() -> None:
    value = {"b": [1, 2], "a": "值"}
    compacted = compact_json(value)

    assert compacted == '{"a":"值","b":[1,2]}'
    assert json.loads(compacted) == value


def test_tool_deduplication_is_exact_and_conflicts_fail_closed() -> None:
    tool = {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search documents.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    assert deduplicate_tools([tool, dict(tool)]) == [tool]

    conflicting = {
        **tool,
        "function": {**tool["function"], "description": "Different meaning."},
    }
    with pytest.raises(ValueError, match="conflicting schemas"):
        deduplicate_tools([tool, conflicting])
