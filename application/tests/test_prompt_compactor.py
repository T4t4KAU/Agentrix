from __future__ import annotations

import json
import random

import pytest

from agentrix_application import (
    ToolResultBackingStore,
    ToolResultCompactionConfig,
    PromptSection,
    compact_json,
    compact_prompt_delta,
    compact_prompt_sections,
    compact_tool_results,
    deduplicate_tools,
    restore_tool_results,
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


def _tool_call(
    call_id: str,
    name: str,
    arguments: dict[str, object],
) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, sort_keys=True),
                },
            }
        ],
    }


def _old_read_trace(content: str) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "Keep this byte-for-byte."},
        _tool_call("call-read", "read_file", {"path": "src/example.py"}),
        {"role": "tool", "tool_call_id": "call-read", "content": content},
        {"role": "assistant", "content": "I inspected the file."},
    ]
    for turn in range(4):
        messages.extend(
            [
                {"role": "user", "content": f"Follow-up {turn}"},
                {"role": "assistant", "content": f"Answer {turn}"},
            ]
        )
    return messages


def test_old_recoverable_tool_result_is_paged_and_exactly_restorable() -> None:
    original_content = "line = 1\n" * 600
    messages = _old_read_trace(original_content)
    original_copy = json.loads(json.dumps(messages))
    store = ToolResultBackingStore()

    result = compact_tool_results(
        messages,
        config=ToolResultCompactionConfig(enabled=True),
        backing_store=store,
    )

    assert messages == original_copy
    assert result.messages[0] == messages[0]
    assert result.messages[1] == messages[1]
    assert result.messages[2]["tool_call_id"] == "call-read"
    assert str(result.messages[2]["content"]).startswith(
        "[Agentrix paged tool result] "
    )
    assert result.messages[3:] == messages[3:]
    assert result.report.tool_results_seen == 1
    assert result.report.compacted_results == 1
    assert result.report.saved_chars > 4_000
    assert result.report.paged_results[0].resource == "src/example.py"
    assert result.report.paged_results[0].age_turns == 4
    assert len(store) == 1
    assert store.stored_chars == len(original_content)
    assert restore_tool_results(result.messages, store) == messages


def test_tool_result_compaction_is_disabled_by_default() -> None:
    messages = _old_read_trace("x = 1\n" * 1000)
    result = compact_tool_results(messages)

    assert result.messages == messages
    assert result.messages is not messages
    assert result.report.compacted_results == 0
    assert result.report.saved_chars == 0
    assert result.report.skipped_reasons == {"disabled": 1}
    assert len(result.backing_store) == 0


def test_compaction_returns_store_and_is_idempotent() -> None:
    messages = _old_read_trace("source line\n" * 500)
    config = ToolResultCompactionConfig(
        enabled=True,
        min_chars=128,
    )

    first = compact_tool_results(messages, config=config)
    second = compact_tool_results(
        first.messages,
        config=config,
        backing_store=first.backing_store,
    )

    assert restore_tool_results(first.messages, first.backing_store) == messages
    assert second.messages == first.messages
    assert second.report.compacted_results == 0
    assert second.report.skipped_reasons == {"already_paged": 1}
    assert second.backing_store is first.backing_store


def test_result_is_preserved_when_stub_would_be_larger() -> None:
    messages = _old_read_trace("short result")
    result = compact_tool_results(
        messages,
        config=ToolResultCompactionConfig(enabled=True, min_chars=1),
    )

    assert result.messages == messages
    assert result.report.saved_chars == 0
    assert result.report.skipped_reasons == {"nonpositive_savings": 1}
    assert len(result.backing_store) == 0


@pytest.mark.parametrize(
    ("name", "arguments", "content", "age_turns", "extra", "reason"),
    [
        ("search", {"path": "src/a.py"}, "x" * 5000, 4, {}, "nonrecoverable_tool"),
        ("read", {}, "x" * 5000, 4, {}, "missing_resource"),
        ("read", {"path": "src/a.py"}, "small", 4, {}, "below_min_chars"),
        ("read", {"path": "src/a.py"}, "x" * 5000, 3, {}, "too_recent"),
        (
            "read",
            {"path": "src/a.py"},
            "Tool error: permission denied\n" + "x" * 5000,
            4,
            {},
            "error_result",
        ),
        (
            "read",
            {"path": "src/a.py"},
            "x" * 5000,
            4,
            {"is_error": True},
            "error_result",
        ),
    ],
)
def test_conservative_policy_preserves_ineligible_results(
    name: str,
    arguments: dict[str, object],
    content: str,
    age_turns: int,
    extra: dict[str, object],
    reason: str,
) -> None:
    messages: list[dict[str, object]] = [
        _tool_call("call-1", name, arguments),
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": content,
            **extra,
        },
        {"role": "assistant", "content": "Consumed."},
    ]
    for turn in range(age_turns):
        messages.extend(
            [
                {"role": "user", "content": f"Question {turn}"},
                {"role": "assistant", "content": f"Answer {turn}"},
            ]
        )

    result = compact_tool_results(
        messages,
        config=ToolResultCompactionConfig(enabled=True),
    )

    assert result.messages == messages
    assert result.report.compacted_results == 0
    assert result.report.skipped_reasons == {reason: 1}


def test_structured_and_orphan_tool_results_are_preserved() -> None:
    messages = [
        {
            "role": "tool",
            "tool_call_id": "missing",
            "content": "x" * 5000,
        },
        _tool_call("structured", "read", {"path": "src/a.py"}),
        {
            "role": "tool",
            "tool_call_id": "structured",
            "content": [{"type": "text", "text": "x" * 5000}],
        },
        {"role": "assistant", "content": "Consumed."},
        *[
            message
            for turn in range(4)
            for message in (
                {"role": "user", "content": f"Question {turn}"},
                {"role": "assistant", "content": f"Answer {turn}"},
            )
        ],
    ]

    result = compact_tool_results(
        messages,
        config=ToolResultCompactionConfig(enabled=True),
    )

    assert result.messages == messages
    assert result.report.skipped_reasons == {
        "unknown_tool_call": 1,
        "structured_content": 1,
    }


def test_compaction_is_deterministic_and_keeps_distinct_file_versions() -> None:
    first = _old_read_trace("version one\n" * 500)
    second = _old_read_trace("version two\n" * 500)
    second[1]["tool_calls"][0]["id"] = "call-read-2"  # type: ignore[index]
    second[2]["tool_call_id"] = "call-read-2"
    messages = [*first, *second]
    store = ToolResultBackingStore()
    config = ToolResultCompactionConfig(enabled=True)

    one = compact_tool_results(messages, config=config, backing_store=store)
    two = compact_tool_results(messages, config=config, backing_store=store)

    assert one == two
    assert one.report.compacted_results == 2
    assert len(store) == 2
    assert restore_tool_results(one.messages, store) == messages


def test_restore_rejects_missing_or_tampered_backing_content() -> None:
    messages = _old_read_trace("content\n" * 700)
    store = ToolResultBackingStore()
    compacted = compact_tool_results(
        messages,
        config=ToolResultCompactionConfig(enabled=True),
        backing_store=store,
    )

    with pytest.raises(KeyError, match="not present"):
        restore_tool_results(compacted.messages, ToolResultBackingStore())

    tampered = json.loads(json.dumps(compacted.messages))
    metadata = json.loads(
        tampered[2]["content"].removeprefix("[Agentrix paged tool result] ")
    )
    metadata["chars"] += 1
    tampered[2]["content"] = "[Agentrix paged tool result] " + compact_json(metadata)
    with pytest.raises(ValueError, match="character count mismatch"):
        restore_tool_results(tampered, store)


def test_tool_result_config_rejects_unsafe_bounds() -> None:
    with pytest.raises(ValueError, match="min_chars"):
        ToolResultCompactionConfig(min_chars=0)
    with pytest.raises(ValueError, match="min_age_turns"):
        ToolResultCompactionConfig(min_age_turns=-1)


def test_randomized_compaction_roundtrips_without_protocol_mutation() -> None:
    rng = random.Random(20260718)
    config = ToolResultCompactionConfig(
        enabled=True,
        min_chars=128,
        min_age_turns=2,
    )

    for case_index in range(250):
        tool_name = rng.choice(("read", "read_file", "search", "public_test"))
        content = "".join(
            rng.choice("abcXYZ0123\n") for _ in range(rng.randint(32, 512))
        )
        is_error = rng.random() < 0.1
        age_turns = rng.randint(0, 6)
        messages: list[dict[str, object]] = [
            {"role": "user", "content": f"Case {case_index}"},
            _tool_call(
                f"call-{case_index}",
                tool_name,
                {"path": f"src/{case_index}.py"},
            ),
            {
                "role": "tool",
                "tool_call_id": f"call-{case_index}",
                "content": content,
                "is_error": is_error,
            },
            {"role": "assistant", "content": f"Consumed {case_index}"},
        ]
        for turn in range(age_turns):
            messages.extend(
                [
                    {"role": "user", "content": f"Question {turn}"},
                    {"role": "assistant", "content": f"Answer {turn}"},
                ]
            )
        store = ToolResultBackingStore()

        compacted = compact_tool_results(
            messages,
            config=config,
            backing_store=store,
        )

        assert restore_tool_results(compacted.messages, store) == messages
        for original, transformed in zip(messages, compacted.messages, strict=True):
            assert transformed["role"] == original["role"]
            if not str(transformed.get("content", "")).startswith(
                "[Agentrix paged tool result] "
            ):
                assert transformed == original
            else:
                assert {
                    key: value for key, value in transformed.items() if key != "content"
                } == {key: value for key, value in original.items() if key != "content"}
