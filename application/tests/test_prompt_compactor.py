from __future__ import annotations

import json
import random
import hashlib
import sqlite3

import pytest

from agentrix_application import (
    ToolResultBackingStore,
    PagedToolStore,
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


def test_disk_pages_preserve_unicode_and_search_across_page_boundaries(tmp_path):
    store = PagedToolStore(tmp_path / "results.sqlite")
    store.open_session("root")
    content = "中" * 4093 + "cross-page-needle" + "尾" * 9000
    digest = store.put("root", content)
    result = store.search("root", digest, "cross-page-needle", limit=300)
    assert result["match_offset"] == 4093
    assert "cross-page-needle" in result["content"]
    assert (
        store.read("root", digest, offset=4080, limit=6000)["content"]
        == content[4080:10080]
    )
    assert store.read("root", digest, offset=len(content))["eof"]
    assert store.search("root", digest, "missing")["match_offset"] is None
    store.close()


def test_branch_snapshots_share_storage_and_reclaim_only_last_owner(tmp_path):
    path = tmp_path / "results.sqlite"
    store = PagedToolStore(path)
    store.open_session("parent")
    original = "original snapshot\n" * 10000
    digest = store.put("parent", original)
    original_size = store.stats()["stored_bytes"]
    for child in ("left", "right"):
        store.open_session(child, parent="parent")
        assert store.put(child, original) == digest
    assert store.stats()["stored_bytes"] == original_size
    changed = store.put("left", "modified snapshot")
    assert store.read("right", digest)["content"] == original[:4096]
    with pytest.raises(KeyError):
        store.read("right", changed)
    occupied = path.stat().st_size
    store.release_session("parent")
    store.release_session("left")
    assert store.stats()["stored_bytes"] == original_size
    assert store.read("right", digest)["content"] == original[:4096]
    store.release_session("right")
    assert store.stats() == dict(objects=0, stored_bytes=0, sessions=0, references=0)
    assert path.stat().st_size < occupied
    store.close()


def test_store_quota_is_atomic_and_sessions_are_isolated_after_reopen(tmp_path):
    path = tmp_path / "results.sqlite"
    store = PagedToolStore(path, max_bytes=16)
    store.open_session("a")
    digest = store.put("a", "0123456789")
    store.open_session("b")
    with pytest.raises(ValueError, match="budget"):
        store.put("b", "another ten bytes")
    with pytest.raises(KeyError):
        store.read("b", digest)
    assert store.stats()["objects"] == 1
    store.close()
    store = PagedToolStore(path, max_bytes=16)
    assert store.read("a", digest)["content"] == "0123456789"
    with pytest.raises(ValueError):
        store.read("a", digest, limit=1000000)
    with pytest.raises(KeyError):
        store.open_session("orphan", parent="absent")
    assert store.stats()["sessions"] == 2
    store.release_session("a")
    assert store.put("b", "another value")
    store.close()


def test_changed_branch_shares_pages_and_quota_counts_only_new_data(tmp_path):
    store = PagedToolStore(tmp_path / "cow.sqlite", max_bytes=3 * 4096)
    store.open_session("parent")
    original = "a" * 4096 + "b" * 4096
    handle = store.put("parent", original)
    store.open_session("left", parent="parent")
    store.open_session("right", parent="parent")
    changed = store.replace_range(
        "left", handle, offset=11, delete_chars=3, content="NEW"
    )
    expected = original[:11] + "NEW" + original[14:]
    assert changed == hashlib.sha256(expected.encode()).hexdigest()
    assert store.read("left", changed, limit=16384)["content"] == expected
    assert store.stats()["stored_bytes"] == 3 * 4096
    assert store.put("left", expected) == changed  # No extra quota charge.
    with pytest.raises(KeyError):
        store.read("right", changed)
    before = store.stats()
    # Both changed pages would be new; every intermediate write must roll back.
    store.max_bytes += 4096  # The first new page fits; the second must fail.
    with pytest.raises(ValueError, match="budget"):
        store.replace_range("right", handle, offset=4095, delete_chars=2, content="XX")
    assert store.stats() == before
    assert store.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 3
    store.release_session("parent")
    store.release_session("left")
    assert store.stats()["stored_bytes"] == 2 * 4096
    assert store.read("right", handle, limit=16384)["content"] == original
    store.release_session("right")
    assert store.stats()["stored_bytes"] == 0
    assert store.db.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 0
    assert store.db.execute("PRAGMA foreign_key_check").fetchall() == []
    store.close()


def test_streamed_snapshots_preserve_unicode_offsets_and_roll_back_on_error(tmp_path):
    store = PagedToolStore(tmp_path / "stream.sqlite")
    store.open_session("agent")
    original = "中😀" * 5000 + "tail"
    handle = store.put_stream(
        "agent", (original[i : i + 13] for i in range(0, len(original), 13))
    )
    assert handle == hashlib.sha256(original.encode()).hexdigest()
    assert store.put("agent", original) == handle
    for offset, deleted, replacement in [
        (4095, 5, "new"),
        (0, 0, "头"),
        (len(original), 0, "尾"),
        (0, len(original), ""),
    ]:
        revised = store.replace_range(
            "agent", handle, offset=offset, delete_chars=deleted, content=replacement
        )
        expected = original[:offset] + replacement + original[offset + deleted :]
        assert revised == hashlib.sha256(expected.encode()).hexdigest()
        assert store.read("agent", revised, limit=16384)["content"] == expected
    before = store.stats()

    def interrupted():
        yield "private" * 1024
        raise RuntimeError("producer failed")

    with pytest.raises(RuntimeError, match="producer failed"):
        store.put_stream("agent", interrupted())
    assert store.stats() == before
    assert (
        store.db.execute(
            "SELECT COUNT(*) FROM chunks WHERE NOT EXISTS "
            "(SELECT 1 FROM pages WHERE pages.chunk_id=chunks.id)"
        ).fetchone()[0]
        == 0
    )
    store.release_session("agent")
    assert store.stats()["stored_bytes"] == 0
    store.close()


def test_existing_private_page_database_migrates_without_changing_handles(tmp_path):
    path = tmp_path / "old.sqlite"
    content = "shared page".ljust(4096) * 3
    handle = hashlib.sha256(content.encode()).hexdigest()
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE sessions (id TEXT PRIMARY KEY);
            CREATE TABLE objects (id TEXT PRIMARY KEY, chars INTEGER, bytes INTEGER);
            CREATE TABLE pages (object_id TEXT REFERENCES objects(id) ON DELETE CASCADE,
                number INTEGER, content TEXT, PRIMARY KEY(object_id, number));
            CREATE TABLE refs (session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
                object_id TEXT REFERENCES objects(id), PRIMARY KEY(session_id, object_id));
        """)
        db.execute("INSERT INTO sessions VALUES ('parent')")
        db.execute(
            "INSERT INTO objects VALUES (?, ?, ?)", (handle, len(content), len(content))
        )
        db.execute("INSERT INTO refs VALUES ('parent', ?)", (handle,))
        db.executemany(
            "INSERT INTO pages VALUES (?, ?, ?)",
            [(handle, i, content[i * 4096 : (i + 1) * 4096]) for i in range(3)],
        )
    store = PagedToolStore(path)
    assert store.stats()["stored_bytes"] == 4096
    assert store.read("parent", handle, limit=16384)["content"] == content
    store.open_session("child", parent="parent")
    store.release_session("parent")
    assert (
        store.read("child", handle, offset=4090, limit=20)["content"]
        == content[4090:4110]
    )
    store.close()
    store = PagedToolStore(path)
    assert store.stats()["objects"] == 1
    store.release_session("child")
    assert store.stats()["stored_bytes"] == 0
    store.close()


def test_checkpoint_reclaims_only_results_without_live_branch_owners(tmp_path):
    store = PagedToolStore(tmp_path / "stages.sqlite")
    store.open_session("workflow")
    shared = store.put("workflow", "shared history")
    obsolete = store.put("workflow", "obsolete intermediate" * 4096)
    store.open_session("reviewer", parent="workflow")
    next_state = store.put("workflow", "next stage state")
    result = store.checkpoint_session("workflow", keep_results=[shared, next_state])
    assert result == dict(
        released_references=1, retained_references=2, reclaimed_bytes=0
    )
    with pytest.raises(KeyError):
        store.read("workflow", obsolete)
    assert store.read("reviewer", obsolete)["content"].startswith(
        "obsolete intermediate"
    )
    assert store.read("workflow", next_state)["content"] == "next stage state"
    with pytest.raises(KeyError):
        store.read("reviewer", next_state)
    occupied = store.stats()["stored_bytes"]
    store.release_session("reviewer")
    assert store.stats()["stored_bytes"] < occupied
    assert store.stats()["objects"] == 2
    assert (
        store.checkpoint_session("workflow", keep_results=[shared, next_state])[
            "released_references"
        ]
        == 0
    )
    store.release_session("workflow")
    assert store.stats()["stored_bytes"] == 0
    store.close()


def test_checkpoint_is_atomic_and_catalog_ids_survive_reclamation_and_reopen(tmp_path):
    path = tmp_path / "checkpoint.sqlite"
    store = PagedToolStore(path)
    store.open_session("agent")
    store.open_session("other")
    foreign = store.put("other", "private data")
    for index in range(3):
        assert store.record_observation("agent", "read", f"body {index}") == index
    catalog = store.list_observations("agent")
    before = store.stats()
    with pytest.raises(KeyError, match="not owned"):
        store.checkpoint_session(
            "agent", keep_results=[catalog[-1]["result_id"], foreign]
        )
    assert store.stats() == before
    assert store.list_observations("agent") == catalog
    reclaimed = store.checkpoint_session("agent", keep_results=[])
    assert reclaimed["released_references"] == 3
    assert reclaimed["reclaimed_bytes"] > 0
    assert store.list_observations("agent") == []
    store.close()
    store = PagedToolStore(path)
    store.open_session("child", parent="agent")
    assert store.record_observation("agent", "new", "new parent data") == 3
    assert store.record_observation("child", "new", "new child data") == 3
    assert store.list_observations("agent", offset=3)[0]["sequence"] == 3
    assert store.read("other", foreign)["content"] == "private data"
    for session in ("agent", "child", "other"):
        store.release_session(session)
    assert store.db.execute("PRAGMA foreign_key_check").fetchall() == []
    assert store.stats()["stored_bytes"] == 0
    store.close()


def test_checkpoint_migrates_existing_observation_sequence_counter(tmp_path):
    path = tmp_path / "sequence.sqlite"
    store = PagedToolStore(path)
    store.open_session("agent")
    for index in range(4):
        store.record_observation("agent", "old", f"old body {index}")
    # Recreate the previous schema: sequence was derived from surviving rows.
    store.db.execute("ALTER TABLE sessions DROP COLUMN next_sequence")
    store.db.commit()
    store.close()
    store = PagedToolStore(path)
    store.checkpoint_session("agent", keep_results=[])
    assert store.record_observation("agent", "new", "new body") == 4
    store.release_session("agent")
    store.close()


def test_tool_context_budget_bounds_long_history_and_allows_cold_recovery(tmp_path):
    store = PagedToolStore(tmp_path / "context.sqlite")
    store.open_session("agent")

    def count(text):
        return len(text.encode("utf-8"))

    originals = []
    for step in range(120):
        body = f"step {step}: " + "原始 evidence\n" * 120
        originals.append(body)
        store.record_observation("agent", f"read:{step}", body)
        context = store.render_context("agent", count_tokens=count, max_tokens=4096)
        assert context["tokens"] == count(context["text"]) <= 4096
        recent = json.loads(context["text"])["recent_results"]
        assert recent[-1]["sequence"] == step
        assert recent[-1]["content"] == body[: recent[-1]["next_offset"]]
    assert context["archived_observations"] > 100
    # Old results remain paginated, owned and byte-exact after prompt eviction.
    catalog = store.list_observations("agent", offset=0, limit=1)
    first = store.read("agent", catalog[0]["result_id"])["content"]
    assert first == originals[0]
    store.record_observation("agent", "revisit:first", first)
    context = store.render_context("agent", count_tokens=count, max_tokens=4096)
    assert json.loads(context["text"])["recent_results"][-1]["content"] == first
    assert store.stats()["objects"] == 120  # Retrieval does not duplicate storage.
    store.release_session("agent")
    assert store.stats() == dict(objects=0, stored_bytes=0, sessions=0, references=0)
    assert store.db.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0
    store.close()


def test_tool_context_forks_share_archive_but_own_their_new_observations(tmp_path):
    store = PagedToolStore(tmp_path / "branches.sqlite")
    store.open_session("parent")
    store.record_observation("parent", "shared", "shared content")
    store.open_session("left", parent="parent")
    store.open_session("right", parent="parent")
    assert store.stats()["objects"] == 1
    store.record_observation("left", "private", "left-only content")
    private = store.list_observations("left", offset=1)[0]
    assert len(store.list_observations("right")) == 1
    with pytest.raises(KeyError):
        store.read("right", private["result_id"])
    store.release_session("parent")
    store.release_session("left")
    assert store.stats()["objects"] == 1
    assert store.list_observations("right")[0]["label"] == "shared"
    store.release_session("right")
    assert store.stats()["stored_bytes"] == 0
    store.close()


def test_tool_context_oversized_observation_is_explicitly_partial(tmp_path):
    path = tmp_path / "partial.sqlite"
    store = PagedToolStore(path)
    store.open_session("agent")
    original = "零😀abcdef" * 10000
    store.record_observation("agent", "large", original)
    context = store.render_context("agent", count_tokens=len, max_tokens=600)
    result = json.loads(context["text"])["recent_results"][0]
    assert context["tokens"] <= 600
    assert result["partial"] and context["partial_observations"] == 1
    assert result["content"] == original[: result["next_offset"]]
    assert result["total_chars"] == len(original)
    store.close()
    store = PagedToolStore(path)
    assert store.list_observations("agent")[0]["result_id"] == result["result_id"]
    remainder = store.read("agent", result["result_id"], offset=result["next_offset"])
    assert (
        remainder["content"]
        == original[result["next_offset"] : result["next_offset"] + 4096]
    )
    unbounded = store.render_context("agent", count_tokens=len, max_tokens=None)
    assert json.loads(unbounded["text"])["recent_results"][0]["content"] == original
    with pytest.raises(ValueError, match="budget"):
        store.render_context("agent", count_tokens=len, max_tokens=1)
    with pytest.raises(ValueError):
        store.list_observations("agent", limit=100000)
    store.release_session("agent")
    store.close()


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
