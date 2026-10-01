import hashlib
import json
import runpy
import tracemalloc
from pathlib import Path

import pytest
from agentrix_application import PagedToolStore
from coding_agent_tools import RepositoryTools


def test_public_test_expands_python_in_build_command(tmp_path: Path) -> None:
    agentrix = tmp_path / ".agentrix"
    agentrix.mkdir()
    public_test = agentrix / "public_test.py"
    public_test.write_text("raise SystemExit(0)\n", encoding="utf-8")
    task = {
        "allowed_paths": ["target.py"],
        "build": [
            {
                "cwd": ".",
                "argv": [
                    "{python}",
                    "-c",
                    "from pathlib import Path; Path('built').touch()",
                ],
            }
        ],
        "public_test_command": ["{python}", ".agentrix/public_test.py"],
        "timeout_seconds": 10,
    }

    event = RepositoryTools(tmp_path, task).public_test()

    assert '"returncode": 0' in event["content"]
    assert (tmp_path / "built").is_file()


def test_paged_read_retrieves_original_snapshot_after_source_changes(tmp_path):
    store = PagedToolStore(tmp_path / "snapshots.sqlite")
    store.open_session("root")
    source = tmp_path / "source.txt"
    source.write_text("original line\n" * 300)
    tools = RepositoryTools(tmp_path, {}, max_output_bytes=512, result_store=store)
    event = tools.read("source.txt")
    assert event["paged"] and not event["truncated"]
    handle = json.loads(event["content"])["result_id"]
    source.write_text("changed source")
    # The old path/handle remain discoverable after the live file changes.
    catalog = json.loads(tools.list_results(limit=1)["content"])
    assert catalog["results"][0]["result_id"] == handle
    assert catalog["results"][0]["arguments"]["path"] == "source.txt"
    assert catalog["total"] == catalog["next_offset"] == 1
    assert json.loads(tools.list_results(offset=1)["content"])["results"] == []
    page = tools.search_result(handle, "200: original line")
    assert "200: original line" in json.loads(page["content"])["content"]
    assert not page["truncated"]
    store.release_session("root")
    assert store.stats()["stored_bytes"] == 0
    store.close()


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\n",
        b"one\n\n",
        b"one\r\ntwo\rthree\nlast",
        "中\v文\f😀\x1cnext\x1dmore\x1eend\x85ls\u2028ps\u2029tail".encode(),
        b"invalid \xff\xfe\nnext",
        b"x" * 65535 + b"\r\nsecond\nthird",
        b"x" * 65535 + "中\u2028tail".encode(),
    ],
)
def test_streamed_file_read_preserves_existing_text_and_line_numbers(tmp_path, data):
    source = tmp_path / "input.txt"
    source.write_bytes(data)
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    for start, end in [(1, 10), (2, 2), (5, 10), (100, 110)]:
        event = RepositoryTools(tmp_path, {}, max_output_bytes=1 << 20).read(
            "input.txt", start, end
        )
        expected = f"File input.txt has {len(lines)} lines.\n" + "\n".join(
            f"{index}: {line}"
            for index, line in enumerate(lines[start - 1 : end], start)
        )
        assert event["content"] == expected
        assert event["original_bytes"] == len(expected.encode())


def test_narrow_read_does_not_materialize_an_unselected_large_line(tmp_path):
    source = tmp_path / "large.log"
    with source.open("wb") as stream:
        stream.write(b"wanted\n")
        for _ in range(128):
            stream.write(b"x" * (64 << 10))
        stream.write(b"\nlast\n")
    tracemalloc.start()
    try:
        event = RepositoryTools(tmp_path, {}).read("large.log", 1, 1)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert event["content"] == "File large.log has 3 lines.\n1: wanted"
    assert peak < 1 << 20


@pytest.mark.parametrize("paging", [False, True])
def test_large_selected_line_has_bounded_memory_and_exact_result(tmp_path, paging):
    source = tmp_path / "large.txt"
    line = "中😀ab" * (1 << 19)
    source.write_text(line + "\r\nlast\n", encoding="utf-8")
    expected = "File large.txt has 2 lines.\n1: " + line
    expected_hash = hashlib.sha256(expected.encode()).hexdigest()
    expected_bytes = len(expected.encode())
    store = PagedToolStore(tmp_path / "snapshots.sqlite") if paging else None
    if store is not None:
        store.open_session("root")
    tools = RepositoryTools(tmp_path, {}, max_output_bytes=511, result_store=store)
    tracemalloc.start()
    try:
        event = tools.read("large.txt", 1, 1)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 3 << 20
    assert event["content_sha256"] == expected_hash
    assert event["original_bytes"] == expected_bytes
    assert event["paged"] is paging
    assert event["truncated"] is not paging
    if store is not None:
        handle = json.loads(event["content"])
        assert handle["total_chars"] == len(expected)
        assert handle["preview"] == expected[:256]
        source.write_text("changed", encoding="utf-8")
        digest = hashlib.sha256()
        for offset in range(0, len(expected), 16384):
            digest.update(
                store.read("root", handle["result_id"], offset=offset, limit=16384)[
                    "content"
                ].encode()
            )
        assert digest.hexdigest() == expected_hash == handle["result_id"]
        store.release_session("root")
        assert store.stats()["stored_bytes"] == 0
        store.close()
    else:
        assert event["content"] == expected.encode()[:511].decode(errors="replace")
        assert (
            event["returned_sha256"]
            == hashlib.sha256(expected.encode()[:511]).hexdigest()
        )


def test_streamed_read_quota_failure_publishes_no_partial_result(tmp_path):
    store = PagedToolStore(tmp_path / "snapshots.sqlite", max_bytes=4096)
    store.open_session("root")
    (tmp_path / "large.txt").write_text("x" * 8192)
    tools = RepositoryTools(tmp_path, {}, max_output_bytes=32, result_store=store)
    with pytest.raises(ValueError, match="budget exceeded"):
        tools.read("large.txt")
    assert tools.events == []
    assert store.stats() == dict(objects=0, stored_bytes=0, sessions=1, references=0)
    store.close()


def test_paging_benchmark_handles_parallel_calls_and_checks_only_proposed_score():
    benchmark = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/benchmark_tool_result_paging.py")
    )
    actions = benchmark["parse_actions"](
        '{"action":"search","source":"builds","needle":"job-0001"}\n'
        '{"action":"search","source":"checks","needle":"job-0001"}'
    )
    assert [action["source"] for action in actions] == ["builds", "checks"]
    with pytest.raises(ValueError, match="contradicts"):
        benchmark["validate_decision"]({"score": 44, "decision": "accept"})
    # Consistency validation must not silently fix or consult the answer oracle.
    proposal = {"revision": "invented", "score": 99, "decision": "accept"}
    benchmark["validate_decision"](proposal)
    assert proposal == {"revision": "invented", "score": 99, "decision": "accept"}


def test_paging_benchmark_restores_exact_reports_with_a_total_bound(tmp_path):
    benchmark = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/benchmark_tool_result_paging.py")
    )
    store = PagedToolStore(tmp_path / "snapshots.sqlite")
    store.open_session("root")
    reports = {"builds": "原始结果\n" * 5000, "checks": ""}
    handles = {name: store.put("root", body) for name, body in reports.items()}
    store.open_session("branch", parent="root")
    store.release_session("root")
    size = sum(map(len, reports.values()))
    try:
        restore = benchmark["restore_reports"]
        assert restore(store, "branch", handles, size - 1) is None
        assert restore(store, "branch", handles, size) == "\n\n".join(
            f"Report {name}:\n{body}" for name, body in reports.items()
        )
        with pytest.raises(KeyError, match="not owned"):
            restore(store, "other", handles, size)
    finally:
        store.release_session("branch")
        store.close()
