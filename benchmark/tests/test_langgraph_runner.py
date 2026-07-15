from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import langgraph_runner
from langgraph_runner import (
    CACHEBLEND_SEPARATOR,
    LocalRAG,
    build_graph,
    compact_rag_results,
    expand_branch_roles,
    format_rag_results,
    summarize_rag_reuse,
)


class FakeRuntime:
    def __init__(self) -> None:
        self.calls = []
        self.tools = []
        self.rag_queries = []
        self.rag = SimpleNamespace(
            search=self._search,
        )
        self.recorder = SimpleNamespace(
            started=0.0,
            add=self._record,
        )

    async def _record(self, event):
        return None

    def _search(self, query, top_k):
        self.rag_queries.append((query, top_k))
        return [{"source": "evidence.md", "text": "shared evidence"}]

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["stage"] == "tool_select":
            name = kwargs["tool_choice"]["function"]["name"]
            tool_call = SimpleNamespace(
                id=f"call-{len(self.calls)}",
                function=SimpleNamespace(
                    name=name,
                    arguments=json.dumps({"query": "prefix attention", "top_k": 1}),
                ),
            )
            return SimpleNamespace(content="", tool_calls=[tool_call])
        return SimpleNamespace(content=f"{kwargs['stage']} answer", tool_calls=[])

    async def run_tool(
        self,
        case_id,
        branch_id,
        name,
        arguments,
        delay_ms=0,
        known_results=None,
    ):
        self.tools.append((case_id, branch_id, name, arguments, known_results))
        return "[]"


def test_local_rag_uses_real_files_and_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(
        "ForkAttention shares prefix KV pages between parallel branches.",
        encoding="utf-8",
    )
    (tmp_path / "b.md").write_text(
        "An unrelated document about storage.", encoding="utf-8"
    )
    rag = LocalRAG(tmp_path)

    first = rag.search("ForkAttention parallel prefix", top_k=1)
    second = rag.search("ForkAttention parallel prefix", top_k=1)

    assert first == second
    assert first[0]["source"] == "a.md"
    assert first[0]["chunk_id"].startswith("a.md:0:")


def test_local_rag_manifest_freezes_corpus_membership(tmp_path: Path) -> None:
    (tmp_path / "included.md").write_text("included evidence", encoding="utf-8")
    (tmp_path / "new-result.md").write_text("must not leak", encoding="utf-8")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("included.md\n", encoding="utf-8")

    rag = LocalRAG(tmp_path, manifest=manifest)

    assert [item["source"] for item in rag.chunks] == ["included.md"]

    original_version = rag.corpus_version
    (tmp_path / "included.md").write_text("revised evidence", encoding="utf-8")
    revised = LocalRAG(tmp_path, manifest=manifest)
    assert revised.corpus_version != original_version


def test_cacheblend_format_keeps_document_segments_stable() -> None:
    first = {
        "chunk_id": "a.md:0:v1",
        "source": "a.md",
        "score": 9.5,
        "text": "stable document A",
    }
    second = {
        "chunk_id": "b.md:0:v1",
        "source": "b.md",
        "score": 4.2,
        "text": "stable document B",
    }

    ordered = format_rag_results([first, second], rag_format="cacheblend")
    reordered = format_rag_results([second, first], rag_format="cacheblend")

    assert "9.5" not in ordered
    assert ordered.count(CACHEBLEND_SEPARATOR) == 3
    ordered_segments = {
        segment.strip()
        for segment in ordered.split(CACHEBLEND_SEPARATOR)
        if segment.strip()
    }
    reordered_segments = {
        segment.strip()
        for segment in reordered.split(CACHEBLEND_SEPARATOR)
        if segment.strip()
    }
    assert ordered_segments == reordered_segments


@pytest.mark.parametrize("rag_format", ["plain", "cacheblend"])
def test_rag_compaction_removes_only_exact_bootstrap_chunks(rag_format: str) -> None:
    known = {
        "chunk_id": "a.md:0:v1",
        "source": "a.md",
        "score": 9.5,
        "text": "stable document A",
    }
    new = {
        "chunk_id": "b.md:0:v1",
        "source": "b.md",
        "score": 4.2,
        "text": "stable document B",
    }

    result = compact_rag_results(
        [known, new], known_results=[known], rag_format=rag_format
    )

    assert "stable document A" not in result.text
    assert "stable document B" in result.text
    assert result.report.removed_duplicate_sections == 1
    assert result.report.output_sections == 1


def test_rag_reuse_summary_distinguishes_reuse_and_reordering() -> None:
    def event(case_id: str, chunks: list[tuple[str, str]]) -> dict:
        return {
            "kind": "tool",
            "stage": "bootstrap_rag",
            "case_id": case_id,
            "result": [
                {"chunk_id": chunk_id, "source": chunk_id, "text": text}
                for chunk_id, text in chunks
            ],
        }

    summary = summarize_rag_reuse(
        [
            event("a", [("one", "a" * 10), ("two", "b" * 10)]),
            event("b", [("two", "b" * 10), ("one", "a" * 10)]),
            event("c", [("three", "c" * 10)]),
        ]
    )

    assert summary["unique_chunks"] == 3
    assert summary["repeated_chunk_occurrences"] == 2
    assert summary["reusable_chars"] == 20
    assert summary["reordered_pairs"] == 1


def test_langgraph_runs_parallel_rag_tool_branches() -> None:
    runtime = FakeRuntime()
    graph = build_graph(
        runtime,
        branches=4,
        token_limits={"planner": 8, "tool_select": 8, "reflect": 8, "reduce": 8},
    )

    output = asyncio.run(
        graph.ainvoke(
            {
                "case_id": "case-0",
                "context_query": "shared Agentrix corpus query",
                "task": "research",
                "branch_outputs": [],
            }
        )
    )

    assert [item["tool"] for item in output["branch_outputs"]] == [
        "rag_search",
        "rag_search",
        "rag_search",
        "rag_search",
    ]
    assert len(runtime.tools) == 4
    assert all(tool[-1] == output["bootstrap_results"] for tool in runtime.tools)
    assert output["answer"] == "reduce answer"
    assert output["bootstrap_evidence"].startswith("[Local source: evidence.md]")
    assert runtime.rag_queries == [("shared Agentrix corpus query", 12)]
    planner = next(call for call in runtime.calls if call["stage"] == "planner")
    assert planner["messages"][1]["content"].startswith("Retrieved local evidence:")
    assert planner["messages"][2]["content"] == "research"
    selections = [call for call in runtime.calls if call["stage"] == "tool_select"]
    assert all(
        call["messages"][: len(planner["messages"])] == planner["messages"]
        for call in selections
    )
    assert all(
        call["messages"][len(planner["messages"])]["role"] == "assistant"
        for call in selections
    )
    assert [call["stage"] for call in runtime.calls].count("tool_select") == 4
    assert [call["stage"] for call in runtime.calls].count("branch_reflect") == 4


def test_langgraph_assigns_distinct_research_roles() -> None:
    runtime = FakeRuntime()
    roles = ["kernel evidence", "scheduler evidence"]
    graph = build_graph(
        runtime,
        branches=2,
        branch_roles=roles,
        token_limits={"planner": 8, "tool_select": 8, "reflect": 8, "reduce": 8},
    )

    output = asyncio.run(
        graph.ainvoke({"case_id": "case-0", "task": "research", "branch_outputs": []})
    )

    assert {item["branch_role"] for item in output["branch_outputs"]} == set(roles)
    selection_prompts = [
        call["messages"][-1]["content"]
        for call in runtime.calls
        if call["stage"] == "tool_select"
    ]
    assert any("kernel evidence" in prompt for prompt in selection_prompts)
    assert any("scheduler evidence" in prompt for prompt in selection_prompts)


def test_branch_roles_expand_to_a_full_verification_cohort() -> None:
    roles = expand_branch_roles(4, ["kernel", "scheduler"])

    assert roles == [
        "kernel",
        "scheduler",
        "independent verification: kernel",
        "independent verification: scheduler",
    ]


def test_dependency_replay_preserves_requests_and_stage_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = []

    class FakeCompletions:
        async def create(self, **request):
            requests.append(request)
            content = request["messages"][0]["content"]
            if content == "case-0:planner":
                await asyncio.sleep(0.01)
            usage = SimpleNamespace(model_dump=lambda: {"total_tokens": 1})
            return SimpleNamespace(usage=usage)

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    source = {
        "events": [
            {
                "kind": "llm",
                "case_id": case_id,
                "stage": stage,
                "started_ms": index * 10,
                "request": {
                    "model": "captured-model",
                    "messages": [{"role": "user", "content": f"{case_id}:{stage}"}],
                    "temperature": 0,
                },
            }
            for index, (case_id, stage) in enumerate(
                (case_id, stage)
                for case_id in ("case-0", "case-1")
                for stage in ("planner", "tool_select", "branch_reflect", "reduce")
            )
        ],
    }
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps(source), encoding="utf-8")
    monkeypatch.setattr(langgraph_runner, "AsyncOpenAI", FakeAsyncOpenAI)
    args = SimpleNamespace(
        trace=trace,
        api_key="local",
        base_url="http://localhost/v1",
        concurrency=4,
        model="qwen3",
        timing="dependency",
    )

    payload = asyncio.run(langgraph_runner.replay_trace(args))

    request_order = [request["messages"][0]["content"] for request in requests]
    for case_id in ("case-0", "case-1"):
        assert [item for item in request_order if item.startswith(case_id)] == [
            f"{case_id}:planner",
            f"{case_id}:tool_select",
            f"{case_id}:branch_reflect",
            f"{case_id}:reduce",
        ]
    assert request_order.index("case-1:tool_select") < request_order.index(
        "case-0:tool_select"
    )
    assert {request["model"] for request in requests} == {"qwen3"}
    assert payload["metadata"]["requests"] == 8


def test_agent_replay_preserves_branch_dependencies_and_allows_disorder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed: list[str] = []

    class FakeCompletions:
        async def create(self, **request):
            label = request["messages"][0]["content"]
            await asyncio.sleep(0.002 if "branch-0" in label else 0)
            completed.append(label)
            usage = SimpleNamespace(model_dump=lambda: {"total_tokens": 1})
            return SimpleNamespace(usage=usage)

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    events = []
    for case_id, start_ms in (("case-0", 0), ("case-1", 1)):
        events.append(
            {
                "kind": "llm",
                "case_id": case_id,
                "stage": "planner",
                "branch_id": None,
                "started_ms": start_ms,
                "request": {
                    "model": "source",
                    "messages": [{"content": f"{case_id}:planner"}],
                },
            }
        )
        for branch_id in (0, 1):
            for stage in ("tool_select", "branch_reflect"):
                events.append(
                    {
                        "kind": "llm",
                        "case_id": case_id,
                        "stage": stage,
                        "branch_id": branch_id,
                        "started_ms": start_ms + 2,
                        "request": {
                            "model": "source",
                            "messages": [
                                {"content": f"{case_id}:branch-{branch_id}:{stage}"}
                            ],
                        },
                    }
                )
            events.append(
                {
                    "kind": "tool",
                    "case_id": case_id,
                    "stage": "tool",
                    "branch_id": branch_id,
                    "latency_ms": 3 if branch_id == 0 else 0,
                }
            )
        events.append(
            {
                "kind": "llm",
                "case_id": case_id,
                "stage": "reduce",
                "branch_id": None,
                "started_ms": start_ms + 10,
                "request": {
                    "model": "source",
                    "messages": [{"content": f"{case_id}:reduce"}],
                },
            }
        )
    trace = tmp_path / "agent-trace.json"
    trace.write_text(json.dumps({"events": events}), encoding="utf-8")
    monkeypatch.setattr(langgraph_runner, "AsyncOpenAI", FakeAsyncOpenAI)
    args = SimpleNamespace(
        trace=trace,
        api_key="local",
        base_url="http://localhost/v1",
        concurrency=8,
        model="qwen3",
        timing="agent",
    )

    payload = asyncio.run(langgraph_runner.replay_trace(args))

    for case_id in ("case-0", "case-1"):
        planner = completed.index(f"{case_id}:planner")
        reducer = completed.index(f"{case_id}:reduce")
        for branch_id in (0, 1):
            select = completed.index(f"{case_id}:branch-{branch_id}:tool_select")
            reflect = completed.index(f"{case_id}:branch-{branch_id}:branch_reflect")
            assert planner < select < reflect < reducer
    assert payload["metadata"]["requests"] == 12
