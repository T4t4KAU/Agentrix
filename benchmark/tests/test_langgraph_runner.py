from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import langgraph_runner
from langgraph_runner import LocalRAG, build_graph


class FakeRuntime:
    def __init__(self) -> None:
        self.calls = []
        self.tools = []
        self.rag = SimpleNamespace(
            search=lambda query, top_k: [
                {"source": "evidence.md", "text": "shared evidence"}
            ]
        )
        self.recorder = SimpleNamespace(
            started=0.0,
            add=self._record,
        )

    async def _record(self, event):
        return None

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

    async def run_tool(self, case_id, branch_id, name, arguments):
        self.tools.append((case_id, branch_id, name, arguments))
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


def test_langgraph_runs_parallel_rag_tool_branches() -> None:
    runtime = FakeRuntime()
    graph = build_graph(
        runtime,
        branches=4,
        token_limits={"planner": 8, "tool_select": 8, "reflect": 8, "reduce": 8},
    )

    output = asyncio.run(
        graph.ainvoke({"case_id": "case-0", "task": "research", "branch_outputs": []})
    )

    assert [item["tool"] for item in output["branch_outputs"]] == [
        "rag_search",
        "rag_search",
        "rag_search",
        "rag_search",
    ]
    assert len(runtime.tools) == 4
    assert output["answer"] == "reduce answer"
    assert output["bootstrap_evidence"].startswith("[Local source: evidence.md]")
    assert [call["stage"] for call in runtime.calls].count("tool_select") == 4
    assert [call["stage"] for call in runtime.calls].count("branch_reflect") == 4


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
