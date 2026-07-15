from __future__ import annotations

import argparse
import asyncio
import json
import math
import operator
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, TypedDict

from openai import AsyncOpenAI

from data import load_records, record_to_prompt


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Search the local Agentrix documentation corpus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["query"],
            },
        },
    },
]


class OverallState(TypedDict, total=False):
    case_id: str
    task: str
    bootstrap_evidence: str
    shared_analysis: str
    branches: int
    branch_outputs: Annotated[list[dict[str, Any]], operator.add]
    answer: str


class BranchState(TypedDict):
    case_id: str
    task: str
    bootstrap_evidence: str
    shared_analysis: str
    branch_id: int
    required_tool: str


def _words(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


class LocalRAG:
    """Small deterministic BM25 index over real local text files."""

    def __init__(self, root: Path, chunk_chars: int = 1800) -> None:
        self.chunks: list[dict[str, str]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".py", ".txt"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            relative = str(path.relative_to(root))
            for offset in range(0, len(text), chunk_chars):
                body = text[offset : offset + chunk_chars].strip()
                if body:
                    self.chunks.append({"source": relative, "text": body})
        self.term_frequencies = [Counter(_words(item["text"])) for item in self.chunks]
        self.lengths = [sum(counter.values()) for counter in self.term_frequencies]
        self.avg_length = statistics.fmean(self.lengths) if self.lengths else 1.0
        self.document_frequency = Counter()
        for counter in self.term_frequencies:
            self.document_frequency.update(counter.keys())

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        query_terms = _words(query)
        total = max(len(self.chunks), 1)
        scored: list[tuple[float, int]] = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            length = self.lengths[index]
            for term in query_terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                documents = self.document_frequency[term]
                inverse = math.log(1 + (total - documents + 0.5) / (documents + 0.5))
                denominator = frequency + 1.5 * (0.25 + 0.75 * length / self.avg_length)
                score += inverse * frequency * 2.5 / denominator
            if score:
                scored.append((score, index))
        return [
            {
                "source": self.chunks[index]["source"],
                "score": round(score, 4),
                "text": self.chunks[index]["text"][:1200],
            }
            for score, index in sorted(scored, reverse=True)[: max(1, min(top_k, 32))]
        ]


class TraceRecorder:
    def __init__(self, model: str) -> None:
        self.model = model
        self.started = time.perf_counter()
        self.events: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def add(self, event: dict[str, Any]) -> None:
        async with self._lock:
            self.events.append(event)

    def payload(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model": self.model,
            "metadata": metadata,
            "events": sorted(self.events, key=lambda item: item["started_ms"]),
        }


class AgentRuntime:
    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        rag: LocalRAG,
        recorder: TraceRecorder,
        concurrency: int,
    ) -> None:
        self.client = client
        self.model = model
        self.rag = rag
        self.recorder = recorder
        self.semaphore = asyncio.Semaphore(concurrency)

    async def complete(
        self,
        *,
        case_id: str,
        stage: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
    ) -> Any:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if tools is not None:
            request["tools"] = tools
        if tool_choice is not None:
            request["tool_choice"] = tool_choice
        started_ms = (time.perf_counter() - self.recorder.started) * 1000
        started = time.perf_counter()
        async with self.semaphore:
            response = await self.client.chat.completions.create(**request)
        latency_ms = (time.perf_counter() - started) * 1000
        message = response.choices[0].message
        response_data = message.model_dump(exclude_none=True)
        usage = response.usage.model_dump() if response.usage else {}
        await self.recorder.add(
            {
                "kind": "llm",
                "case_id": case_id,
                "stage": stage,
                "started_ms": started_ms,
                "latency_ms": latency_ms,
                "request": request,
                "response": response_data,
                "usage": usage,
            }
        )
        return message

    async def run_tool(
        self, case_id: str, branch_id: int, name: str, arguments: dict[str, Any]
    ) -> str:
        query = str(arguments.get("query") or "Agentrix shared prefix attention")
        top_k = max(1, min(int(arguments.get("top_k") or 3), 5))
        started_ms = (time.perf_counter() - self.recorder.started) * 1000
        started = time.perf_counter()
        if name == "rag_search":
            result: Any = self.rag.search(query, top_k)
        else:
            result = {"error": f"unsupported tool: {name}"}
        latency_ms = (time.perf_counter() - started) * 1000
        content = json.dumps(result, ensure_ascii=False)
        await self.recorder.add(
            {
                "kind": "tool",
                "case_id": case_id,
                "branch_id": branch_id,
                "stage": "tool",
                "tool": name,
                "arguments": arguments,
                "started_ms": started_ms,
                "latency_ms": latency_ms,
                "result": result,
            }
        )
        return content


def _tool_call(message: Any, required_tool: str, fallback_query: str) -> dict[str, Any]:
    calls = list(message.tool_calls or [])
    if calls:
        call = calls[0]
        try:
            arguments = json.loads(call.function.arguments)
        except json.JSONDecodeError:
            arguments = {"query": fallback_query, "top_k": 3}
        return {"id": call.id, "name": call.function.name, "arguments": arguments}
    return {
        "id": f"fallback-{required_tool}",
        "name": required_tool,
        "arguments": {"query": fallback_query, "top_k": 3},
    }


def build_graph(
    runtime: AgentRuntime,
    branches: int,
    token_limits: dict[str, int],
    bootstrap_chunks: int = 12,
    bootstrap_max_chars: int = 32000,
):
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send

    async def retrieve_context(state: OverallState) -> dict[str, Any]:
        started_ms = (time.perf_counter() - runtime.recorder.started) * 1000
        started = time.perf_counter()
        results = runtime.rag.search(state["task"], bootstrap_chunks)
        evidence = "\n\n".join(
            f"[Local source: {item['source']}]\n{item['text']}" for item in results
        )[:bootstrap_max_chars]
        await runtime.recorder.add(
            {
                "kind": "tool",
                "case_id": state["case_id"],
                "branch_id": -1,
                "stage": "bootstrap_rag",
                "tool": "rag_search",
                "arguments": {"query": state["task"], "top_k": bootstrap_chunks},
                "started_ms": started_ms,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "result": results,
            }
        )
        return {"bootstrap_evidence": evidence}

    async def planner(state: OverallState) -> dict[str, Any]:
        message = await runtime.complete(
            case_id=state["case_id"],
            stage="planner",
            messages=[
                {
                    "role": "system",
                    "content": "You are the planner for a research agent. Identify uncertainties that need evidence from the local RAG corpus.",
                },
                {"role": "user", "content": state["task"]},
                {
                    "role": "user",
                    "content": "Retrieved local evidence:\n\n"
                    + state["bootstrap_evidence"],
                },
            ],
            max_tokens=token_limits["planner"],
        )
        return {"shared_analysis": message.content or "Investigate the task."}

    def fanout(state: OverallState) -> list[Any]:
        return [
            Send(
                "branch_agent",
                {
                    "case_id": state["case_id"],
                    "task": state["task"],
                    "bootstrap_evidence": state["bootstrap_evidence"],
                    "shared_analysis": state["shared_analysis"],
                    "branch_id": branch_id,
                    "required_tool": "rag_search",
                },
            )
            for branch_id in range(branches)
        ]

    async def branch_agent(state: BranchState) -> dict[str, Any]:
        required_tool = state["required_tool"]
        shared_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "You are one branch of a parallel research agent. Use the required tool and ground the answer in returned evidence.",
            },
            {"role": "user", "content": state["task"]},
            {
                "role": "user",
                "content": "Retrieved local evidence:\n\n"
                + state["bootstrap_evidence"],
            },
            {"role": "assistant", "content": state["shared_analysis"]},
        ]
        branch_prompt = f"Research branch {state['branch_id']}. Call {required_tool} with a focused query."
        selection_messages = [
            *shared_messages,
            {"role": "user", "content": branch_prompt},
        ]
        selected = await runtime.complete(
            case_id=state["case_id"],
            stage="tool_select",
            messages=selection_messages,
            max_tokens=token_limits["tool_select"],
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": required_tool}},
        )
        call = _tool_call(selected, required_tool, state["task"][:500])
        tool_content = await runtime.run_tool(
            state["case_id"], state["branch_id"], call["name"], call["arguments"]
        )
        assistant_call = {
            "role": "assistant",
            "content": selected.content or "",
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                    },
                }
            ],
        }
        reflection_messages = [
            *selection_messages,
            assistant_call,
            {"role": "tool", "tool_call_id": call["id"], "content": tool_content},
            {
                "role": "user",
                "content": "Synthesize the evidence and cite its source titles or local paths.",
            },
        ]
        reflected = await runtime.complete(
            case_id=state["case_id"],
            stage="branch_reflect",
            messages=reflection_messages,
            max_tokens=token_limits["reflect"],
        )
        return {
            "branch_outputs": [
                {
                    "branch_id": state["branch_id"],
                    "tool": call["name"],
                    "answer": reflected.content or "",
                }
            ]
        }

    async def reducer(state: OverallState) -> dict[str, Any]:
        evidence = "\n\n".join(
            f"Branch {item['branch_id']} ({item['tool']}):\n{item['answer']}"
            for item in sorted(
                state["branch_outputs"], key=lambda item: item["branch_id"]
            )
        )
        message = await runtime.complete(
            case_id=state["case_id"],
            stage="reduce",
            messages=[
                {
                    "role": "system",
                    "content": "You are the reducer. Produce a concise evidence-grounded final answer and flag conflicts.",
                },
                {"role": "user", "content": state["task"]},
                {
                    "role": "user",
                    "content": "Retrieved local evidence:\n\n"
                    + state["bootstrap_evidence"],
                },
                {"role": "assistant", "content": state["shared_analysis"]},
                {"role": "user", "content": evidence},
            ],
            max_tokens=token_limits["reduce"],
        )
        return {"answer": message.content or ""}

    graph = StateGraph(OverallState)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("planner", planner)
    graph.add_node("branch_agent", branch_agent)
    graph.add_node("reducer", reducer)
    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "planner")
    graph.add_conditional_edges("planner", fanout, ["branch_agent"])
    graph.add_edge("branch_agent", "reducer")
    graph.add_edge("reducer", END)
    return graph.compile()


async def run_live(args: argparse.Namespace) -> dict[str, Any]:
    if args.task_file:
        task_records = [
            json.loads(line)
            for line in args.task_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        selected_tasks = [
            (str(item.get("id", index)), str(item["task"]))
            for index, item in enumerate(
                task_records[args.sample_index : args.sample_index + args.cases],
                args.sample_index,
            )
        ]
        dataset_name = "task_file"
    else:
        records = load_records(args.dataset, args.data_path)
        selected = records[args.sample_index : args.sample_index + args.cases]
        selected_tasks = [
            (
                f"{args.dataset}-{args.sample_index + index}",
                record_to_prompt(args.dataset, record),
            )
            for index, record in enumerate(selected)
        ]
        dataset_name = args.dataset
    if len(selected_tasks) != args.cases:
        raise ValueError("requested cases exceed available task records")
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    recorder = TraceRecorder(args.model)
    rag = LocalRAG(args.rag_root)
    runtime = AgentRuntime(
        client=client,
        model=args.model,
        rag=rag,
        recorder=recorder,
        concurrency=args.concurrency,
    )
    limits = {
        "planner": args.planner_tokens,
        "tool_select": args.tool_tokens,
        "reflect": args.reflect_tokens,
        "reduce": args.reduce_tokens,
    }
    graph = build_graph(
        runtime,
        args.branches,
        limits,
        bootstrap_chunks=args.bootstrap_chunks,
        bootstrap_max_chars=args.bootstrap_max_chars,
    )
    started = time.perf_counter()
    outputs = await asyncio.gather(
        *(
            graph.ainvoke(
                {
                    "case_id": case_id,
                    "task": task,
                    "branches": args.branches,
                    "branch_outputs": [],
                }
            )
            for case_id, task in selected_tasks
        )
    )
    wall_ms = (time.perf_counter() - started) * 1000
    metadata = {
        "mode": "live",
        "dataset": dataset_name,
        "sample_index": args.sample_index,
        "cases": args.cases,
        "branches_per_case": args.branches,
        "rag_root": str(args.rag_root),
        "rag_chunks": len(rag.chunks),
        "bootstrap_chunks": args.bootstrap_chunks,
        "bootstrap_max_chars": args.bootstrap_max_chars,
        "bootstrap_chars": [len(output["bootstrap_evidence"]) for output in outputs],
        "wall_ms": wall_ms,
    }
    payload = recorder.payload(metadata)
    payload["outputs"] = [
        {"case_id": output["case_id"], "answer": output["answer"]} for output in outputs
    ]
    return payload


async def replay_trace(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.trace.read_text(encoding="utf-8"))
    llm_events = [event for event in source["events"] if event["kind"] == "llm"]
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    semaphore = asyncio.Semaphore(args.concurrency)
    replayed: list[dict[str, Any]] = []
    started = time.perf_counter()

    async def replay(event: dict[str, Any], target_ms: float | None = None) -> None:
        if target_ms is not None:
            delay = target_ms / 1000 - (time.perf_counter() - started)
            if delay > 0:
                await asyncio.sleep(delay)
        request = dict(event["request"])
        request["model"] = args.model
        request_started = time.perf_counter()
        async with semaphore:
            response = await client.chat.completions.create(**request)
        replayed.append(
            {
                "case_id": event["case_id"],
                "stage": event["stage"],
                "source_started_ms": event["started_ms"],
                "started_ms": (request_started - started) * 1000,
                "latency_ms": (time.perf_counter() - request_started) * 1000,
                "usage": response.usage.model_dump() if response.usage else {},
            }
        )

    if args.timing == "captured":
        await asyncio.gather(
            *(replay(event, event["started_ms"]) for event in llm_events)
        )
    else:
        stage_order = ("planner", "tool_select", "branch_reflect", "reduce")
        unexpected_stages = {
            event["stage"] for event in llm_events if event["stage"] not in stage_order
        }
        if unexpected_stages:
            raise ValueError(
                "dependency replay does not support stages: "
                + ", ".join(sorted(unexpected_stages))
            )
        case_events: dict[str, list[dict[str, Any]]] = {}
        for event in llm_events:
            case_events.setdefault(event["case_id"], []).append(event)

        async def replay_case(events: list[dict[str, Any]]) -> None:
            for stage in stage_order:
                await asyncio.gather(
                    *(replay(event) for event in events if event["stage"] == stage)
                )

        await asyncio.gather(*(replay_case(events) for events in case_events.values()))
    wall_ms = (time.perf_counter() - started) * 1000
    return {
        "schema_version": 1,
        "model": args.model,
        "metadata": {
            "mode": "replay",
            "source_trace": str(args.trace),
            "requests": len(llm_events),
            "timing": args.timing,
            "wall_ms": wall_ms,
        },
        "events": sorted(replayed, key=lambda item: item["started_ms"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LangGraph RAG tool-calling Agent benchmark"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("live", "replay"):
        item = subparsers.add_parser(command)
        item.add_argument("--base-url", default="http://127.0.0.1:9000/v1")
        item.add_argument("--api-key", default="local")
        item.add_argument("--model", required=True)
        item.add_argument("--concurrency", type=int, default=8)
        item.add_argument("--output", type=Path, required=True)
        if command == "live":
            source = item.add_mutually_exclusive_group(required=True)
            source.add_argument(
                "--dataset",
                choices=["swebench", "agencybench", "agentboard", "appworld"],
            )
            source.add_argument("--task-file", type=Path)
            item.add_argument("--data-path", type=Path)
            item.add_argument("--sample-index", type=int, default=0)
            item.add_argument("--cases", type=int, default=1)
            item.add_argument("--branches", type=int, default=4)
            item.add_argument("--rag-root", type=Path, default=Path("..") / "docs")
            item.add_argument("--bootstrap-chunks", type=int, default=12)
            item.add_argument("--bootstrap-max-chars", type=int, default=32000)
            item.add_argument("--planner-tokens", type=int, default=96)
            item.add_argument("--tool-tokens", type=int, default=64)
            item.add_argument("--reflect-tokens", type=int, default=128)
            item.add_argument("--reduce-tokens", type=int, default=128)
        else:
            item.add_argument("--trace", type=Path, required=True)
            item.add_argument(
                "--timing", choices=["dependency", "captured"], default="dependency"
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = asyncio.run(
        run_live(args) if args.command == "live" else replay_trace(args)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
