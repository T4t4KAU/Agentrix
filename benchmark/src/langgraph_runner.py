from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import operator
import random
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Annotated, Any, TypedDict

from openai import AsyncOpenAI

from agentrix_application import (
    CompactedPrompt,
    PromptSection,
    compact_prompt_delta,
)
from data import load_records, record_to_prompt
from hotpot import (
    HotpotExample,
    evaluate_predictions as evaluate_hotpot_predictions,
    load_hotpot,
    stratified_sample,
)


CACHEBLEND_SEPARATOR = "§CACHEBLEND§"
CACHEBLEND_PROTOCOL = "Agentrix stable RAG context protocol v1"

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

HOTPOT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "paragraph_search",
            "description": (
                "Search the candidate HotpotQA paragraphs for evidence. "
                "Results preserve title and sentence IDs for citation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        },
    },
]


class OverallState(TypedDict, total=False):
    case_id: str
    task: str
    context_query: str
    bootstrap_evidence: str
    bootstrap_results: list[dict[str, Any]]
    shared_analysis: str
    branches: int
    branch_roles: list[str]
    branch_specs: list[dict[str, Any]]
    branch_outputs: Annotated[list[dict[str, Any]], operator.add]
    answer: Any


class BranchState(TypedDict, total=False):
    case_id: str
    task: str
    bootstrap_evidence: str
    bootstrap_results: list[dict[str, Any]]
    shared_analysis: str
    branch_id: int
    branch_role: str
    branch_spec: dict[str, Any]
    required_tool: str


def _words(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


class LocalRAG:
    """Small deterministic BM25 index over real local text files."""

    def __init__(
        self,
        root: Path,
        chunk_chars: int = 1800,
        manifest: Path | None = None,
    ) -> None:
        self.chunks: list[dict[str, str]] = []
        if manifest is None:
            paths = sorted(root.rglob("*"))
        else:
            entries = [
                line.strip()
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            paths = [root / entry for entry in entries]
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    "RAG manifest entries do not exist: " + ", ".join(missing)
                )
        for path in paths:
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
                    content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
                    self.chunks.append(
                        {
                            "chunk_id": f"{relative}:{offset}:{content_hash}",
                            "source": relative,
                            "text": body,
                        }
                    )
        self.term_frequencies = [Counter(_words(item["text"])) for item in self.chunks]
        self.lengths = [sum(counter.values()) for counter in self.term_frequencies]
        self.avg_length = statistics.fmean(self.lengths) if self.lengths else 1.0
        self.document_frequency = Counter()
        for counter in self.term_frequencies:
            self.document_frequency.update(counter.keys())
        corpus_material = "\n".join(item["chunk_id"] for item in self.chunks)
        self.corpus_version = hashlib.sha256(
            corpus_material.encode("utf-8")
        ).hexdigest()[:16]

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
                "chunk_id": self.chunks[index]["chunk_id"],
                "source": self.chunks[index]["source"],
                "score": round(score, 4),
                "text": self.chunks[index]["text"][:1200],
            }
            for score, index in sorted(scored, reverse=True)[: max(1, min(top_k, 32))]
        ]


class HotpotRAG:
    """Deterministic BM25 retrieval scoped to each HotpotQA question.

    The distractor split already provides a ten-paragraph candidate set per
    question. Keeping the index case-scoped prevents concurrent cases from
    leaking paragraphs into one another while preserving a realistic retrieval
    step over gold and distractor documents.
    """

    def __init__(self, examples: list[HotpotExample]) -> None:
        self._case_chunks: dict[str, list[dict[str, Any]]] = {}
        all_chunks: dict[str, dict[str, Any]] = {}
        for example in examples:
            chunks: list[dict[str, Any]] = []
            for paragraph in example.context:
                rendered = "\n".join(
                    f"[{sentence_id}] {sentence}"
                    for sentence_id, sentence in enumerate(paragraph.sentences)
                )
                digest = hashlib.sha256(
                    (paragraph.title + "\0" + rendered).encode("utf-8")
                ).hexdigest()[:12]
                chunk = {
                    "chunk_id": f"hotpot:{digest}",
                    "source": paragraph.title,
                    "title": paragraph.title,
                    "text": rendered,
                    "sentences": [
                        {"sentence_id": sentence_id, "text": sentence}
                        for sentence_id, sentence in enumerate(paragraph.sentences)
                    ],
                }
                chunks.append(chunk)
                all_chunks.setdefault(chunk["chunk_id"], chunk)
            self._case_chunks[example.example_id] = chunks
        self.chunks = list(all_chunks.values())
        corpus_material = "\n".join(sorted(all_chunks))
        self.corpus_version = hashlib.sha256(
            corpus_material.encode("utf-8")
        ).hexdigest()[:16]

    def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        case_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if case_id is None or case_id not in self._case_chunks:
            raise KeyError(f"Hotpot retrieval requires a known case_id: {case_id!r}")
        chunks = self._case_chunks[case_id]
        frequencies = [
            Counter(_words(chunk["title"] + " " + chunk["text"])) for chunk in chunks
        ]
        lengths = [sum(counter.values()) for counter in frequencies]
        average_length = statistics.fmean(lengths) if lengths else 1.0
        document_frequency: Counter[str] = Counter()
        for counter in frequencies:
            document_frequency.update(counter.keys())
        query_terms = _words(query)
        total = max(len(chunks), 1)
        scored: list[tuple[float, int]] = []
        for index, counter in enumerate(frequencies):
            score = 0.0
            for term in query_terms:
                frequency = counter[term]
                if not frequency:
                    continue
                documents = document_frequency[term]
                inverse = math.log(1 + (total - documents + 0.5) / (documents + 0.5))
                denominator = frequency + 1.5 * (
                    0.25 + 0.75 * lengths[index] / average_length
                )
                score += inverse * frequency * 2.5 / denominator
            scored.append((score, index))
        # Bootstrap retrieval deliberately admits the full augmented candidate
        # set (normally 40 paragraphs). AgentRuntime still caps branch-local
        # paragraph_search calls at 10, so only the shared parent prompt grows.
        limit = max(1, min(top_k, len(chunks), 64))
        ordered = sorted(
            scored,
            key=lambda item: (-item[0], chunks[item[1]]["title"]),
        )[:limit]
        return [{**chunks[index], "score": round(score, 4)} for score, index in ordered]


def rag_prompt_sections(
    results: list[dict[str, Any]], *, rag_format: str
) -> list[PromptSection]:
    sections: list[PromptSection] = []
    for item in results:
        chunk_id = str(item.get("chunk_id", item["source"]))
        heading = (
            f"[Document {chunk_id}]\nSource: {item['source']}"
            if rag_format == "cacheblend"
            else f"[Local source: {item['source']}]"
        )
        sections.append(
            PromptSection(
                segment_id=f"rag:{chunk_id}",
                heading=heading,
                content=str(item["text"]),
            )
        )
    return sections


def _limit_rag_sections(
    sections: list[PromptSection], *, max_chars: int | None
) -> list[PromptSection]:
    selected: list[PromptSection] = []
    used_chars = 0
    for section in sections:
        rendered = section.render()
        separator_chars = 0 if not selected else 2
        if (
            max_chars is not None
            and used_chars + separator_chars + len(rendered) > max_chars
        ):
            break
        selected.append(section)
        used_chars += separator_chars + len(rendered)
    return selected


def _wrap_rag_text(text: str, rag_format: str) -> str:
    if rag_format != "cacheblend" or not text:
        return text
    separator = f"\n{CACHEBLEND_SEPARATOR}\n"
    return separator + text + separator


def format_rag_results(
    results: list[dict[str, Any]],
    *,
    rag_format: str,
    max_chars: int | None = None,
) -> str:
    """Render retrieval results without polluting reusable chunks with scores."""
    separator = f"\n{CACHEBLEND_SEPARATOR}\n" if rag_format == "cacheblend" else "\n\n"
    sections = _limit_rag_sections(
        rag_prompt_sections(results, rag_format=rag_format), max_chars=max_chars
    )
    return _wrap_rag_text(
        separator.join(section.render() for section in sections), rag_format
    )


def compact_rag_results(
    results: list[dict[str, Any]],
    *,
    known_results: list[dict[str, Any]],
    rag_format: str,
) -> CompactedPrompt:
    """Remove only exact RAG chunks already present in an earlier message."""
    separator = f"\n{CACHEBLEND_SEPARATOR}\n" if rag_format == "cacheblend" else "\n\n"
    compacted = compact_prompt_delta(
        rag_prompt_sections(results, rag_format=rag_format),
        known_sections=rag_prompt_sections(known_results, rag_format=rag_format),
        separator=separator,
    )
    return CompactedPrompt(
        text=_wrap_rag_text(compacted.text, rag_format),
        report=compacted.report,
    )


def stage_system_prompt(instruction: str, rag_format: str) -> str:
    if rag_format != "cacheblend":
        return instruction
    separator = f"\n{CACHEBLEND_SEPARATOR}\n"
    return f"{CACHEBLEND_PROTOCOL}{separator}{instruction}{separator}"


def summarize_rag_reuse(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe natural cross-case reuse in the captured bootstrap retrievals."""
    retrievals = [
        event.get("result", [])
        for event in events
        if event.get("kind") == "tool" and event.get("stage") == "bootstrap_rag"
    ]
    orders = [
        [str(item.get("chunk_id", item.get("source", ""))) for item in result]
        for result in retrievals
    ]
    text_sizes: dict[str, int] = {}
    occurrences: Counter[str] = Counter()
    total_chars = 0
    for result in retrievals:
        for item in result:
            chunk_id = str(item.get("chunk_id", item.get("source", "")))
            size = len(str(item.get("text", "")))
            text_sizes.setdefault(chunk_id, size)
            occurrences[chunk_id] += 1
            total_chars += size
    reusable_chars = sum(
        text_sizes[chunk_id] * (count - 1)
        for chunk_id, count in occurrences.items()
        if count > 1
    )
    pairwise_jaccard: list[float] = []
    reordered_pairs = 0
    for left_index, left in enumerate(orders):
        for right in orders[left_index + 1 :]:
            left_set, right_set = set(left), set(right)
            union = left_set | right_set
            if union:
                pairwise_jaccard.append(len(left_set & right_set) / len(union))
            common = left_set & right_set
            if len(common) >= 2:
                left_order = [item for item in left if item in common]
                right_order = [item for item in right if item in common]
                if left_order != right_order:
                    reordered_pairs += 1
    return {
        "retrievals": len(retrievals),
        "chunk_occurrences": sum(occurrences.values()),
        "unique_chunks": len(occurrences),
        "repeated_chunk_occurrences": sum(
            count - 1 for count in occurrences.values() if count > 1
        ),
        "reusable_chars": reusable_chars,
        "reuse_ratio": reusable_chars / total_chars if total_chars else 0.0,
        "mean_pairwise_jaccard": (
            statistics.fmean(pairwise_jaccard) if pairwise_jaccard else 0.0
        ),
        "reordered_pairs": reordered_pairs,
    }


def summarize_prompt_compaction(events: list[dict[str, Any]]) -> dict[str, Any]:
    reports = [
        event["compaction"]
        for event in events
        if event.get("kind") == "tool" and event.get("compaction") is not None
    ]
    return {
        "tool_results": len(reports),
        "input_sections": sum(item["input_sections"] for item in reports),
        "output_sections": sum(item["output_sections"] for item in reports),
        "removed_empty_sections": sum(
            item["removed_empty_sections"] for item in reports
        ),
        "removed_duplicate_sections": sum(
            item["removed_duplicate_sections"] for item in reports
        ),
        "before_chars": sum(item["before_chars"] for item in reports),
        "after_chars": sum(item["after_chars"] for item in reports),
        "saved_chars": sum(item["saved_chars"] for item in reports),
    }


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
        rag: LocalRAG | HotpotRAG,
        recorder: TraceRecorder,
        concurrency: int,
        rag_format: str = "plain",
        prompt_compaction: bool = False,
        workload: str = "legacy",
    ) -> None:
        self.client = client
        self.model = model
        self.rag = rag
        self.recorder = recorder
        self.semaphore = asyncio.Semaphore(concurrency)
        self.rag_format = rag_format
        self.prompt_compaction = prompt_compaction
        self.workload = workload

    async def complete(
        self,
        *,
        case_id: str,
        stage: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        branch_id: int | None = None,
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
                "branch_id": branch_id,
                "started_ms": started_ms,
                "latency_ms": latency_ms,
                "request": request,
                "response": response_data,
                "usage": usage,
            }
        )
        return message

    async def run_tool(
        self,
        case_id: str,
        branch_id: int,
        name: str,
        arguments: dict[str, Any],
        delay_ms: float = 0,
        known_results: list[dict[str, Any]] | None = None,
    ) -> str:
        query = str(arguments.get("query") or "Agentrix shared prefix attention")
        top_k_limit = 10 if self.workload == "hotpot" else 5
        top_k = max(1, min(int(arguments.get("top_k") or 3), top_k_limit))
        started_ms = (time.perf_counter() - self.recorder.started) * 1000
        started = time.perf_counter()
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000)
        if name == "rag_search":
            result: Any = self.rag.search(query, top_k)
        elif name == "paragraph_search" and isinstance(self.rag, HotpotRAG):
            result = self.rag.search(query, top_k, case_id=case_id)
        else:
            result = {"error": f"unsupported tool: {name}"}
        latency_ms = (time.perf_counter() - started) * 1000
        compaction = None
        if isinstance(result, list) and self.prompt_compaction and known_results:
            compacted = compact_rag_results(
                result,
                known_results=known_results,
                rag_format=self.rag_format,
            )
            content = compacted.text
            compaction = asdict(compacted.report)
            compaction["saved_chars"] = compacted.report.saved_chars
        elif isinstance(result, list):
            content = format_rag_results(result, rag_format=self.rag_format)
        else:
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
                "compaction": compaction,
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
        return {"id": call.id, "name": required_tool, "arguments": arguments}
    return {
        "id": f"fallback-{required_tool}",
        "name": required_tool,
        "arguments": {"query": fallback_query, "top_k": 3},
    }


def expand_branch_roles(branches: int, branch_roles: list[str] | None) -> list[str]:
    roles = list(branch_roles or [])
    if len(roles) > branches:
        raise ValueError("branch_roles cannot contain more roles than branches")
    if not roles:
        return [f"research perspective {index}" for index in range(branches)]
    original_roles = list(roles)
    while len(roles) < branches:
        base_role = original_roles[len(roles) % len(original_roles)]
        roles.append(f"independent verification: {base_role}")
    return roles


def _parse_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _hotpot_branch_specs(
    planner_text: str,
    *,
    question: str,
    bootstrap_results: list[dict[str, Any]],
    minimum: int,
    maximum: int,
) -> list[dict[str, Any]]:
    parsed = _parse_json_object(planner_text) or {}
    raw_specs = parsed.get("branch_specs")
    specs: list[dict[str, Any]] = []
    if isinstance(raw_specs, list):
        for item in raw_specs:
            if not isinstance(item, dict):
                continue
            query = str(item.get("query") or "").strip()
            if not query:
                continue
            specs.append(
                {
                    "kind": str(item.get("kind") or "candidate_path"),
                    "query": query,
                    "goal": str(item.get("goal") or "Find citable evidence."),
                }
            )
            if len(specs) >= maximum:
                break

    # Planner output is deliberately treated as untrusted. Retrieval-derived
    # fallbacks preserve benchmark progress without consulting gold facts.
    seen_queries = {spec["query"] for spec in specs}
    for result in bootstrap_results:
        if len(specs) >= max(minimum, 1):
            break
        title = str(result.get("title") or result.get("source") or "").strip()
        query = f"{question} Evidence in {title}" if title else question
        if query in seen_queries:
            continue
        specs.append(
            {
                "kind": "retrieved_candidate",
                "query": query,
                "goal": f"Verify whether {title or 'this candidate'} supports an answer.",
            }
        )
        seen_queries.add(query)
    if not specs:
        specs.append(
            {
                "kind": "general_verification",
                "query": question,
                "goal": "Find the strongest citable evidence for the answer.",
            }
        )
    return specs[:maximum]


def _normalize_supporting_facts(value: Any) -> list[list[Any]]:
    if not isinstance(value, list):
        return []
    normalized: list[list[Any]] = []
    seen: set[tuple[str, int]] = set()
    for item in value:
        if isinstance(item, dict):
            item = (item.get("title"), item.get("sentence_id"))
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], int)
            or isinstance(item[1], bool)
            or item[1] < 0
        ):
            continue
        fact = (item[0], item[1])
        if fact not in seen:
            normalized.append([fact[0], fact[1]])
            seen.add(fact)
    return normalized


def _hotpot_branch_output(
    text: str | None,
    *,
    branch_id: int,
    branch_spec: dict[str, Any],
) -> dict[str, Any]:
    parsed = _parse_json_object(text) or {}
    confidence = parsed.get("confidence", 0.0)
    try:
        confidence = min(1.0, max(0.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "branch_id": branch_id,
        "kind": branch_spec.get("kind", "candidate_path"),
        "query": branch_spec.get("query", ""),
        "claim": str(parsed.get("claim") or text or ""),
        "evidence": _normalize_supporting_facts(parsed.get("evidence", [])),
        "confidence": confidence,
    }


def _hotpot_final_output(text: str | None) -> dict[str, Any]:
    parsed = _parse_json_object(text) or {}
    return {
        "answer": str(parsed.get("answer") or text or "").strip(),
        "supporting_facts": _normalize_supporting_facts(
            parsed.get("supporting_facts", parsed.get("sp", []))
        ),
    }


def build_graph(
    runtime: AgentRuntime,
    branches: int,
    token_limits: dict[str, int],
    bootstrap_chunks: int = 12,
    bootstrap_max_chars: int = 32000,
    tool_delay_ms: float = 0,
    branch_roles: list[str] | None = None,
    workload: str = "legacy",
    branch_min: int = 4,
    branch_max: int = 8,
    tool_delay_profile: str = "legacy",
    seed: int = 2026,
):
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send

    def shared_root_messages(
        bootstrap_evidence: str,
        task: str,
        rag_format: str,
    ) -> list[dict[str, Any]]:
        """Build the exact parent request inherited by every fanout branch."""
        if workload == "hotpot":
            system_instruction = (
                "You are coordinating parallel evidence verification for a "
                "HotpotQA multi-hop question. Use only retrieved paragraphs, "
                "preserve title and sentence IDs, and do not answer before the "
                "evidence branches finish."
            )
            planner_instruction = (
                "Plan candidate bridge or comparison paths. Return one JSON object "
                "with keys analysis and branch_specs. branch_specs must be a list "
                f"of {branch_min} to {branch_max} objects with kind, query, and "
                "goal. Do not use any gold answer or supporting-fact knowledge."
            )
        else:
            system_instruction = (
                "You are a parallel research agent. Build a shared plan, then use "
                "local RAG tools to investigate role-specific evidence."
            )
            planner_instruction = (
                "Create the common analysis inherited by all parallel research "
                "branches. Identify uncertainties and divide the remaining "
                "evidence work without answering the task yet."
            )
        return [
            {
                "role": "system",
                "content": stage_system_prompt(
                    system_instruction,
                    rag_format,
                ),
            },
            {
                "role": "user",
                "content": "Retrieved local evidence:\n\n" + bootstrap_evidence,
            },
            {"role": "user", "content": task},
            {
                "role": "user",
                "content": planner_instruction,
            },
        ]

    async def retrieve_context(state: OverallState) -> dict[str, Any]:
        started_ms = (time.perf_counter() - runtime.recorder.started) * 1000
        started = time.perf_counter()
        query = state.get("context_query", state["task"])
        if workload == "hotpot" and isinstance(runtime.rag, HotpotRAG):
            searched_results = runtime.rag.search(
                query, bootstrap_chunks, case_id=state["case_id"]
            )
            bootstrap_tool = "paragraph_search"
        else:
            searched_results = runtime.rag.search(query, bootstrap_chunks)
            bootstrap_tool = "rag_search"
        rag_format = getattr(runtime, "rag_format", "plain")
        included_sections = _limit_rag_sections(
            rag_prompt_sections(searched_results, rag_format=rag_format),
            max_chars=bootstrap_max_chars,
        )
        results = searched_results[: len(included_sections)]
        evidence = format_rag_results(
            results,
            rag_format=rag_format,
        )
        await runtime.recorder.add(
            {
                "kind": "tool",
                "case_id": state["case_id"],
                "branch_id": -1,
                "stage": "bootstrap_rag",
                "tool": bootstrap_tool,
                "arguments": {
                    "query": query,
                    "top_k": bootstrap_chunks,
                },
                "started_ms": started_ms,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "result": results,
            }
        )
        return {"bootstrap_evidence": evidence, "bootstrap_results": results}

    async def planner(state: OverallState) -> dict[str, Any]:
        rag_format = getattr(runtime, "rag_format", "plain")
        message = await runtime.complete(
            case_id=state["case_id"],
            stage="planner",
            messages=shared_root_messages(
                state["bootstrap_evidence"], state["task"], rag_format
            ),
            max_tokens=token_limits["planner"],
        )
        shared_analysis = message.content or "Investigate the task."
        if workload != "hotpot":
            return {"shared_analysis": shared_analysis}
        return {
            "shared_analysis": shared_analysis,
            "branch_specs": _hotpot_branch_specs(
                shared_analysis,
                question=state["task"],
                bootstrap_results=state["bootstrap_results"],
                minimum=branch_min,
                maximum=branch_max,
            ),
        }

    def fanout(state: OverallState) -> list[Any]:
        if workload == "hotpot":
            specs = state.get("branch_specs") or _hotpot_branch_specs(
                "",
                question=state["task"],
                bootstrap_results=state["bootstrap_results"],
                minimum=branch_min,
                maximum=branch_max,
            )
            return [
                Send(
                    "branch_agent",
                    {
                        "case_id": state["case_id"],
                        "task": state["task"],
                        "bootstrap_evidence": state["bootstrap_evidence"],
                        "bootstrap_results": state["bootstrap_results"],
                        "shared_analysis": state["shared_analysis"],
                        "branch_id": branch_id,
                        "branch_role": spec.get("goal", f"candidate {branch_id}"),
                        "branch_spec": spec,
                        "required_tool": "paragraph_search",
                    },
                )
                for branch_id, spec in enumerate(specs)
            ]
        roles = expand_branch_roles(branches, branch_roles)
        return [
            Send(
                "branch_agent",
                {
                    "case_id": state["case_id"],
                    "task": state["task"],
                    "bootstrap_evidence": state["bootstrap_evidence"],
                    "bootstrap_results": state["bootstrap_results"],
                    "shared_analysis": state["shared_analysis"],
                    "branch_id": branch_id,
                    "branch_role": roles[branch_id],
                    "required_tool": "rag_search",
                },
            )
            for branch_id in range(branches)
        ]

    async def branch_agent(state: BranchState) -> dict[str, Any]:
        required_tool = state["required_tool"]
        rag_format = getattr(runtime, "rag_format", "plain")
        shared_messages: list[dict[str, Any]] = [
            *shared_root_messages(
                state["bootstrap_evidence"], state["task"], rag_format
            ),
            {"role": "assistant", "content": state["shared_analysis"]},
        ]
        branch_spec = state.get("branch_spec", {})
        if workload == "hotpot":
            branch_prompt = (
                "Verify this candidate path and call paragraph_search using its "
                "query. Candidate specification:\n"
                + json.dumps(branch_spec, ensure_ascii=False, sort_keys=True)
            )
            available_tools = HOTPOT_TOOLS
            fallback_query = str(branch_spec.get("query") or state["task"][:500])
        else:
            branch_prompt = (
                f"Research role: {state['branch_role']}. Call {required_tool} with a "
                "focused query for that role."
            )
            available_tools = TOOLS
            fallback_query = state["task"][:500]
        selection_messages = [
            *shared_messages,
            {"role": "user", "content": branch_prompt},
        ]
        selected = await runtime.complete(
            case_id=state["case_id"],
            stage="tool_select",
            messages=selection_messages,
            max_tokens=token_limits["tool_select"],
            tools=available_tools,
            tool_choice={"type": "function", "function": {"name": required_tool}},
            branch_id=state["branch_id"],
        )
        call = _tool_call(selected, required_tool, fallback_query)
        if workload == "hotpot":
            call["arguments"]["query"] = fallback_query
            call["arguments"].setdefault("top_k", 3)
        if workload == "hotpot" and tool_delay_profile == "lognormal":
            delay_rng = random.Random(f"{seed}:{state['case_id']}:{state['branch_id']}")
            delay_ms = tool_delay_ms * delay_rng.lognormvariate(-0.25, 0.75)
        elif workload == "hotpot" and tool_delay_profile == "synchronized":
            delay_ms = 0.0
        elif workload == "hotpot":
            delay_ms = tool_delay_ms
        else:
            delay_ms = tool_delay_ms * (1 + state["branch_id"] % 4)
        tool_content = await runtime.run_tool(
            state["case_id"],
            state["branch_id"],
            call["name"],
            call["arguments"],
            delay_ms=delay_ms,
            known_results=state["bootstrap_results"],
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
                "content": (
                    "Return one JSON object with claim, evidence, and confidence. "
                    "evidence must be a list of [title, sentence_id] pairs copied "
                    "from the tool result."
                    if workload == "hotpot"
                    else "Synthesize the evidence and cite its source titles or local paths."
                ),
            },
        ]
        reflected = await runtime.complete(
            case_id=state["case_id"],
            stage="branch_reflect",
            messages=reflection_messages,
            max_tokens=token_limits["reflect"],
            branch_id=state["branch_id"],
        )
        if workload == "hotpot":
            return {
                "branch_outputs": [
                    _hotpot_branch_output(
                        reflected.content,
                        branch_id=state["branch_id"],
                        branch_spec=branch_spec,
                    )
                ]
            }
        return {
            "branch_outputs": [
                {
                    "branch_id": state["branch_id"],
                    "branch_role": state["branch_role"],
                    "tool": call["name"],
                    "answer": reflected.content or "",
                }
            ]
        }

    async def reducer(state: OverallState) -> dict[str, Any]:
        if workload == "hotpot":
            evidence = json.dumps(
                sorted(state["branch_outputs"], key=lambda item: item["branch_id"]),
                ensure_ascii=False,
                sort_keys=True,
            )
            reducer_instruction = (
                "You are the HotpotQA reducer. Use only cited branch evidence. "
                "Return one JSON object with answer and supporting_facts; "
                "supporting_facts must be [title, sentence_id] pairs."
            )
        else:
            reducer_instruction = (
                "You are the reducer. Produce a concise evidence-grounded final "
                "answer and flag conflicts."
            )
            evidence = "\n\n".join(
                f"Branch {item['branch_id']} — {item.get('branch_role', 'research')} "
                f"({item['tool']}):\n{item['answer']}"
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
                    "content": stage_system_prompt(
                        reducer_instruction,
                        getattr(runtime, "rag_format", "plain"),
                    ),
                },
                {
                    "role": "user",
                    "content": "Retrieved local evidence:\n\n"
                    + state["bootstrap_evidence"],
                },
                {"role": "user", "content": state["task"]},
                {"role": "assistant", "content": state["shared_analysis"]},
                {"role": "user", "content": evidence},
            ],
            max_tokens=token_limits["reduce"],
        )
        if workload == "hotpot":
            return {"answer": _hotpot_final_output(message.content)}
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
    hotpot_examples: list[HotpotExample] = []
    hotpot_rag_examples: list[HotpotExample] = []
    workload = "hotpot" if getattr(args, "hotpot_path", None) else "legacy"
    if workload == "hotpot":
        all_examples = load_hotpot(args.hotpot_path)
        case_file = getattr(args, "hotpot_case_file", None)
        if case_file is not None:
            case_records = [
                json.loads(line)
                for line in case_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            requested_ids = [record["id"] for record in case_records]
            by_id = {example.example_id: example for example in all_examples}
            missing = [
                example_id for example_id in requested_ids if example_id not in by_id
            ]
            if missing:
                raise ValueError(
                    "Hotpot case file contains unknown IDs: " + ", ".join(missing[:5])
                )
            selected_records = case_records[
                args.sample_index : args.sample_index + args.cases
            ]
            hotpot_examples = [by_id[record["id"]] for record in selected_records]
            for record, example in zip(selected_records, hotpot_examples):
                paragraphs = list(example.context)
                seen_titles = {paragraph.title for paragraph in paragraphs}
                for donor_id in record.get("distractor_ids", []):
                    donor = by_id.get(str(donor_id))
                    if donor is None:
                        raise ValueError(f"unknown Hotpot distractor ID: {donor_id}")
                    for paragraph in donor.context:
                        if paragraph.title not in seen_titles:
                            paragraphs.append(paragraph)
                            seen_titles.add(paragraph.title)
                hotpot_rag_examples.append(replace(example, context=tuple(paragraphs)))
        else:
            sample_count = min(len(all_examples), args.sample_index + args.cases)
            sampled = stratified_sample(
                all_examples,
                sample_count,
                seed=getattr(args, "hotpot_seed", 2026),
            )
            hotpot_examples = sampled[args.sample_index : sample_count]
            hotpot_rag_examples = list(hotpot_examples)
        selected_tasks = [
            {
                "case_id": example.example_id,
                "task": "HotpotQA question: " + example.question,
                "context_query": example.question,
                "branches": getattr(args, "hotpot_branch_max", args.branches),
                "start_delay_ms": 0.0,
                "tool_delay_ms": float(getattr(args, "hotpot_tool_delay_ms", 0.0)),
                "bootstrap_chunks": args.bootstrap_chunks,
                "bootstrap_max_chars": args.bootstrap_max_chars,
                "scenario": "hotpot_agentrix",
                "branch_roles": [],
            }
            for example in hotpot_examples
        ]
        dataset_name = "hotpotqa_distractor"
    elif args.task_file:
        task_records = [
            json.loads(line)
            for line in args.task_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.scenario:
            task_records = [
                item for item in task_records if item.get("scenario") == args.scenario
            ]
        selected_records = task_records[
            args.sample_index : args.sample_index + args.cases
        ]
        selected_tasks = [
            {
                "case_id": str(item.get("id", index)),
                "task": str(item["task"]),
                "context_query": str(item.get("context_query", item["task"])),
                "branches": int(item.get("branches", args.branches)),
                "start_delay_ms": float(item.get("start_delay_ms", 0)),
                "tool_delay_ms": float(item.get("tool_delay_ms", 0)),
                "bootstrap_chunks": int(
                    item.get("bootstrap_chunks", args.bootstrap_chunks)
                ),
                "bootstrap_max_chars": int(
                    item.get("bootstrap_max_chars", args.bootstrap_max_chars)
                ),
                "scenario": str(item.get("scenario", "unspecified")),
                "branch_roles": [str(role) for role in item.get("branch_roles", [])],
            }
            for index, item in enumerate(selected_records, args.sample_index)
        ]
        dataset_name = "task_file"
    else:
        records = load_records(args.dataset, args.data_path)
        selected = records[args.sample_index : args.sample_index + args.cases]
        selected_tasks = [
            {
                "case_id": f"{args.dataset}-{args.sample_index + index}",
                "task": record_to_prompt(args.dataset, record),
                "context_query": record_to_prompt(args.dataset, record),
                "branches": args.branches,
                "start_delay_ms": 0,
                "tool_delay_ms": 0,
                "bootstrap_chunks": args.bootstrap_chunks,
                "bootstrap_max_chars": args.bootstrap_max_chars,
                "scenario": "dataset",
                "branch_roles": [],
            }
            for index, record in enumerate(selected)
        ]
        dataset_name = args.dataset
    if workload != "hotpot":
        for spec in selected_tasks:
            spec["branch_roles"] = expand_branch_roles(
                spec["branches"], spec["branch_roles"]
            )
    if len(selected_tasks) != args.cases:
        raise ValueError("requested cases exceed available task records")
    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    recorder = TraceRecorder(args.model)
    rag: LocalRAG | HotpotRAG
    if workload == "hotpot":
        rag = HotpotRAG(hotpot_rag_examples)
    else:
        rag = LocalRAG(args.rag_root, manifest=args.rag_manifest)
    if (
        args.expected_rag_corpus_version is not None
        and rag.corpus_version != args.expected_rag_corpus_version
    ):
        raise ValueError(
            "RAG corpus version mismatch: expected "
            f"{args.expected_rag_corpus_version}, got {rag.corpus_version}"
        )
    runtime = AgentRuntime(
        client=client,
        model=args.model,
        rag=rag,
        recorder=recorder,
        concurrency=args.concurrency,
        rag_format=args.rag_format,
        prompt_compaction=args.prompt_compaction,
        workload=workload,
    )
    limits = {
        "planner": args.planner_tokens,
        "tool_select": args.tool_tokens,
        "reflect": args.reflect_tokens,
        "reduce": args.reduce_tokens,
    }
    case_semaphore = asyncio.Semaphore(
        args.case_concurrency if args.case_concurrency > 0 else len(selected_tasks)
    )

    async def invoke_case(spec: dict[str, Any]) -> dict[str, Any]:
        if spec["start_delay_ms"] > 0:
            await asyncio.sleep(spec["start_delay_ms"] / 1000)
        async with case_semaphore:
            graph = build_graph(
                runtime,
                spec["branches"],
                limits,
                bootstrap_chunks=spec["bootstrap_chunks"],
                bootstrap_max_chars=spec["bootstrap_max_chars"],
                tool_delay_ms=spec["tool_delay_ms"],
                branch_roles=spec["branch_roles"] or None,
                workload=workload,
                branch_min=getattr(args, "hotpot_branch_min", 4),
                branch_max=getattr(args, "hotpot_branch_max", 8),
                tool_delay_profile=getattr(args, "hotpot_tool_delay_profile", "legacy"),
                seed=getattr(args, "hotpot_seed", 2026),
            )
            return await graph.ainvoke(
                {
                    "case_id": spec["case_id"],
                    "task": spec["task"],
                    "context_query": spec["context_query"],
                    "branches": spec["branches"],
                    "branch_roles": spec["branch_roles"],
                    "branch_outputs": [],
                }
            )

    started = time.perf_counter()
    outputs = await asyncio.gather(*(invoke_case(spec) for spec in selected_tasks))
    wall_ms = (time.perf_counter() - started) * 1000
    metadata = {
        "mode": "live",
        "workload": workload,
        "dataset": dataset_name,
        "sample_index": args.sample_index,
        "cases": args.cases,
        "case_concurrency": args.case_concurrency,
        "branches_per_case": [spec["branches"] for spec in selected_tasks],
        "start_delays_ms": [spec["start_delay_ms"] for spec in selected_tasks],
        "tool_delays_ms": [spec["tool_delay_ms"] for spec in selected_tasks],
        "scenarios": [spec["scenario"] for spec in selected_tasks],
        "branch_roles": [spec["branch_roles"] for spec in selected_tasks],
        "rag_format": args.rag_format,
        "prompt_compaction": args.prompt_compaction,
        "rag_root": str(args.rag_root) if workload != "hotpot" else None,
        "rag_manifest": (
            str(args.rag_manifest)
            if workload != "hotpot" and args.rag_manifest
            else None
        ),
        "rag_chunks": len(rag.chunks),
        "rag_corpus_version": rag.corpus_version,
        "bootstrap_chunks_per_case": [
            spec["bootstrap_chunks"] for spec in selected_tasks
        ],
        "bootstrap_max_chars_per_case": [
            spec["bootstrap_max_chars"] for spec in selected_tasks
        ],
        "bootstrap_chars": [len(output["bootstrap_evidence"]) for output in outputs],
        "wall_ms": wall_ms,
    }
    metadata["rag_reuse"] = summarize_rag_reuse(recorder.events)
    metadata["prompt_compaction_report"] = summarize_prompt_compaction(recorder.events)
    payload = recorder.payload(metadata)
    if workload == "hotpot":
        answer_map: dict[str, str] = {}
        fact_map: dict[str, list[list[Any]]] = {}
        payload["outputs"] = []
        for output in outputs:
            final = output.get("answer")
            if not isinstance(final, dict):
                final = _hotpot_final_output(str(final or ""))
            case_id = output["case_id"]
            answer_map[case_id] = str(final.get("answer") or "")
            fact_map[case_id] = _normalize_supporting_facts(
                final.get("supporting_facts", [])
            )
            payload["outputs"].append(
                {
                    "case_id": case_id,
                    "answer": answer_map[case_id],
                    "supporting_facts": fact_map[case_id],
                    "branch_count": len(output.get("branch_outputs", [])),
                }
            )
        predictions = {"answer": answer_map, "sp": fact_map}
        payload["predictions"] = predictions
        payload["evaluation"] = evaluate_hotpot_predictions(
            hotpot_examples, predictions
        )
        metadata["hotpot_path"] = str(args.hotpot_path)
        metadata["hotpot_sha256"] = hashlib.sha256(
            args.hotpot_path.read_bytes()
        ).hexdigest()
        metadata["hotpot_case_ids"] = [
            example.example_id for example in hotpot_examples
        ]
        metadata["actual_branches_per_case"] = [
            len(output.get("branch_outputs", [])) for output in outputs
        ]
        bootstrap_by_case = {
            event["case_id"]: event.get("result", [])
            for event in recorder.events
            if event.get("stage") == "bootstrap_rag"
        }
        recalls = []
        for example in hotpot_examples:
            retrieved_titles = {
                str(item.get("title") or item.get("source"))
                for item in bootstrap_by_case.get(example.example_id, [])
            }
            gold_titles = {title for title, _ in example.supporting_facts}
            recalls.append(
                len(retrieved_titles & gold_titles) / len(gold_titles)
                if gold_titles
                else 0.0
            )
        metadata["bootstrap_supporting_title_recall"] = (
            statistics.fmean(recalls) if recalls else 0.0
        )
    else:
        payload["outputs"] = [
            {"case_id": output["case_id"], "answer": output["answer"]}
            for output in outputs
        ]
    return payload


async def replay_trace(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.trace.read_text(encoding="utf-8"))
    llm_events = [event for event in source["events"] if event["kind"] == "llm"]
    tool_events = [event for event in source["events"] if event["kind"] == "tool"]
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
        if getattr(args, "fixed_output_tokens", False):
            output_length_source = getattr(args, "fixed_output_length_source", "max")
            target_tokens = request.get("max_tokens")
            if output_length_source == "captured":
                captured_tokens = event.get("usage", {}).get("completion_tokens")
                if isinstance(captured_tokens, int) and captured_tokens > 0:
                    target_tokens = captured_tokens
                    request["max_tokens"] = captured_tokens
            if isinstance(target_tokens, int) and target_tokens > 0:
                extra_body = dict(request.get("extra_body") or {})
                extra_body.update({"min_tokens": target_tokens, "ignore_eos": True})
                request["extra_body"] = extra_body
            # A forced function-call grammar has a finite JSON language and
            # cannot legally continue to max_tokens after producing a valid
            # call. Keep the identical tool schema in the normalized replay,
            # but remove the constrained-decoding FSM so output length can be
            # held constant across backends.
            if isinstance(request.get("tool_choice"), dict):
                request["tool_choice"] = "auto"
        request_started = time.perf_counter()
        async with semaphore:
            response = await client.chat.completions.create(**request)
        choices = getattr(response, "choices", None)
        response_data = (
            choices[0].message.model_dump(exclude_none=True) if choices else {}
        )
        replayed.append(
            {
                "case_id": event["case_id"],
                "stage": event["stage"],
                "branch_id": event.get("branch_id"),
                "source_started_ms": event["started_ms"],
                "started_ms": (request_started - started) * 1000,
                "latency_ms": (time.perf_counter() - request_started) * 1000,
                "usage": response.usage.model_dump() if response.usage else {},
                "response": response_data,
            }
        )

    if args.timing == "captured":
        await asyncio.gather(
            *(replay(event, event["started_ms"]) for event in llm_events)
        )
    elif args.timing == "sequential":
        gap_s = max(0.0, float(getattr(args, "sequential_gap_ms", 0.0))) / 1000
        for event in sorted(llm_events, key=lambda item: item["started_ms"]):
            await replay(event)
            if gap_s:
                await asyncio.sleep(gap_s)
    elif args.timing == "agent":
        case_events: dict[str, list[dict[str, Any]]] = {}
        for event in llm_events:
            case_events.setdefault(event["case_id"], []).append(event)
        first_planner_ms = min(
            event["started_ms"] for event in llm_events if event["stage"] == "planner"
        )
        tool_delays = {
            (event["case_id"], event.get("branch_id")): event["latency_ms"]
            for event in tool_events
            if event["stage"] == "tool"
        }

        async def replay_agent_case(
            events: list[dict[str, Any]], *, preserve_initial_delay: bool
        ) -> None:
            planner = next(event for event in events if event["stage"] == "planner")
            initial_delay = max(0, planner["started_ms"] - first_planner_ms)
            if preserve_initial_delay and initial_delay:
                await asyncio.sleep(initial_delay / 1000)
            await replay(planner)
            selections = {
                event.get("branch_id"): event
                for event in events
                if event["stage"] == "tool_select"
            }
            reflections = {
                event.get("branch_id"): event
                for event in events
                if event["stage"] == "branch_reflect"
            }
            if None in selections or None in reflections:
                raise ValueError("agent replay requires branch_id on branch LLM events")

            async def replay_branch(branch_id: int) -> None:
                await replay(selections[branch_id])
                delay_ms = tool_delays.get((planner["case_id"], branch_id), 0)
                if delay_ms > 0:
                    await asyncio.sleep(delay_ms / 1000)
                await replay(reflections[branch_id])

            await asyncio.gather(*(replay_branch(key) for key in selections))
            reducer = next(event for event in events if event["stage"] == "reduce")
            await replay(reducer)

        case_concurrency = max(0, int(getattr(args, "case_concurrency", 0)))
        ordered_cases = sorted(
            case_events.values(),
            key=lambda events: next(
                event["started_ms"] for event in events if event["stage"] == "planner"
            ),
        )
        if case_concurrency:
            # Closed-loop Agent replay: keep at most N complete cases in
            # flight and release the next captured case as soon as one
            # finishes. Reusing the source trace's absolute planner timestamps
            # would pin a faster backend to the capture backend's throughput.
            case_slots = asyncio.Semaphore(case_concurrency)

            async def replay_bounded_case(events: list[dict[str, Any]]) -> None:
                async with case_slots:
                    await replay_agent_case(events, preserve_initial_delay=False)

            await asyncio.gather(
                *(replay_bounded_case(events) for events in ordered_cases)
            )
        else:
            await asyncio.gather(
                *(
                    replay_agent_case(events, preserve_initial_delay=True)
                    for events in ordered_cases
                )
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
            "case_concurrency": int(getattr(args, "case_concurrency", 0)),
            "fixed_output_tokens": bool(getattr(args, "fixed_output_tokens", False)),
            "fixed_output_length_source": getattr(
                args, "fixed_output_length_source", "max"
            ),
            "forced_tool_choice_normalized": bool(
                getattr(args, "fixed_output_tokens", False)
            ),
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
            source.add_argument(
                "--hotpot-path",
                type=Path,
                help="Official HotpotQA dev distractor JSON array.",
            )
            item.add_argument(
                "--hotpot-case-file",
                type=Path,
                help="Optional JSONL manifest containing fixed Hotpot IDs.",
            )
            item.add_argument("--hotpot-seed", type=int, default=2026)
            item.add_argument("--hotpot-branch-min", type=int, default=4)
            item.add_argument("--hotpot-branch-max", type=int, default=8)
            item.add_argument("--hotpot-tool-delay-ms", type=float, default=0.0)
            item.add_argument(
                "--hotpot-tool-delay-profile",
                choices=["synchronized", "fixed", "lognormal"],
                default="synchronized",
            )
            item.add_argument("--data-path", type=Path)
            item.add_argument("--sample-index", type=int, default=0)
            item.add_argument("--cases", type=int, default=1)
            item.add_argument(
                "--case-concurrency",
                type=int,
                default=0,
                help="Maximum simultaneously active LangGraph cases; 0 is unlimited.",
            )
            item.add_argument("--scenario")
            item.add_argument(
                "--rag-format",
                choices=["plain", "cacheblend"],
                default="plain",
            )
            item.add_argument(
                "--prompt-compaction",
                action="store_true",
                help=(
                    "Omit byte-identical RAG chunks already present in the "
                    "shared bootstrap context."
                ),
            )
            item.add_argument("--branches", type=int, default=4)
            item.add_argument("--rag-root", type=Path, default=Path("..") / "docs")
            item.add_argument("--rag-manifest", type=Path)
            item.add_argument(
                "--expected-rag-corpus-version",
                help="Fail if the content-addressed RAG corpus version differs.",
            )
            item.add_argument("--bootstrap-chunks", type=int, default=12)
            item.add_argument("--bootstrap-max-chars", type=int, default=32000)
            item.add_argument("--planner-tokens", type=int, default=96)
            item.add_argument("--tool-tokens", type=int, default=64)
            item.add_argument("--reflect-tokens", type=int, default=128)
            item.add_argument("--reduce-tokens", type=int, default=128)
        else:
            item.add_argument("--trace", type=Path, required=True)
            item.add_argument(
                "--case-concurrency",
                type=int,
                default=0,
                help=(
                    "For agent replay, keep at most this many complete cases "
                    "in flight and release the next case on completion; 0 "
                    "preserves captured planner arrival times."
                ),
            )
            item.add_argument(
                "--timing",
                choices=["dependency", "captured", "agent", "sequential"],
                default="dependency",
            )
            item.add_argument(
                "--sequential-gap-ms",
                type=float,
                default=0.0,
                help="Idle delay after each request in sequential replay mode.",
            )
            item.add_argument(
                "--fixed-output-tokens",
                action="store_true",
                help=(
                    "Force every replayed request to generate exactly its "
                    "captured max_tokens by setting min_tokens and ignore_eos."
                ),
            )
            item.add_argument(
                "--fixed-output-length-source",
                choices=["max", "captured"],
                default="max",
                help=(
                    "Use each request's max_tokens or its captured completion "
                    "token count as the exact replay output length."
                ),
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
