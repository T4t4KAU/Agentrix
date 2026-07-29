"""Pure helpers for the equal-KV Agentrix full-stack experiment."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
_ACTION_RE = re.compile(
    r"(?im)^\s*(SEARCH|READ|FINAL)\s*:\s*(.+?)\s*$"
)


@dataclass(frozen=True)
class Passage:
    passage_id: str
    text: str


@dataclass(frozen=True)
class AgentAction:
    kind: str
    value: str


def tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(text)]


def build_passages(
    context: str, *, max_chars: int = 1800, overlap_chars: int = 180
) -> list[Passage]:
    """Split a document into stable, opaque passages."""

    if max_chars < 128:
        raise ValueError("max_chars must be at least 128")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be in [0, max_chars)")
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", context)
        if paragraph.strip()
    ]
    if not paragraphs:
        paragraphs = [context.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue
        step = max_chars - overlap_chars
        for start in range(0, len(paragraph), step):
            chunk = paragraph[start : start + max_chars].strip()
            if chunk:
                chunks.append(chunk)
            if start + max_chars >= len(paragraph):
                break
    return [
        Passage(passage_id=f"P{index:04d}", text=text)
        for index, text in enumerate(chunks)
    ]


class BM25Index:
    """Small dependency-free BM25 index scoped to one LongBench document."""

    def __init__(self, passages: Iterable[Passage]) -> None:
        self.passages = list(passages)
        if not self.passages:
            raise ValueError("at least one passage is required")
        self.tokens = [tokenize(passage.text) for passage in self.passages]
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / len(self.lengths)
        self.document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            self.document_frequency.update(set(tokens))

    def search(self, query: str, top_k: int = 4) -> list[tuple[Passage, float]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return [(passage, 0.0) for passage in self.passages[:top_k]]
        scores = []
        count = len(self.passages)
        for index, passage in enumerate(self.passages):
            score = 0.0
            length_norm = 1 - 0.75 + 0.75 * (
                self.lengths[index] / max(self.average_length, 1)
            )
            for token in query_tokens:
                frequency = self.term_frequencies[index][token]
                if not frequency:
                    continue
                df = self.document_frequency[token]
                inverse_document_frequency = math.log(
                    1 + (count - df + 0.5) / (df + 0.5)
                )
                score += inverse_document_frequency * (
                    frequency * 2.5 / (frequency + 1.5 * length_norm)
                )
            scores.append((passage, score))
        scores.sort(key=lambda item: (-item[1], item[0].passage_id))
        return scores[: max(1, top_k)]


def parse_action(text: str) -> AgentAction | None:
    match = _ACTION_RE.search(text)
    if match is None:
        return None
    kind, value = match.groups()
    value = value.strip()
    if kind == "FINAL":
        value = value.split("||", 1)[0].strip()
    if not value:
        return None
    return AgentAction(kind=kind.casefold(), value=value)


def render_search_results(
    matches: Iterable[tuple[Passage, float]], *, snippet_chars: int = 900
) -> str:
    return "\n\n".join(
        f"[{passage.passage_id}] score={score:.4f}\n"
        f"{passage.text[:snippet_chars]}"
        for passage, score in matches
    )


def validate_equal_kv_metadata(
    baseline: dict[str, Any], optimized: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed when a supposedly strict A/B changed capacity controls."""

    keys = (
        "model",
        "dtype",
        "data_parallel_size",
        "num_gpu_blocks_override",
        "block_size_tokens",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "enforce_eager",
        "async_scheduling",
        "case_manifest_sha256",
        "question_limit",
        "tool_rounds",
        "tool_delay_ms",
        "action_tokens",
        "answer_tokens",
    )
    mismatches = {
        key: {"baseline": baseline.get(key), "optimized": optimized.get(key)}
        for key in keys
        if baseline.get(key) != optimized.get(key)
    }
    if mismatches:
        raise ValueError(
            "strict equal-KV A/B validation failed: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    blocks = baseline.get("num_gpu_blocks_override")
    if not isinstance(blocks, int) or blocks <= 0:
        raise ValueError("num_gpu_blocks_override must be a positive integer")
    return {
        "strict_equal_gpu_kv_capacity": True,
        "matched_controls": list(keys),
        "num_gpu_blocks_override_per_rank": blocks,
        "data_parallel_size": baseline["data_parallel_size"],
        "total_gpu_blocks": blocks * baseline["data_parallel_size"],
    }


def load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload
