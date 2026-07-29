"""LongBench v1 shared-document QA loading, scoring, and case selection."""
from __future__ import annotations
import hashlib, json, re, string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

def normalize_answer(text: str) -> str:
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())

def qa_f1(prediction: str, reference: str) -> float:
    predicted, gold = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not predicted or not gold:
        return float(predicted == gold)
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)

def score_answer(prediction: str, references: Iterable[str]) -> dict[str, float]:
    refs = list(references)
    if not refs:
        raise ValueError("at least one reference answer is required")
    return {
        "exact_match": max(float(normalize_answer(prediction) == normalize_answer(ref)) for ref in refs),
        "f1": max(qa_f1(prediction, ref) for ref in refs),
    }

def build_shared_document_cases(
    source_paths: Iterable[Path], *, minimum_questions: int = 2,
    maximum_questions: int = 4, maximum_cases: int = 32,
    token_counter: Callable[[str], int] | None = None,
    maximum_context_tokens: int = 38_000,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in source_paths:
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                context, dataset = str(row["context"]), str(row.get("dataset") or source.stem)
                grouped[(dataset, context)].append({
                    "question": str(row["input"]),
                    "answers": [str(value) for value in row["answers"]],
                    "source_id": str(row.get("_id", f"{source.name}:{line_number}")),
                })
    candidates = []
    for (dataset, context), questions in grouped.items():
        if len(questions) < minimum_questions:
            continue
        tokens = token_counter(context) if token_counter else max(1, len(context) // 4)
        if tokens > maximum_context_tokens:
            continue
        digest = hashlib.sha256(context.encode()).hexdigest()
        candidates.append({
            "case_id": f"{dataset}-{digest[:16]}", "dataset": dataset,
            "context_sha256": digest, "context_tokens": tokens, "context": context,
            "questions": questions[:maximum_questions],
        })
    candidates.sort(key=lambda case: (-int(case["context_tokens"]), str(case["dataset"]), str(case["case_id"])))
    return candidates[:maximum_cases]

def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for case in cases:
        if len(case.get("questions", [])) < 2:
            raise ValueError(f"{case.get('case_id')}: requires at least two questions")
        if hashlib.sha256(case["context"].encode()).hexdigest() != case["context_sha256"]:
            raise ValueError(f"{case.get('case_id')}: context hash mismatch")
    return cases
