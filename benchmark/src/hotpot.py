"""HotpotQA data structures, sampling, and official-style evaluation.

This module deliberately depends only on the Python standard library so that the
benchmark can load and score HotpotQA without the legacy HotpotQA environment.
"""

from __future__ import annotations

import json
import random
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SupportingFact = tuple[str, int]


@dataclass(frozen=True)
class HotpotSentence:
    title: str
    sentence_id: int
    text: str


@dataclass(frozen=True)
class HotpotParagraph:
    title: str
    sentences: tuple[str, ...]

    def sentence(self, sentence_id: int) -> HotpotSentence:
        try:
            text = self.sentences[sentence_id]
        except IndexError as exc:
            raise KeyError(f"sentence {sentence_id} not found in {self.title!r}") from exc
        if sentence_id < 0:
            raise KeyError(f"sentence {sentence_id} not found in {self.title!r}")
        return HotpotSentence(self.title, sentence_id, text)

    def text(self, separator: str = "") -> str:
        """Return the paragraph in the same sentence order as the dataset."""

        return separator.join(self.sentences)


@dataclass(frozen=True)
class HotpotExample:
    example_id: str
    question: str
    answer: str | None
    supporting_facts: tuple[SupportingFact, ...]
    context: tuple[HotpotParagraph, ...]
    question_type: str | None = None
    level: str | None = None

    @property
    def id(self) -> str:
        """Alias matching the terminology used by benchmark traces."""

        return self.example_id


class HotpotCorpus:
    """A title-addressable paragraph and sentence corpus.

    HotpotQA repeats paragraphs between examples. Identical repeats are stored
    once. Conflicting text under the same title is rejected because title-based
    retrieval would otherwise be ambiguous.
    """

    def __init__(self, paragraphs: Iterable[HotpotParagraph]) -> None:
        by_title: dict[str, HotpotParagraph] = {}
        for paragraph in paragraphs:
            previous = by_title.get(paragraph.title)
            if previous is not None and previous.sentences != paragraph.sentences:
                raise ValueError(
                    f"conflicting paragraphs share title {paragraph.title!r}"
                )
            by_title[paragraph.title] = paragraph
        self._by_title = by_title

    @classmethod
    def from_examples(cls, examples: Iterable[HotpotExample]) -> "HotpotCorpus":
        return cls(
            paragraph
            for example in examples
            for paragraph in example.context
        )

    def __len__(self) -> int:
        return len(self._by_title)

    def __contains__(self, title: object) -> bool:
        return title in self._by_title

    @property
    def paragraphs(self) -> tuple[HotpotParagraph, ...]:
        return tuple(self._by_title.values())

    @property
    def sentences(self) -> tuple[HotpotSentence, ...]:
        return tuple(
            HotpotSentence(paragraph.title, sentence_id, text)
            for paragraph in self._by_title.values()
            for sentence_id, text in enumerate(paragraph.sentences)
        )

    def get_paragraph(self, title: str) -> HotpotParagraph:
        try:
            return self._by_title[title]
        except KeyError as exc:
            raise KeyError(f"paragraph not found: {title!r}") from exc

    def get_sentence(self, title: str, sentence_id: int) -> HotpotSentence:
        return self.get_paragraph(title).sentence(sentence_id)


def load_hotpot(path: str | Path) -> list[HotpotExample]:
    """Load a HotpotQA JSON array into immutable benchmark records."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"HotpotQA dataset not found: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid HotpotQA JSON: {source}") from exc
    if not isinstance(payload, list):
        raise ValueError(f"HotpotQA dataset must be a JSON array: {source}")
    return [_parse_example(value, index, source) for index, value in enumerate(payload)]


def _parse_example(value: Any, index: int, source: Path) -> HotpotExample:
    label = f"{source}[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{label}: example must be an object")

    example_id = _required_string(value, "_id", label)
    question = _required_string(value, "question", label)
    answer = value.get("answer")
    if answer is not None and not isinstance(answer, str):
        raise ValueError(f"{label}.answer: expected a string")

    raw_context = value.get("context")
    if not isinstance(raw_context, list):
        raise ValueError(f"{label}.context: expected a list")
    context: list[HotpotParagraph] = []
    for paragraph_index, raw_paragraph in enumerate(raw_context):
        paragraph_label = f"{label}.context[{paragraph_index}]"
        if (
            not isinstance(raw_paragraph, list)
            or len(raw_paragraph) != 2
            or not isinstance(raw_paragraph[0], str)
            or not isinstance(raw_paragraph[1], list)
            or not all(isinstance(sentence, str) for sentence in raw_paragraph[1])
        ):
            raise ValueError(
                f"{paragraph_label}: expected [title, [sentence, ...]]"
            )
        context.append(
            HotpotParagraph(raw_paragraph[0], tuple(raw_paragraph[1]))
        )

    raw_facts = value.get("supporting_facts", [])
    if not isinstance(raw_facts, list):
        raise ValueError(f"{label}.supporting_facts: expected a list")
    supporting_facts: list[SupportingFact] = []
    for fact_index, raw_fact in enumerate(raw_facts):
        if (
            not isinstance(raw_fact, list)
            or len(raw_fact) != 2
            or not isinstance(raw_fact[0], str)
            or not isinstance(raw_fact[1], int)
            or isinstance(raw_fact[1], bool)
            or raw_fact[1] < 0
        ):
            raise ValueError(
                f"{label}.supporting_facts[{fact_index}]: "
                "expected [title, non-negative sentence_id]"
            )
        supporting_facts.append((raw_fact[0], raw_fact[1]))

    question_type = value.get("type")
    level = value.get("level")
    if question_type is not None and not isinstance(question_type, str):
        raise ValueError(f"{label}.type: expected a string")
    if level is not None and not isinstance(level, str):
        raise ValueError(f"{label}.level: expected a string")
    return HotpotExample(
        example_id=example_id,
        question=question,
        answer=answer,
        supporting_facts=tuple(supporting_facts),
        context=tuple(context),
        question_type=question_type,
        level=level,
    )


def _required_string(value: Mapping[str, Any], key: str, label: str) -> str:
    field = value.get(key)
    if not isinstance(field, str) or not field:
        raise ValueError(f"{label}.{key}: expected a non-empty string")
    return field


def stratified_sample(
    examples: Sequence[HotpotExample],
    sample_size: int,
    *,
    seed: int,
    strata: tuple[str, ...] = ("question_type", "level"),
) -> list[HotpotExample]:
    """Take a deterministic, proportional sample over the requested strata.

    Allocation uses largest remainders after proportional quotas. Both strata
    and records are sorted before seeded shuffling, so results do not depend on
    the input sequence order.
    """

    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    if sample_size > len(examples):
        raise ValueError(
            f"sample_size ({sample_size}) exceeds dataset size ({len(examples)})"
        )
    if not strata:
        raise ValueError("at least one stratum field is required")
    allowed_fields = {"question_type", "level"}
    unknown = set(strata) - allowed_fields
    if unknown:
        raise ValueError(f"unsupported stratum fields: {sorted(unknown)}")
    if sample_size == 0:
        return []

    groups: dict[tuple[str | None, ...], list[HotpotExample]] = defaultdict(list)
    for example in examples:
        key = tuple(getattr(example, field) for field in strata)
        groups[key].append(example)

    ordered_keys = sorted(groups, key=lambda key: tuple(value or "" for value in key))
    total = len(examples)
    quotas = {
        key: sample_size * len(groups[key]) / total
        for key in ordered_keys
    }
    allocation = {key: int(quotas[key]) for key in ordered_keys}
    remaining = sample_size - sum(allocation.values())
    remainder_order = sorted(
        ordered_keys,
        key=lambda key: (-(quotas[key] - allocation[key]), ordered_keys.index(key)),
    )
    for key in remainder_order[:remaining]:
        allocation[key] += 1

    rng = random.Random(seed)
    result: list[HotpotExample] = []
    for key in ordered_keys:
        group = sorted(groups[key], key=lambda example: example.example_id)
        rng.shuffle(group)
        result.extend(group[: allocation[key]])
    rng.shuffle(result)
    return result


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation)


def normalize_answer(value: str) -> str:
    """Normalize an answer exactly as the official HotpotQA evaluator does."""

    lowered = value.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in _PUNCTUATION
    )
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def answer_scores(prediction: str, gold: str) -> dict[str, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    exact_match = float(normalized_prediction == normalized_gold)

    special_answers = {"yes", "no", "noanswer"}
    if (
        normalized_prediction in special_answers
        or normalized_gold in special_answers
    ) and normalized_prediction != normalized_gold:
        return {"em": exact_match, "f1": 0.0, "prec": 0.0, "recall": 0.0}

    prediction_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    common_count = sum(common.values())
    if common_count == 0:
        return {"em": exact_match, "f1": 0.0, "prec": 0.0, "recall": 0.0}
    precision = common_count / len(prediction_tokens)
    recall = common_count / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"em": exact_match, "f1": f1, "prec": precision, "recall": recall}


def supporting_fact_scores(
    prediction: Iterable[Sequence[Any]],
    gold: Iterable[Sequence[Any]],
) -> dict[str, float]:
    predicted_facts = {_fact_tuple(fact) for fact in prediction}
    gold_facts = {_fact_tuple(fact) for fact in gold}
    true_positive = len(predicted_facts & gold_facts)
    false_positive = len(predicted_facts - gold_facts)
    false_negative = len(gold_facts - predicted_facts)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "em": float(false_positive + false_negative == 0),
        "f1": f1,
        "prec": precision,
        "recall": recall,
    }


def _fact_tuple(value: Sequence[Any]) -> SupportingFact:
    if (
        isinstance(value, (str, bytes))
        or len(value) != 2
        or not isinstance(value[0], str)
        or not isinstance(value[1], int)
        or isinstance(value[1], bool)
    ):
        raise ValueError(f"invalid supporting fact: {value!r}")
    return value[0], value[1]


def evaluate_prediction(
    predicted_answer: str,
    predicted_supporting_facts: Iterable[Sequence[Any]],
    gold_answer: str,
    gold_supporting_facts: Iterable[Sequence[Any]],
) -> dict[str, float]:
    """Evaluate one answer and its supporting facts with official metrics."""

    answer = answer_scores(predicted_answer, gold_answer)
    supporting = supporting_fact_scores(
        predicted_supporting_facts, gold_supporting_facts
    )
    joint_precision = answer["prec"] * supporting["prec"]
    joint_recall = answer["recall"] * supporting["recall"]
    joint_f1 = (
        2 * joint_precision * joint_recall / (joint_precision + joint_recall)
        if joint_precision + joint_recall
        else 0.0
    )
    return {
        "em": answer["em"],
        "f1": answer["f1"],
        "prec": answer["prec"],
        "recall": answer["recall"],
        "sp_em": supporting["em"],
        "sp_f1": supporting["f1"],
        "sp_prec": supporting["prec"],
        "sp_recall": supporting["recall"],
        "joint_em": answer["em"] * supporting["em"],
        "joint_f1": joint_f1,
        "joint_prec": joint_precision,
        "joint_recall": joint_recall,
    }


def evaluate_predictions(
    examples: Sequence[HotpotExample],
    predictions: Mapping[str, Any],
) -> dict[str, float]:
    """Evaluate a prediction mapping and return macro-averaged metrics.

    ``predictions`` accepts either the official ``{"answer": {id: ...},
    "sp": {id: ...}}`` shape or ``{id: {"answer": ...,
    "supporting_facts": ...}}``. Missing predictions receive zero for all
    metrics, matching the official evaluator's denominator behavior.
    """

    metric_names = (
        "em", "f1", "prec", "recall", "sp_em", "sp_f1", "sp_prec",
        "sp_recall", "joint_em", "joint_f1", "joint_prec", "joint_recall",
    )
    totals = {name: 0.0 for name in metric_names}
    if not examples:
        return totals

    official_shape = isinstance(predictions.get("answer"), Mapping) or isinstance(
        predictions.get("sp"), Mapping
    )
    answer_map = predictions.get("answer", {}) if official_shape else {}
    fact_map = predictions.get("sp", {}) if official_shape else {}

    for example in examples:
        if example.answer is None:
            raise ValueError(f"example {example.example_id!r} has no gold answer")
        if official_shape:
            if example.example_id not in answer_map or example.example_id not in fact_map:
                continue
            predicted_answer = answer_map[example.example_id]
            predicted_facts = fact_map[example.example_id]
        else:
            prediction = predictions.get(example.example_id)
            if not isinstance(prediction, Mapping):
                continue
            if "answer" not in prediction:
                continue
            fact_key = (
                "supporting_facts"
                if "supporting_facts" in prediction
                else "sp"
            )
            if fact_key not in prediction:
                continue
            predicted_answer = prediction["answer"]
            predicted_facts = prediction[fact_key]
        if not isinstance(predicted_answer, str):
            raise ValueError(f"prediction for {example.example_id!r} has invalid answer")
        if not isinstance(predicted_facts, Iterable) or isinstance(
            predicted_facts, (str, bytes, Mapping)
        ):
            raise ValueError(
                f"prediction for {example.example_id!r} has invalid supporting facts"
            )
        scores = evaluate_prediction(
            predicted_answer,
            predicted_facts,
            example.answer,
            example.supporting_facts,
        )
        for name in metric_names:
            totals[name] += scores[name]

    return {name: total / len(examples) for name, total in totals.items()}
