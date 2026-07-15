from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class PromptSection:
    """One application-owned prompt section with a stable semantic identity."""

    segment_id: str
    content: str
    heading: str | None = None

    def render(self) -> str:
        if self.heading is None:
            return self.content
        return f"{self.heading}\n{self.content}"


@dataclass(frozen=True)
class CompactionReport:
    input_sections: int
    output_sections: int
    removed_empty_sections: int
    removed_duplicate_sections: int
    before_chars: int
    after_chars: int

    @property
    def saved_chars(self) -> int:
        return self.before_chars - self.after_chars


@dataclass(frozen=True)
class CompactedPrompt:
    text: str
    report: CompactionReport


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _known_fingerprints(sections: Iterable[PromptSection]) -> dict[str, str]:
    seen: dict[str, str] = {}
    for section in sections:
        if not section.content.strip():
            continue
        fingerprint = _digest(section.render())
        previous = seen.get(section.segment_id)
        if previous is not None and previous != fingerprint:
            raise ValueError(
                f"prompt segment {section.segment_id!r} has conflicting content"
            )
        seen[section.segment_id] = fingerprint
    return seen


def compact_prompt_delta(
    sections: Iterable[PromptSection],
    *,
    known_sections: Iterable[PromptSection] = (),
    separator: str = "\n\n",
) -> CompactedPrompt:
    """Compact new sections against byte-identical sections already in context."""

    source = list(sections)
    rendered_source = [section.render() for section in source]
    seen = _known_fingerprints(known_sections)
    output: list[str] = []
    removed_empty = 0
    removed_duplicate = 0

    for section, rendered in zip(source, rendered_source, strict=True):
        if not section.content.strip():
            removed_empty += 1
            continue
        fingerprint = _digest(rendered)
        previous = seen.get(section.segment_id)
        if previous is not None:
            if previous != fingerprint:
                raise ValueError(
                    f"prompt segment {section.segment_id!r} has conflicting content"
                )
            removed_duplicate += 1
            continue
        seen[section.segment_id] = fingerprint
        output.append(rendered)

    text = separator.join(output)
    return CompactedPrompt(
        text=text,
        report=CompactionReport(
            input_sections=len(source),
            output_sections=len(output),
            removed_empty_sections=removed_empty,
            removed_duplicate_sections=removed_duplicate,
            before_chars=len(separator.join(rendered_source)),
            after_chars=len(text),
        ),
    )


def compact_prompt_sections(
    sections: Iterable[PromptSection], *, separator: str = "\n\n"
) -> CompactedPrompt:
    """Compose application-owned sections without rewriting free-form text."""

    return compact_prompt_delta(sections, separator=separator)


def compact_json(value: Any) -> str:
    """Serialize structured prompt data without representation-only spaces."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def deduplicate_tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove canonical-equivalent tools and reject named schema conflicts."""

    output: list[dict[str, Any]] = []
    identities: dict[tuple[str, str], str] = {}
    fingerprints: set[str] = set()
    for tool in tools:
        canonical = compact_json(tool)
        fingerprint = _digest(canonical)
        function = tool.get("function")
        name = function.get("name", "") if isinstance(function, dict) else ""
        identity = (str(tool.get("type", "")), str(name))
        if name:
            previous = identities.get(identity)
            if previous is not None and previous != fingerprint:
                raise ValueError(
                    f"tool definition {identity!r} has conflicting schemas"
                )
            identities[identity] = fingerprint
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        output.append(tool)
    return output
