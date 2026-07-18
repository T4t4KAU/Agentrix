from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


TOOL_RESULT_STUB_PREFIX = "[Agentrix paged tool result] "


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


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


@dataclass(frozen=True)
class ToolResultCompactionConfig:
    """Conservative policy for paging old, recoverable tool results."""

    enabled: bool = False
    min_chars: int = 4096
    min_age_turns: int = 4
    recoverable_tools: tuple[str, ...] = ("read", "read_file")
    resource_argument_names: tuple[str, ...] = ("path", "file_path", "filename")

    def __post_init__(self) -> None:
        if self.min_chars < 1:
            raise ValueError("min_chars must be positive")
        if self.min_age_turns < 0:
            raise ValueError("min_age_turns must be non-negative")
        if not self.recoverable_tools:
            raise ValueError("recoverable_tools must not be empty")
        if not self.resource_argument_names:
            raise ValueError("resource_argument_names must not be empty")

    @classmethod
    def from_env(cls) -> "ToolResultCompactionConfig":
        tools = tuple(
            item.strip()
            for item in os.getenv(
                "AGENTRIX_PROMPT_COMPACTION_RECOVERABLE_TOOLS", "read,read_file"
            ).split(",")
            if item.strip()
        )
        return cls(
            enabled=_env_bool("AGENTRIX_PROMPT_TOOL_RESULT_COMPACTION_ENABLED", False),
            min_chars=int(
                os.getenv("AGENTRIX_PROMPT_COMPACTION_MIN_RESULT_CHARS", "4096")
            ),
            min_age_turns=int(
                os.getenv("AGENTRIX_PROMPT_COMPACTION_MIN_AGE_TURNS", "4")
            ),
            recoverable_tools=tools,
        )


@dataclass(frozen=True)
class PagedToolResult:
    """Auditable metadata for one tool result replaced by a retrieval handle."""

    message_index: int
    tool_call_id: str
    tool_name: str
    resource: str
    content_sha256: str
    original_chars: int
    original_lines: int
    age_turns: int


@dataclass(frozen=True)
class ToolResultCompactionReport:
    input_messages: int
    output_messages: int
    tool_results_seen: int
    compacted_results: int
    before_chars: int
    after_chars: int
    skipped_reasons: dict[str, int]
    paged_results: tuple[PagedToolResult, ...]

    @property
    def saved_chars(self) -> int:
        return self.before_chars - self.after_chars


@dataclass(frozen=True)
class CompactedMessages:
    messages: list[dict[str, Any]]
    report: ToolResultCompactionReport
    backing_store: ToolResultBackingStore


class ToolResultBackingStore:
    """Content-addressed storage for exact historical tool-result recovery."""

    def __init__(self) -> None:
        self._content: dict[str, str] = {}

    def put(self, content: str) -> str:
        digest = _digest(content)
        previous = self._content.get(digest)
        if previous is not None and previous != content:
            raise ValueError(f"SHA-256 collision for tool result {digest}")
        self._content[digest] = content
        return digest

    def get(self, digest: str) -> str:
        try:
            return self._content[digest]
        except KeyError as error:
            raise KeyError(f"tool result {digest} is not present in backing store") from error

    def __contains__(self, digest: object) -> bool:
        return digest in self._content

    def __len__(self) -> int:
        return len(self._content)

    @property
    def stored_chars(self) -> int:
        return sum(len(content) for content in self._content.values())


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


@dataclass(frozen=True)
class _ToolCall:
    name: str
    arguments: Mapping[str, Any]
    message_index: int


def _tool_calls_by_id(messages: Sequence[Mapping[str, Any]]) -> dict[str, _ToolCall]:
    calls: dict[str, _ToolCall] = {}
    for message_index, message in enumerate(messages):
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
            continue
        for raw_call in tool_calls:
            if not isinstance(raw_call, Mapping):
                continue
            call_id = raw_call.get("id")
            function = raw_call.get("function")
            if not isinstance(call_id, str) or not isinstance(function, Mapping):
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name:
                continue
            raw_arguments = function.get("arguments", {})
            if isinstance(raw_arguments, str):
                try:
                    parsed_arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    parsed_arguments = {}
            else:
                parsed_arguments = raw_arguments
            arguments = parsed_arguments if isinstance(parsed_arguments, Mapping) else {}
            call = _ToolCall(
                name=name,
                arguments=arguments,
                message_index=message_index,
            )
            previous = calls.get(call_id)
            if previous is not None and previous != call:
                raise ValueError(f"tool call ID {call_id!r} has conflicting definitions")
            calls[call_id] = call
    return calls


def _message_content_chars(messages: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        len(content)
        for message in messages
        if isinstance((content := message.get("content")), str)
    )


def _later_user_turns(messages: Sequence[Mapping[str, Any]]) -> list[int]:
    counts = [0] * len(messages)
    later_users = 0
    for index in range(len(messages) - 1, -1, -1):
        counts[index] = later_users
        if messages[index].get("role") == "user":
            later_users += 1
    return counts


def _has_later_assistant(
    messages: Sequence[Mapping[str, Any]], message_index: int
) -> bool:
    return any(message.get("role") == "assistant" for message in messages[message_index + 1 :])


def _looks_like_error(message: Mapping[str, Any], content: str) -> bool:
    if message.get("is_error") is True:
        return True
    status = message.get("status")
    if isinstance(status, str) and status.lower() in {"error", "failed", "failure"}:
        return True
    first_line = content.lstrip().splitlines()[0].lower() if content.strip() else ""
    return first_line.startswith(
        (
            "error:",
            "tool error:",
            "failed:",
            "failure:",
            "exception:",
            "traceback (most recent call last):",
            "permission denied",
            "no such file",
            "[errno",
        )
    )


def _resource_identity(
    call: _ToolCall, argument_names: Sequence[str]
) -> str | None:
    for name in argument_names:
        value = call.arguments.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _line_count(content: str) -> int:
    return len(content.splitlines())


def _tool_result_stub(page: PagedToolResult) -> str:
    return TOOL_RESULT_STUB_PREFIX + compact_json(
        {
            "chars": page.original_chars,
            "lines": page.original_lines,
            "recovery": "Re-run the same tool call if current content is needed.",
            "resource": page.resource,
            "sha256": page.content_sha256,
            "tool": page.tool_name,
            "version": 1,
        }
    )


def compact_tool_results(
    messages: Sequence[Mapping[str, Any]],
    *,
    config: ToolResultCompactionConfig | None = None,
    backing_store: ToolResultBackingStore | None = None,
) -> CompactedMessages:
    """Page only old, large, successful results from recoverable read tools.

    User and assistant text, tool invocations, message ordering, and tool-result
    envelopes are preserved. The original result body is retained by content
    hash in ``backing_store`` so the transformation is exactly reversible.
    """

    policy = config or ToolResultCompactionConfig.from_env()
    store = backing_store if backing_store is not None else ToolResultBackingStore()
    output = [copy.deepcopy(dict(message)) for message in messages]
    before_chars = _message_content_chars(messages)
    if not policy.enabled:
        return CompactedMessages(
            messages=output,
            report=ToolResultCompactionReport(
                input_messages=len(messages),
                output_messages=len(output),
                tool_results_seen=sum(
                    message.get("role") == "tool" for message in messages
                ),
                compacted_results=0,
                before_chars=before_chars,
                after_chars=before_chars,
                skipped_reasons={"disabled": 1},
                paged_results=(),
            ),
            backing_store=store,
        )

    calls = _tool_calls_by_id(messages)
    later_turns = _later_user_turns(messages)
    recoverable_tools = {name.casefold() for name in policy.recoverable_tools}
    skipped: dict[str, int] = {}
    pages: list[PagedToolResult] = []
    tool_results_seen = 0

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for index, (source, target) in enumerate(zip(messages, output, strict=True)):
        if source.get("role") != "tool":
            continue
        tool_results_seen += 1
        content = source.get("content")
        if not isinstance(content, str):
            skip("structured_content")
            continue
        if content.startswith(TOOL_RESULT_STUB_PREFIX):
            skip("already_paged")
            continue
        call_id = source.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            skip("missing_tool_call_id")
            continue
        call = calls.get(call_id)
        if call is None or call.message_index >= index:
            skip("unknown_tool_call")
            continue
        if call.name.casefold() not in recoverable_tools:
            skip("nonrecoverable_tool")
            continue
        resource = _resource_identity(call, policy.resource_argument_names)
        if resource is None:
            skip("missing_resource")
            continue
        if len(content) < policy.min_chars:
            skip("below_min_chars")
            continue
        if later_turns[index] < policy.min_age_turns:
            skip("too_recent")
            continue
        if not _has_later_assistant(messages, index):
            skip("not_consumed")
            continue
        if _looks_like_error(source, content):
            skip("error_result")
            continue

        digest = _digest(content)
        page = PagedToolResult(
            message_index=index,
            tool_call_id=call_id,
            tool_name=call.name,
            resource=resource,
            content_sha256=digest,
            original_chars=len(content),
            original_lines=_line_count(content),
            age_turns=later_turns[index],
        )
        stub = _tool_result_stub(page)
        if len(stub) >= len(content):
            skip("nonpositive_savings")
            continue
        stored_digest = store.put(content)
        if stored_digest != digest:
            raise AssertionError("backing store returned an unexpected digest")
        target["content"] = stub
        pages.append(page)

    after_chars = _message_content_chars(output)
    return CompactedMessages(
        messages=output,
        report=ToolResultCompactionReport(
            input_messages=len(messages),
            output_messages=len(output),
            tool_results_seen=tool_results_seen,
            compacted_results=len(pages),
            before_chars=before_chars,
            after_chars=after_chars,
            skipped_reasons=skipped,
            paged_results=tuple(pages),
        ),
        backing_store=store,
    )


def restore_tool_results(
    messages: Sequence[Mapping[str, Any]],
    backing_store: ToolResultBackingStore,
) -> list[dict[str, Any]]:
    """Restore every Agentrix tool-result stub to its exact historical body."""

    output = [copy.deepcopy(dict(message)) for message in messages]
    for message in output:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith(
            TOOL_RESULT_STUB_PREFIX
        ):
            continue
        encoded = content[len(TOOL_RESULT_STUB_PREFIX) :]
        try:
            metadata = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise ValueError("malformed Agentrix tool-result stub") from error
        if not isinstance(metadata, dict) or metadata.get("version") != 1:
            raise ValueError("unsupported Agentrix tool-result stub")
        digest = metadata.get("sha256")
        if not isinstance(digest, str):
            raise ValueError("Agentrix tool-result stub is missing sha256")
        restored = backing_store.get(digest)
        if _digest(restored) != digest:
            raise ValueError(f"backing-store content hash mismatch for {digest}")
        if metadata.get("chars") != len(restored):
            raise ValueError(f"backing-store character count mismatch for {digest}")
        if metadata.get("lines") != _line_count(restored):
            raise ValueError(f"backing-store line count mismatch for {digest}")
        message["content"] = restored
    return output
