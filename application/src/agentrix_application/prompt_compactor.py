from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
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
            raise KeyError(
                f"tool result {digest} is not present in backing store"
            ) from error

    def __contains__(self, digest: object) -> bool:
        return digest in self._content

    def __len__(self) -> int:
        return len(self._content)

    @property
    def stored_chars(self) -> int:
        return sum(len(content) for content in self._content.values())


class PagedToolStore:
    """Immutable tool snapshots with bounded reads and session-owned references.

    Forking copies references, never result bodies. Changed snapshots share
    identical pages. Releasing the last owner of a page reclaims its storage.
    One owner thread must serialize calls (for example an Agent event loop).
    """

    PAGE_CHARS = 4096

    def __init__(self, path: Path, *, max_bytes: int = 1 << 30) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            PRAGMA foreign_keys=ON;
            PRAGMA auto_vacuum=FULL;
            PRAGMA cache_size=-2048;
            PRAGMA mmap_size=0;
            CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS objects (
                id TEXT PRIMARY KEY, chars INTEGER NOT NULL, bytes INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, content TEXT NOT NULL, bytes INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pages (
                object_id TEXT REFERENCES objects(id) ON DELETE CASCADE,
                number INTEGER, chunk_id TEXT REFERENCES chunks(id),
                PRIMARY KEY (object_id, number)
            );
            CREATE TABLE IF NOT EXISTS refs (
                session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
                object_id TEXT REFERENCES objects(id),
                PRIMARY KEY (session_id, object_id)
            );
            CREATE INDEX IF NOT EXISTS refs_object ON refs(object_id);
            CREATE TABLE IF NOT EXISTS observations (
                session_id TEXT, sequence INTEGER, object_id TEXT, label TEXT,
                PRIMARY KEY (session_id, sequence),
                FOREIGN KEY (session_id, object_id)
                    REFERENCES refs(session_id, object_id) ON DELETE CASCADE
            );
        """)
        # Preserve existing handles, sessions and catalogs when opening an old
        # database whose pages still contain private copies of the text.
        try:
            with self.db:
                self.db.execute("BEGIN")
                if "content" in {
                    row[1] for row in self.db.execute("PRAGMA table_info(pages)")
                }:
                    self.db.execute("""CREATE TABLE shared_pages (
                        object_id TEXT REFERENCES objects(id) ON DELETE CASCADE,
                        number INTEGER, chunk_id TEXT REFERENCES chunks(id),
                        PRIMARY KEY (object_id, number))""")
                    for object_id, number, content in self.db.execute(
                        "SELECT object_id, number, content FROM pages"
                    ):
                        data = content.encode("utf-8")
                        chunk_id = hashlib.sha256(data).hexdigest()
                        self.db.execute(
                            "INSERT OR IGNORE INTO chunks VALUES (?, ?, ?)",
                            (chunk_id, content, len(data)),
                        )
                        self.db.execute(
                            "INSERT INTO shared_pages VALUES (?, ?, ?)",
                            (object_id, number, chunk_id),
                        )
                    self.db.execute("DROP TABLE pages")
                    self.db.execute("ALTER TABLE shared_pages RENAME TO pages")
                self.db.execute(
                    "CREATE INDEX IF NOT EXISTS pages_chunk ON pages(chunk_id)"
                )
        except BaseException:
            self.db.close()
            raise

    def open_session(self, session_id: str, *, parent: str | None = None) -> None:
        if not isinstance(session_id, str) or not 0 < len(session_id) <= 128:
            raise ValueError("session_id must contain 1..128 characters")
        with self.db:
            if parent is not None:
                self._require_session(parent)
            self.db.execute("INSERT INTO sessions VALUES (?)", (session_id,))
            if parent is not None:
                self.db.execute(
                    "INSERT INTO refs SELECT ?, object_id FROM refs WHERE session_id=?",
                    (session_id, parent),
                )
                self.db.execute(
                    "INSERT INTO observations SELECT ?, sequence, object_id, label "
                    "FROM observations WHERE session_id=?",
                    (session_id, parent),
                )

    def _require_session(self, session_id: str) -> None:
        if (
            self.db.execute(
                "SELECT 1 FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            is None
        ):
            raise KeyError(f"unknown tool-result session {session_id!r}")

    def put(self, session_id: str, content: str) -> str:
        return self.put_stream(session_id, (content,))

    def put_stream(self, session_id: str, pieces: Iterable[str]) -> str:
        """Store a stream without assembling its complete body in host memory.

        The quota charges unique UTF-8 pages. An interrupted stream or quota
        failure rolls back every new page, mapping and session reference.
        Producers must not mutate this store while yielding pieces.
        """

        def pages():
            pending = ""
            for piece in pieces:
                if not isinstance(piece, str):
                    raise TypeError("tool-result pieces must be strings")
                for start in range(0, len(piece), self.PAGE_CHARS):
                    pending += piece[start : start + self.PAGE_CHARS]
                    if len(pending) >= self.PAGE_CHARS:
                        yield pending[: self.PAGE_CHARS]
                        pending = pending[self.PAGE_CHARS :]
            if pending:
                yield pending

        with self.db:
            self._require_session(session_id)
            temporary = uuid.uuid4().hex
            self.db.execute("INSERT INTO objects VALUES (?, 0, 0)", (temporary,))
            used = self.stats()["stored_bytes"]
            hasher, chars, size = hashlib.sha256(), 0, 0
            for number, page in enumerate(pages()):
                data = page.encode("utf-8")
                hasher.update(data)
                chars += len(page)
                size += len(data)
                chunk_id = hashlib.sha256(data).hexdigest()
                if (
                    self.db.execute(
                        "SELECT 1 FROM chunks WHERE id=?", (chunk_id,)
                    ).fetchone()
                    is None
                ):
                    if used + len(data) > self.max_bytes:
                        raise ValueError("tool-result storage budget exceeded")
                    self.db.execute(
                        "INSERT INTO chunks VALUES (?, ?, ?)",
                        (chunk_id, page, len(data)),
                    )
                    used += len(data)
                self.db.execute(
                    "INSERT INTO pages VALUES (?, ?, ?)", (temporary, number, chunk_id)
                )
            digest = hasher.hexdigest()
            if (
                self.db.execute(
                    "SELECT 1 FROM objects WHERE id=?", (digest,)
                ).fetchone()
                is None
            ):
                self.db.execute(
                    "INSERT INTO objects VALUES (?, ?, ?)", (digest, chars, size)
                )
                self.db.execute(
                    "UPDATE pages SET object_id=? WHERE object_id=?",
                    (digest, temporary),
                )
            self.db.execute("DELETE FROM objects WHERE id=?", (temporary,))
            self.db.execute(
                "INSERT OR IGNORE INTO refs VALUES (?, ?)", (session_id, digest)
            )
        return digest

    def replace_range(
        self,
        session_id: str,
        result_id: str,
        *,
        offset: int,
        delete_chars: int,
        content: str,
    ) -> str:
        """Create a private immutable revision using bounded host memory.

        Equal-length edits and appends reuse unchanged pages. Insertions that
        shift page boundaries may need new pages throughout the changed suffix.
        The caller keeps both handles until the owning session is released.
        """
        total = self.read(session_id, result_id, offset=offset, limit=1)["total_chars"]
        if type(delete_chars) is not int or not 0 <= delete_chars <= total - offset:
            raise ValueError("delete_chars exceeds the available character range")
        if not isinstance(content, str):
            raise TypeError("replacement content must be a string")

        def pieces():
            for start in range(0, offset, self.PAGE_CHARS):
                yield self.read(
                    session_id,
                    result_id,
                    offset=start,
                    limit=min(self.PAGE_CHARS, offset - start),
                )["content"]
            yield content
            for start in range(offset + delete_chars, total, self.PAGE_CHARS):
                yield self.read(
                    session_id,
                    result_id,
                    offset=start,
                    limit=min(self.PAGE_CHARS, total - start),
                )["content"]

        return self.put_stream(session_id, pieces())

    def read(
        self, session_id: str, result_id: str, *, offset: int = 0, limit: int = 4096
    ) -> dict[str, Any]:
        """Read a character range from an owned historical snapshot."""
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        if type(limit) is not int or not 1 <= limit <= 16384:
            raise ValueError("limit must be in 1..16384 characters")
        row = self.db.execute(
            "SELECT chars FROM objects JOIN refs ON objects.id=refs.object_id "
            "WHERE session_id=? AND objects.id=?",
            (session_id, result_id),
        ).fetchone()
        if row is None:
            raise KeyError("result is not owned by this session")
        total = row[0]
        if offset > total:
            raise ValueError("offset exceeds result length")
        end = min(total, offset + limit)
        pages = self.db.execute(
            "SELECT content FROM pages JOIN chunks ON chunks.id=pages.chunk_id "
            "WHERE object_id=? AND number BETWEEN ? "
            "AND ? ORDER BY number",
            (
                result_id,
                offset // self.PAGE_CHARS,
                max(offset, end - 1) // self.PAGE_CHARS,
            ),
        )
        content = "".join(page[0] for page in pages)
        begin = offset % self.PAGE_CHARS
        return dict(
            result_id=result_id,
            offset=offset,
            next_offset=end,
            total_chars=total,
            eof=end == total,
            content=content[begin : begin + end - offset],
        )

    def search(
        self,
        session_id: str,
        result_id: str,
        needle: str,
        *,
        offset: int = 0,
        limit: int = 4096,
    ) -> dict[str, Any]:
        """Find a literal string without materializing the complete result."""
        if not isinstance(needle, str) or not 1 <= len(needle) <= 256:
            raise ValueError("needle must contain 1..256 characters")
        # Validate the response bounds even when the search has no match.
        page = self.read(session_id, result_id, offset=offset, limit=limit)
        while True:
            scan = self.read(
                session_id, result_id, offset=offset, limit=self.PAGE_CHARS
            )
            index = scan["content"].find(needle)
            if index >= 0:
                position = offset + index
                return self.read(
                    session_id,
                    result_id,
                    offset=max(0, position - min(128, max(0, limit - len(needle)))),
                    limit=limit,
                ) | {"match_offset": position}
            if scan["eof"]:
                return dict(
                    result_id=result_id,
                    match_offset=None,
                    total_chars=page["total_chars"],
                    content="",
                )
            offset = scan["next_offset"] - len(needle) + 1

    def release_session(self, session_id: str) -> None:
        with self.db:
            self.db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            self.db.execute(
                "DELETE FROM objects WHERE NOT EXISTS "
                "(SELECT 1 FROM refs WHERE refs.object_id=objects.id)"
            )
            self.db.execute(
                "DELETE FROM chunks WHERE NOT EXISTS "
                "(SELECT 1 FROM pages WHERE pages.chunk_id=chunks.id)"
            )

    def record_observation(self, session_id: str, label: str, content: str) -> int:
        """Archive a tool observation; keep its body out of in-memory history."""
        if not isinstance(label, str) or not 1 <= len(label) <= 256:
            raise ValueError("observation label must contain 1..256 characters")
        digest = self.put(session_id, content)
        with self.db:
            sequence = self.db.execute(
                "SELECT COALESCE(MAX(sequence), -1) + 1 FROM observations "
                "WHERE session_id=?",
                (session_id,),
            ).fetchone()[0]
            self.db.execute(
                "INSERT INTO observations VALUES (?, ?, ?, ?)",
                (session_id, sequence, digest, label),
            )
        return sequence

    def list_observations(
        self, session_id: str, *, offset: int = 0, limit: int = 16
    ) -> list[dict[str, Any]]:
        """Page through archived tool metadata without loading result bodies."""
        self._require_session(session_id)
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        if type(limit) is not int or not 1 <= limit <= 64:
            raise ValueError("limit must be in 1..64 observations")
        rows = self.db.execute(
            "SELECT sequence, object_id, label, chars FROM observations "
            "JOIN objects ON objects.id=object_id WHERE session_id=? "
            "AND sequence>=? ORDER BY sequence LIMIT ?",
            (session_id, offset, limit),
        )
        return [
            dict(sequence=n, result_id=key, label=label, total_chars=chars)
            for n, key, label, chars in rows
        ]

    def render_context(
        self,
        session_id: str,
        *,
        count_tokens: Callable[[str], int],
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """Render recent tool evidence within a cumulative token budget.

        The bound includes catalog metadata and JSON framing, using the caller's
        model tokenizer. It excludes other messages and chat-template overhead.
        Older observations remain discoverable with list_observations and can
        be read again by handle. The newest oversized result is an exact prefix,
        explicitly marked partial; nothing is summarized or silently truncated.
        None renders all observations for the unbounded benchmark baseline.
        """
        self._require_session(session_id)
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
            raise ValueError("max_tokens must be positive or None")
        total = self.db.execute(
            "SELECT COUNT(*) FROM observations WHERE session_id=?", (session_id,)
        ).fetchone()[0]

        def encode(entries):
            return json.dumps(
                dict(
                    observation_count=total,
                    archived_count=total - len(entries),
                    recent_results=list(reversed(entries)),
                ),
                ensure_ascii=False,
                separators=(",", ":"),
            )

        entries: list[dict[str, Any]] = []
        text = encode(entries)
        if max_tokens is not None and count_tokens(text) > max_tokens:
            raise ValueError("tool-context budget cannot fit catalog metadata")
        rows = self.db.execute(
            "SELECT sequence, object_id, label, chars FROM observations "
            "JOIN objects ON objects.id=object_id WHERE session_id=? "
            "ORDER BY sequence DESC",
            (session_id,),
        )
        for sequence, digest, label, chars in rows:
            # Bound temporary host memory too; large bodies stay on disk.
            limit = chars if max_tokens is None else min(chars, 16384)
            content = "".join(
                self.read(
                    session_id, digest, offset=start, limit=min(16384, limit - start)
                )["content"]
                for start in range(0, limit, 16384)
            )
            entry = dict(
                sequence=sequence,
                result_id=digest,
                label=label,
                total_chars=chars,
                next_offset=len(content),
                partial=len(content) < chars,
                content=content,
            )
            candidate = encode([*entries, entry])
            if max_tokens is not None and count_tokens(candidate) > max_tokens:
                if entries:
                    break
                # Keep a bounded exact prefix of the most recent result.
                # Token lengths need not be monotone; every accepted candidate
                # is measured, so the hard bound does not depend on monotonicity.
                low, high = 0, len(content)
                fitted = None
                while low <= high:
                    middle = (low + high) // 2
                    trial = dict(
                        entry,
                        content=content[:middle],
                        next_offset=middle,
                        partial=middle < chars,
                    )
                    candidate = encode([trial])
                    if count_tokens(candidate) <= max_tokens:
                        fitted = trial
                        low = middle + 1
                    else:
                        high = middle - 1
                if fitted is None:
                    raise ValueError(
                        "tool-context budget cannot fit newest result handle"
                    )
                entry = fitted
            entries.append(entry)
            text = encode(entries)
            if entry["partial"]:
                break
        return dict(
            text=text,
            tokens=count_tokens(text),
            resident_observations=len(entries),
            total_observations=total,
            archived_observations=total - len(entries),
            partial_observations=sum(row["partial"] for row in entries),
        )

    def stats(self) -> dict[str, int]:
        count = self.db.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
        size = self.db.execute("SELECT COALESCE(SUM(bytes), 0) FROM chunks").fetchone()[
            0
        ]
        return dict(
            objects=count,
            stored_bytes=size,
            sessions=self.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
            references=self.db.execute("SELECT COUNT(*) FROM refs").fetchone()[0],
        )

    def close(self) -> None:
        self.db.close()


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
            arguments = (
                parsed_arguments if isinstance(parsed_arguments, Mapping) else {}
            )
            call = _ToolCall(
                name=name,
                arguments=arguments,
                message_index=message_index,
            )
            previous = calls.get(call_id)
            if previous is not None and previous != call:
                raise ValueError(
                    f"tool call ID {call_id!r} has conflicting definitions"
                )
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
    return any(
        message.get("role") == "assistant" for message in messages[message_index + 1 :]
    )


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


def _resource_identity(call: _ToolCall, argument_names: Sequence[str]) -> str | None:
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
