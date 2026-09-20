from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentrix_application import PagedToolStore


class ToolError(RuntimeError):
    pass


class RepositoryTools:
    def __init__(
        self,
        workspace: Path,
        task: dict[str, Any],
        *,
        max_output_bytes: int = 32_768,
        result_store: PagedToolStore | None = None,
        session_id: str = "root",
    ) -> None:
        self.workspace = workspace.resolve()
        self.task = task
        self.max_output_bytes = max_output_bytes
        self.events: list[dict[str, Any]] = []
        self.result_store = result_store
        self.session_id = session_id

    def _path(self, relative: str) -> Path:
        candidate = (self.workspace / relative).resolve()
        if not candidate.is_relative_to(self.workspace):
            raise ToolError(f"path escapes workspace: {relative}")
        return candidate

    def _record(
        self, tool: str, arguments: dict[str, Any], content: str, started: float
    ) -> dict[str, Any]:
        full_encoded = content.encode("utf-8", errors="replace")
        original_bytes = len(full_encoded)
        content_sha256 = hashlib.sha256(full_encoded).hexdigest()
        encoded = full_encoded
        truncated = original_bytes > self.max_output_bytes and tool not in {
            "read_result",
            "search_result",
            "list_results",
        }
        paged = (
            self.result_store is not None
            and tool in {"read", "search"}
            and original_bytes > self.max_output_bytes
        )
        if paged:
            assert self.result_store is not None
            result_id = self.result_store.put(self.session_id, content)
            content = json.dumps(
                {
                    "result_id": result_id,
                    "total_chars": len(content),
                    "preview": content[:256],
                    "retrieval": "Use search_result with a literal needle, or "
                    "read_result with offset and limit, to read this exact snapshot.",
                },
                ensure_ascii=False,
            )
            encoded = content.encode("utf-8")
            truncated = False
        elif truncated:
            encoded = encoded[: self.max_output_bytes]
            content = encoded.decode("utf-8", errors="replace")
        event = {
            "sequence": len(self.events),
            "tool": tool,
            "arguments": arguments,
            "content": content,
            "content_sha256": content_sha256,
            "returned_sha256": hashlib.sha256(encoded).hexdigest(),
            "original_bytes": original_bytes,
            "returned_bytes": len(encoded),
            "truncated": truncated,
            "paged": paged,
            "wall_time_ms": (time.perf_counter() - started) * 1000,
        }
        self.events.append(event)
        return event

    def list_results(self, offset: int = 0, limit: int = 16) -> dict[str, Any]:
        """Find old handles even after their observations leave Agent history."""
        started = time.perf_counter()
        if self.result_store is None:
            raise ToolError("tool result paging is disabled")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        if type(limit) is not int or not 1 <= limit <= 32:
            raise ValueError("limit must be in 1..32 results")
        snapshots = [event for event in self.events if event["paged"]]
        entries = []
        for event in snapshots[offset : offset + limit]:
            handle = json.loads(event["content"])
            entries.append(
                dict(
                    sequence=event["sequence"],
                    tool=event["tool"],
                    arguments=event["arguments"],
                    result_id=handle["result_id"],
                    total_chars=handle["total_chars"],
                )
            )
        return self._record(
            "list_results",
            {"offset": offset, "limit": limit},
            json.dumps(
                dict(
                    results=entries,
                    next_offset=offset + len(entries),
                    total=len(snapshots),
                ),
                ensure_ascii=False,
            ),
            started,
        )

    def read_result(
        self, result_id: str, offset: int = 0, limit: int = 4096
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if self.result_store is None:
            raise ToolError("tool result paging is disabled")
        result = self.result_store.read(
            self.session_id, result_id, offset=offset, limit=limit
        )
        return self._record(
            "read_result",
            {"result_id": result_id, "offset": offset},
            json.dumps(result, ensure_ascii=False),
            started,
        )

    def search_result(
        self, result_id: str, needle: str, offset: int = 0
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if self.result_store is None:
            raise ToolError("tool result paging is disabled")
        result = self.result_store.search(
            self.session_id, result_id, needle, offset=offset
        )
        return self._record(
            "search_result",
            {"result_id": result_id, "needle": needle},
            json.dumps(result, ensure_ascii=False),
            started,
        )

    def search(self, pattern: str, glob: str = "*") -> dict[str, Any]:
        started = time.perf_counter()
        result = subprocess.run(
            (
                "rg",
                "-n",
                "--no-heading",
                "--color",
                "never",
                "--glob",
                glob,
                "--",
                pattern,
                ".",
            ),
            cwd=self.workspace,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise ToolError(result.stderr.strip())
        return self._record(
            "search", {"pattern": pattern, "glob": glob}, result.stdout, started
        )

    def read(
        self, path: str, start_line: int = 1, end_line: int = 400
    ) -> dict[str, Any]:
        started = time.perf_counter()
        source = self._path(path)
        if not source.is_file():
            raise ToolError(f"not a file: {path}")
        if start_line < 1 or end_line < start_line or end_line - start_line > 2000:
            raise ToolError("invalid line range")
        # Retain only the requested range. Still scan to EOF for the exact line
        # count, preserving read_text().splitlines() semantics, including Unicode
        # separators, universal newlines and an unterminated final line.
        parts = []
        line_number = 1
        at_line_start = True
        with source.open(encoding="utf-8", errors="replace") as stream:
            while block := stream.read(64 << 10):
                for fragment in block.splitlines(keepends=True):
                    ended = fragment[-1] in "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"
                    if start_line <= line_number <= end_line:
                        if at_line_start:
                            parts.append(f"{line_number}: ")
                        parts.append(fragment[:-1] if ended else fragment)
                        if ended:
                            parts.append("\n")
                    if ended:
                        line_number += 1
                    at_line_start = ended
        total_lines = line_number - 1 + int(not at_line_start)
        body = "".join(parts)
        if body.endswith("\n"):
            body = body[:-1]
        content = f"File {path} has {total_lines} lines.\n{body}"
        return self._record(
            "read",
            {"path": path, "start_line": start_line, "end_line": end_line},
            content,
            started,
        )

    def apply_patch(self, patch: str) -> dict[str, Any]:
        started = time.perf_counter()
        parsed = subprocess.run(
            ("git", "apply", "--numstat", "-"),
            cwd=self.workspace,
            input=patch,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if parsed.returncode != 0:
            raise ToolError(parsed.stderr.strip())
        paths = [line.split("\t", 2)[-1] for line in parsed.stdout.splitlines()]
        allowed = set(self.task["allowed_paths"])
        if not paths or not set(paths).issubset(allowed):
            raise ToolError(f"patch paths outside task scope: {paths}")
        checked = subprocess.run(
            ("git", "apply", "--check", "-"),
            cwd=self.workspace,
            input=patch,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if checked.returncode != 0:
            raise ToolError(checked.stderr.strip())
        subprocess.run(
            ("git", "apply", "-"),
            cwd=self.workspace,
            input=patch,
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )
        return self._record("apply_patch", {"paths": paths}, "patch applied", started)

    def diff(self) -> dict[str, Any]:
        started = time.perf_counter()
        result = subprocess.run(
            ("git", "diff", "--", *self.task["allowed_paths"]),
            cwd=self.workspace,
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )
        return self._record("diff", {}, result.stdout, started)

    def public_test(self) -> dict[str, Any]:
        started = time.perf_counter()
        for command in self.task.get("build", []):
            result = subprocess.run(
                tuple(
                    value.format(python=sys.executable, workspace=str(self.workspace))
                    for value in command["argv"]
                ),
                cwd=self._path(command.get("cwd", ".")),
                text=True,
                capture_output=True,
                timeout=self.task["timeout_seconds"],
                check=False,
            )
            if result.returncode != 0:
                return self._record(
                    "public_test",
                    {},
                    f"build failed ({result.returncode})\n{result.stderr}",
                    started,
                )
        result = subprocess.run(
            tuple(
                value.format(python=sys.executable, workspace=str(self.workspace))
                for value in self.task["public_test_command"]
            ),
            cwd=self.workspace,
            text=True,
            capture_output=True,
            timeout=self.task["timeout_seconds"],
            check=False,
        )
        content = json.dumps(
            {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            ensure_ascii=False,
        )
        return self._record("public_test", {}, content, started)
