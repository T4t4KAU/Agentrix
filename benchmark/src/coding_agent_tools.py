from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class ToolError(RuntimeError):
    pass


class RepositoryTools:
    def __init__(
        self,
        workspace: Path,
        task: dict[str, Any],
        *,
        max_output_bytes: int = 32_768,
    ) -> None:
        self.workspace = workspace.resolve()
        self.task = task
        self.max_output_bytes = max_output_bytes
        self.events: list[dict[str, Any]] = []

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
        truncated = original_bytes > self.max_output_bytes
        if truncated:
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
            "wall_time_ms": (time.perf_counter() - started) * 1000,
        }
        self.events.append(event)
        return event

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

    def read(self, path: str, start_line: int = 1, end_line: int = 400) -> dict[str, Any]:
        started = time.perf_counter()
        source = self._path(path)
        if not source.is_file():
            raise ToolError(f"not a file: {path}")
        if start_line < 1 or end_line < start_line or end_line - start_line > 2000:
            raise ToolError("invalid line range")
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
        body = "\n".join(
            f"{index}: {line}"
            for index, line in enumerate(lines[start_line - 1 : end_line], start_line)
        )
        content = f"File {path} has {len(lines)} lines.\n{body}"
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
        return self._record(
            "apply_patch", {"paths": paths}, "patch applied", started
        )

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
                tuple(command["argv"]),
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
                value.format(
                    python=sys.executable, workspace=str(self.workspace)
                )
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
