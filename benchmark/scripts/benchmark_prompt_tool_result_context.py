#!/usr/bin/env python3
"""Measure prompt-token savings from conservative paging on real source reads."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "application" / "src"))

from agentrix_application import (  # noqa: E402
    ToolResultCompactionConfig,
    compact_tool_results,
    restore_tool_results,
)
from transformers import AutoTokenizer  # noqa: E402


REPOSITORIES = {
    "django": {
        "paths": (
            "django/db/transaction.py",
            "django/test/client.py",
            "django/http/multipartparser.py",
            "django/utils/module_loading.py",
            "django/contrib/sessions/backends/cached_db.py",
            "django/core/handlers/base.py",
            "django/core/handlers/wsgi.py",
            "django/db/backends/base/base.py",
            "django/utils/decorators.py",
            "django/utils/datastructures.py",
            "tests/transactions/tests.py",
            "tests/test_client/tests.py",
            "tests/requests_tests/tests.py",
            "tests/sessions_tests/tests.py",
            "tests/handlers/tests.py",
            "tests/utils_tests/test_module_loading.py",
        ),
    },
    "ffmpeg": {
        "paths": (
            "libavutil/bprint.c",
            "libavutil/dict.c",
            "libavutil/fifo.c",
            "libavutil/parseutils.c",
            "libavutil/frame.c",
            "libavutil/opt.c",
            "libavutil/avstring.c",
            "libavutil/buffer.c",
            "libavutil/log.c",
            "libavutil/mem.c",
            "libavutil/tests/bprint.c",
            "libavutil/tests/dict.c",
            "libavutil/tests/fifo.c",
            "libavutil/tests/parseutils.c",
            "libavutil/tests/opt.c",
            "libavutil/time.c",
        ),
    },
    "sqlite": {
        "paths": (
            "src/func.c",
            "src/printf.c",
            "src/sqliteInt.h",
            "src/utf.c",
            "src/util.c",
            "src/vdbe.c",
            "src/vdbeapi.c",
            "src/vdbemem.c",
            "src/pager.c",
            "src/btree.c",
            "src/select.c",
            "src/where.c",
            "src/expr.c",
            "src/os_unix.c",
            "test/func.test",
            "test/printf.test",
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/hwx/Documents/models/Qwen3-0.6B"),
    )
    parser.add_argument(
        "--django-repo",
        type=Path,
        default=Path("/home/hwx/Documents/codes/django"),
    )
    parser.add_argument(
        "--ffmpeg-repo",
        type=Path,
        default=Path("/home/hwx/Documents/codes/FFmpeg"),
    )
    parser.add_argument(
        "--sqlite-repo",
        type=Path,
        default=Path("/home/hwx/Documents/codes/sqlite"),
    )
    parser.add_argument(
        "--django-cases",
        type=Path,
        default=REPO_ROOT / "benchmark/data/django_agentrix/cases_30k_b16.jsonl",
    )
    parser.add_argument(
        "--ffmpeg-cases",
        type=Path,
        default=REPO_ROOT / "benchmark/data/ffmpeg_agentrix/cases_30k_b16.jsonl",
    )
    parser.add_argument(
        "--sqlite-cases",
        type=Path,
        default=REPO_ROOT / "benchmark/data/sqlite_agentrix/cases_30k_b16.jsonl",
    )
    parser.add_argument("--min-result-chars", type=int, default=4096)
    parser.add_argument("--min-age-turns", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_parent(path: Path) -> list[dict[str, Any]]:
    line = next(line for line in path.read_text(encoding="utf-8").splitlines() if line)
    payload = json.loads(line)
    return [dict(message) for message in payload["shared_messages"]]


def render_read_result(path: Path, relative: str, cap_bytes: int) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    body = "\n".join(
        f"{index}: {line}"
        for index, line in enumerate(lines[:400], 1)
    )
    content = f"File {relative} has {len(lines)} lines.\n{body}"
    encoded = content.encode("utf-8", errors="replace")[:cap_bytes]
    return encoded.decode("utf-8", errors="replace")


def build_suffix(
    repo: Path,
    paths: Sequence[str],
    *,
    read_count: int,
    cap_bytes: int,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for index, relative in enumerate(paths[:read_count]):
        call_id = f"read-{index}"
        arguments = {
            "path": relative,
            "start_line": 1,
            "end_line": 400,
        }
        messages.extend(
            (
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": json.dumps(arguments, sort_keys=True),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": render_read_result(
                        repo / relative,
                        relative,
                        cap_bytes,
                    ),
                },
                {
                    "role": "assistant",
                    "content": f"Inspected {relative} and retained the relevant findings.",
                },
            )
        )
    for turn in range(4):
        messages.extend(
            (
                {
                    "role": "user",
                    "content": f"Continue the investigation, follow-up turn {turn + 1}.",
                },
                {
                    "role": "assistant",
                    "content": f"Continued analysis for follow-up turn {turn + 1}.",
                },
            )
        )
    return messages


def token_count(tokenizer: Any, messages: Sequence[Mapping[str, Any]]) -> int:
    encoded = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
    )
    input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    return len(input_ids)


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    inputs = {
        "django": (args.django_repo, args.django_cases),
        "ffmpeg": (args.ffmpeg_repo, args.ffmpeg_cases),
        "sqlite": (args.sqlite_repo, args.sqlite_cases),
    }
    rows = []
    for repository, (repo_path, cases_path) in inputs.items():
        parent = load_parent(cases_path)
        parent_tokens = token_count(tokenizer, parent)
        paths = REPOSITORIES[repository]["paths"]
        missing = [relative for relative in paths if not (repo_path / relative).is_file()]
        if missing:
            raise FileNotFoundError(f"{repository} paths missing: {missing}")
        for cap_bytes in (4096, 32768):
            for read_count in (1, 2, 4, 8, 16):
                suffix = build_suffix(
                    repo_path,
                    paths,
                    read_count=read_count,
                    cap_bytes=cap_bytes,
                )
                raw_messages = [*parent, *suffix]
                compacted = compact_tool_results(
                    raw_messages,
                    config=ToolResultCompactionConfig(
                        enabled=True,
                        min_chars=args.min_result_chars,
                        min_age_turns=args.min_age_turns,
                    ),
                )
                restored = restore_tool_results(
                    compacted.messages,
                    compacted.backing_store,
                )
                raw_tokens = token_count(tokenizer, raw_messages)
                compacted_tokens = token_count(tokenizer, compacted.messages)
                raw_suffix_tokens = raw_tokens - parent_tokens
                compacted_suffix_tokens = compacted_tokens - parent_tokens
                rows.append(
                    {
                        "repository": repository,
                        "read_cap_bytes": cap_bytes,
                        "read_count": read_count,
                        "parent_tokens": parent_tokens,
                        "raw_tokens": raw_tokens,
                        "compacted_tokens": compacted_tokens,
                        "saved_tokens": raw_tokens - compacted_tokens,
                        "total_token_reduction": safe_ratio(
                            raw_tokens - compacted_tokens,
                            raw_tokens,
                        ),
                        "raw_suffix_tokens": raw_suffix_tokens,
                        "compacted_suffix_tokens": compacted_suffix_tokens,
                        "saved_suffix_tokens": raw_suffix_tokens - compacted_suffix_tokens,
                        "suffix_token_reduction": safe_ratio(
                            raw_suffix_tokens - compacted_suffix_tokens,
                            raw_suffix_tokens,
                        ),
                        "tool_results_seen": compacted.report.tool_results_seen,
                        "compacted_results": compacted.report.compacted_results,
                        "saved_chars": compacted.report.saved_chars,
                        "stored_chars": compacted.backing_store.stored_chars,
                        "exact_roundtrip": restored == raw_messages,
                        "skipped_reasons": compacted.report.skipped_reasons,
                    }
                )

    report = {
        "schema_version": 1,
        "model": str(args.model),
        "tokenizer": tokenizer.__class__.__name__,
        "read_format": "RepositoryTools.read-compatible first 400 lines",
        "read_caps_bytes": [4096, 32768],
        "read_counts": [1, 2, 4, 8, 16],
        "min_result_chars": args.min_result_chars,
        "min_age_turns": args.min_age_turns,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
