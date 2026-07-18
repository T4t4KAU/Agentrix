#!/usr/bin/env python3
"""Audit conservative tool-result compaction on captured chat sessions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "application" / "src"))

from agentrix_application import (  # noqa: E402
    PagedToolResult,
    ToolResultBackingStore,
    ToolResultCompactionConfig,
    compact_tool_results,
    restore_tool_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--min-chars", type=int, default=4096)
    parser.add_argument("--min-age-turns", type=int, default=4)
    parser.add_argument(
        "--recoverable-tools",
        default="read,read_file",
        help="Comma-separated tool names whose results can be re-read.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _is_message(value: Any) -> bool:
    return isinstance(value, Mapping) and isinstance(value.get("role"), str)


def _sessions_from_value(value: Any) -> Iterable[list[dict[str, Any]]]:
    if isinstance(value, list) and all(_is_message(item) for item in value):
        yield [dict(item) for item in value]
        return
    if not isinstance(value, Mapping):
        return
    messages = value.get("messages")
    if isinstance(messages, list) and all(_is_message(item) for item in messages):
        yield [dict(item) for item in messages]
    sessions = value.get("sessions")
    if isinstance(sessions, list):
        for session in sessions:
            yield from _sessions_from_value(session)


def load_sessions(path: Path) -> list[list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        values = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]
        if values and all(_is_message(value) for value in values):
            return [[dict(value) for value in values]]
        sessions: list[list[dict[str, Any]]] = []
        for value in values:
            sessions.extend(_sessions_from_value(value))
        return sessions
    return list(_sessions_from_value(value))


def _arguments(call: Mapping[str, Any]) -> Mapping[str, Any]:
    function = call.get("function")
    if not isinstance(function, Mapping):
        return {}
    raw = function.get("arguments", {})
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, Mapping) else {}


def _resource(
    call: Mapping[str, Any], argument_names: Sequence[str]
) -> tuple[str, str] | None:
    function = call.get("function")
    if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
        return None
    for argument_name in argument_names:
        value = _arguments(call).get(argument_name)
        if isinstance(value, str) and value.strip():
            return str(function["name"]).casefold(), value.strip()
    return None


def potential_faults(
    messages: Sequence[Mapping[str, Any]],
    pages: Sequence[PagedToolResult],
    argument_names: Sequence[str],
) -> int:
    faults = 0
    for page in pages:
        target = page.tool_name.casefold(), page.resource
        found = False
        for message in messages[page.message_index + 1 :]:
            raw_calls = message.get("tool_calls")
            if not isinstance(raw_calls, list):
                continue
            for call in raw_calls:
                if isinstance(call, Mapping) and _resource(call, argument_names) == target:
                    found = True
                    break
            if found:
                break
        faults += found
    return faults


def binomial_upper_95(successes: int, trials: int) -> float | None:
    if trials == 0:
        return None
    if successes == 0:
        return 1.0 - math.pow(0.05, 1.0 / trials)
    z = 1.959963984540054
    probability = successes / trials
    denominator = 1 + z * z / trials
    center = probability + z * z / (2 * trials)
    radius = z * math.sqrt(
        probability * (1 - probability) / trials + z * z / (4 * trials * trials)
    )
    return min(1.0, (center + radius) / denominator)


def evaluate(
    sessions: Sequence[Sequence[Mapping[str, Any]]],
    config: ToolResultCompactionConfig,
) -> dict[str, Any]:
    totals = {
        "sessions": len(sessions),
        "messages": 0,
        "tool_results_seen": 0,
        "compacted_results": 0,
        "before_chars": 0,
        "after_chars": 0,
        "potential_faults": 0,
        "roundtrip_failures": 0,
    }
    skipped_reasons: dict[str, int] = {}
    for raw_messages in sessions:
        messages = [dict(message) for message in raw_messages]
        store = ToolResultBackingStore()
        compacted = compact_tool_results(
            messages,
            config=config,
            backing_store=store,
        )
        restored = restore_tool_results(compacted.messages, store)
        totals["messages"] += len(messages)
        totals["tool_results_seen"] += compacted.report.tool_results_seen
        totals["compacted_results"] += compacted.report.compacted_results
        totals["before_chars"] += compacted.report.before_chars
        totals["after_chars"] += compacted.report.after_chars
        totals["potential_faults"] += potential_faults(
            messages,
            compacted.report.paged_results,
            config.resource_argument_names,
        )
        totals["roundtrip_failures"] += restored != messages
        for reason, count in compacted.report.skipped_reasons.items():
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + count

    compacted_results = totals["compacted_results"]
    before_chars = totals["before_chars"]
    fault_rate = (
        totals["potential_faults"] / compacted_results if compacted_results else None
    )
    return {
        "schema_version": 1,
        "policy": {
            "min_chars": config.min_chars,
            "min_age_turns": config.min_age_turns,
            "recoverable_tools": list(config.recoverable_tools),
        },
        **totals,
        "saved_chars": before_chars - totals["after_chars"],
        "saved_char_ratio": (
            (before_chars - totals["after_chars"]) / before_chars
            if before_chars
            else None
        ),
        "potential_fault_rate": fault_rate,
        "potential_fault_rate_upper_95": binomial_upper_95(
            totals["potential_faults"], compacted_results
        ),
        "exact_roundtrip": totals["roundtrip_failures"] == 0,
        "skipped_reasons": skipped_reasons,
    }


def main() -> None:
    args = parse_args()
    sessions = [
        session
        for path in args.inputs
        for session in load_sessions(path)
    ]
    if not sessions:
        raise SystemExit("no chat message sessions found in inputs")
    tools = tuple(
        tool.strip() for tool in args.recoverable_tools.split(",") if tool.strip()
    )
    report = evaluate(
        sessions,
        ToolResultCompactionConfig(
            enabled=True,
            min_chars=args.min_chars,
            min_age_turns=args.min_age_turns,
            recoverable_tools=tools,
        ),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not report["exact_roundtrip"]:
        raise SystemExit("exact round-trip validation failed")


if __name__ == "__main__":
    main()
