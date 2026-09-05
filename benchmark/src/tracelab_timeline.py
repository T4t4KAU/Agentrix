"""Observed input-ready timestamps and reproducible TraceLab time windows."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

INPUT_EVENTS = {"user_message", "tool_result"}
OUTPUT_EVENTS = {"reasoning", "text", "tool_call"}


def timestamp_ms(value: Any) -> int | None:
    """Parse an explicitly zoned timestamp without guessing the source timezone."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return round(parsed.timestamp() * 1000) if parsed.tzinfo else None
    except (ValueError, OverflowError):
        return None


def input_ready_ms(row: dict[str, Any]) -> int | None:
    """Match TraceLab's latest input preceding the first model-output event.

    This is an observable arrival proxy, not a server receipt timestamp.
    Usage reports alone do not establish when model output began.
    """
    inputs, outputs = [], []
    for event in row.get("timing_events", []):
        stamp = timestamp_ms(event.get("timestamp"))
        if stamp is None:
            continue
        if event.get("event_type") in INPUT_EVENTS:
            inputs.append(stamp)
        elif event.get("event_type") in OUTPUT_EVENTS:
            outputs.append(stamp)
    if not inputs or not outputs:
        return None
    first_output = min(outputs)
    return max((stamp for stamp in inputs if stamp <= first_output), default=None)


def select_timeline(
    sessions: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    window_seconds: float,
    copies: int,
    max_model_len: int,
    window_start_ms: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a dense real-time window, then overlay independent load copies.

    Selection maximizes distinct eligible sessions, then requests, with the
    earliest window breaking ties. No time compression or provider balancing
    is performed. Missing/out-of-window rounds break synthetic context chains.
    """
    events = []
    rejected: Counter[str] = Counter()
    for identity, rows in sorted(sessions.items()):
        ordered = sorted(
            rows, key=lambda row: (row["source_round_index"], row["source_row"])
        )
        for sequence, row in enumerate(ordered):
            if row["input_ready_ms"] is None:
                rejected["missing_input_ready_time"] += 1
            elif (
                row["prefix_len"] + row["input_len"] + row["output_len"] > max_model_len
            ):
                rejected["context_limit"] += 1
            else:
                events.append((row["input_ready_ms"], identity, sequence, row))
    events.sort(key=lambda event: (event[0], event[1], event[2]))
    if not events:
        raise ValueError("No timed requests fit the model context limit")
    duration_ms = round(window_seconds * 1000)
    if duration_ms <= 0 or copies < 1:
        raise ValueError("Window duration and load copies must be positive")
    if window_start_ms is None:
        right = 0
        counts: Counter[tuple[str, str]] = Counter()
        best_score = (0, 0)
        for left, event in enumerate(events):
            while right < len(events) and events[right][0] < event[0] + duration_ms:
                counts[events[right][1]] += 1
                right += 1
            score = (len(counts), right - left)
            if score > best_score:
                best_score = score
                window_start_ms = event[0]
            counts[event[1]] -= 1
            if not counts[event[1]]:
                del counts[event[1]]
    assert window_start_ms is not None
    selected: dict[tuple[str, str], list] = defaultdict(list)
    for stamp, identity, sequence, row in events:
        if window_start_ms <= stamp < window_start_ms + duration_ms:
            selected[identity].append((sequence, row))
    if not selected:
        raise ValueError("The requested time window contains no eligible requests")
    result = []
    for copy in range(copies):
        for (provider, session_id), rows in sorted(selected.items()):
            rounds = []
            previous_sequence = None
            for sequence, row in rows:
                rounds.append(
                    dict(
                        row,
                        arrival_time_ms=row["input_ready_ms"] - window_start_ms,
                        context_reset=previous_sequence is None
                        or sequence != previous_sequence + 1,
                    )
                )
                previous_sequence = sequence
            result.append(
                {
                    "provider": provider,
                    "source_session_id": session_id,
                    "session_id": f"{provider}:{session_id}:copy-{copy}",
                    "load_copy": copy,
                    "arrival_time_ms": rounds[0]["arrival_time_ms"],
                    "rounds": rounds,
                }
            )
    return result, {
        "timing_mode": "trace_open_loop",
        "arrival_proxy": "latest_input_before_first_model_output",
        "time_scale": 1.0,
        "load_copies": copies,
        "window_start_utc": datetime.fromtimestamp(
            window_start_ms / 1000, timezone.utc
        ).isoformat(),
        "window_seconds": window_seconds,
        "window_source_sessions": len(selected),
        "window_source_requests": sum(map(len, selected.values())),
        "eligible_timed_requests": len(events),
        "rejected_requests": dict(rejected),
        "prompt_mode": "synthetic_previous_prompt_without_generated_output",
    }
