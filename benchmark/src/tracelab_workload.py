"""Deterministic, bounded windows from the public TraceLab coding trace."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tracelab_timeline import input_ready_ms, select_timeline, timestamp_ms

RELEASE_SHA256 = "9d265eae69a31cae203848bea936f018148eed7ca8bf56050c5abe96da0b4e6b"
RELEASE_URL = (
    "https://github.com/uw-syfi/TraceLab/releases/download/v0.0.1/"
    "syfi_coding_trace.jsonl.gz"
)


def normalize_round(row: dict[str, Any], ordinal: int) -> dict[str, Any]:
    """Use TraceLab's canonical CSV token and summed tool-wall-time mapping."""
    wait_ms = 0.0
    for tool in row.get("tools", []):
        value = tool.get("tool_wall_latency_ms")
        if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
            wait_ms += value
    return {
        "source_round_index": int(row.get("round_index") or 0),
        "source_row": ordinal,
        "prefix_len": max(0, int(row.get("prefix_tokens") or 0)),
        "input_len": max(1, int(row.get("newly_append_tokens") or 0)),
        "output_len": max(1, int(row.get("output_tokens") or 0)),
        "tool_wait_after_ms": wait_ms,
        "input_ready_ms": input_ready_ms(row),
    }


def select_sessions(
    sessions: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    sessions_per_provider: int,
    rounds: int,
    max_model_len: int,
    max_wait_seconds: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Sample equal provider strata without clipping tokens or interior waits."""
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    for (provider, session_id), rows in sorted(sessions.items()):
        window = sorted(
            rows, key=lambda row: (row["source_round_index"], row["source_row"])
        )[:rounds]
        if len(window) != rounds:
            rejected["short_session"] += 1
            continue
        if any(
            row["prefix_len"] + row["input_len"] + row["output_len"] > max_model_len
            for row in window
        ):
            rejected["context_limit"] += 1
            continue
        wait_ms = sum(row["tool_wait_after_ms"] for row in window[:-1])
        if wait_ms > max_wait_seconds * 1000:
            rejected["wait_limit"] += 1
            continue
        window = [dict(row) for row in window]
        window[-1]["tool_wait_after_ms"] = 0.0
        candidates[provider].append(
            {"provider": provider, "session_id": session_id, "rounds": window}
        )

    rng = random.Random(seed)
    selected = []
    for provider in ("claude", "codex"):
        pool = candidates[provider]
        if len(pool) < sessions_per_provider:
            raise ValueError(f"Only {len(pool)} eligible {provider} sessions")
        selected.extend(rng.sample(pool, sessions_per_provider))
    rng.shuffle(selected)
    return selected, {
        "eligible_by_provider": {key: len(value) for key, value in candidates.items()},
        "rejected_sessions": dict(rejected),
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    with args.source.open("rb") as source:
        hasher = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
        digest = hasher.hexdigest()
    if digest != RELEASE_SHA256:
        raise ValueError(f"TraceLab release checksum mismatch: {digest}")
    sessions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(args.source, "rt", encoding="utf-8") as source:
        for ordinal, line in enumerate(source):
            row = json.loads(line)
            provider, session_id = row.get("provider"), row.get("session_id")
            if provider in {"claude", "codex"} and isinstance(session_id, str):
                sessions[provider, session_id].append(normalize_round(row, ordinal))
    if args.timing == "trace":
        selected, selection = select_timeline(
            sessions,
            window_seconds=args.window_seconds,
            copies=args.load_copies,
            max_model_len=args.max_model_len,
            window_start_ms=timestamp_ms(args.window_start),
        )
    else:
        selected, selection = select_sessions(
            sessions,
            sessions_per_provider=args.sessions_per_provider,
            rounds=args.rounds,
            max_model_len=args.max_model_len,
            max_wait_seconds=args.max_wait_seconds,
            seed=args.seed,
        )
        for index, session in enumerate(selected):
            session["arrival_time_ms"] = index * args.arrival_gap_ms
        selection.update(
            timing_mode="legacy_closed_loop",
            window_rounds=args.rounds,
            max_window_wait_seconds=args.max_wait_seconds,
            synthetic_arrival_gap_ms=args.arrival_gap_ms,
        )
    rows = [row for session in selected for row in session["rounds"]]
    metadata = {
        "source_url": RELEASE_URL,
        "source_sha256": digest,
        "license": "CC BY 4.0; attribution: uw-syfi/TraceLab",
        "source_sessions": len(sessions),
        "source_rounds": sum(map(len, sessions.values())),
        "seed": args.seed,
        "max_model_len": args.max_model_len,
        "selected_sessions": len(selected),
        "selected_rounds": len(rows),
        "prompt_tokens": sum(row["prefix_len"] + row["input_len"] for row in rows),
        "output_tokens": sum(row["output_len"] for row in rows),
        "tool_wait_ms": sum(row["tool_wait_after_ms"] for row in rows),
        "max_prompt_tokens": max(row["prefix_len"] + row["input_len"] for row in rows),
        **selection,
    }
    return {"schema_version": 2, "metadata": metadata, "sessions": selected}


class PromptBuilder:
    """Preserve previous context and real output IDs; synthesize only missing input."""

    def __init__(self, session_id: str, seed: int):
        digest = hashlib.sha256(f"{seed}:{session_id}".encode()).digest()
        self.rng = random.Random(int.from_bytes(digest, "big"))
        self.context: list[int] = []

    def build(self, row: dict[str, Any]) -> list[int]:
        prefix_len = row["prefix_len"]
        missing = max(0, prefix_len - len(self.context))
        self.context.extend(self.rng.randrange(1000, 30000) for _ in range(missing))
        return self.context[:prefix_len] + [
            self.rng.randrange(1000, 30000) for _ in range(row["input_len"])
        ]

    def commit(self, prompt: list[int], output: list[int]) -> None:
        self.context = prompt + output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing", choices=("trace", "legacy"), default="trace")
    parser.add_argument("--window-seconds", type=float, default=300)
    parser.add_argument("--window-start", help="ISO 8601 timestamp with timezone")
    parser.add_argument("--load-copies", type=int, default=1)
    parser.add_argument("--sessions-per-provider", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-wait-seconds", type=float, default=60)
    parser.add_argument("--arrival-gap-ms", type=float, default=100)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if min(args.sessions_per_provider, args.rounds, args.max_model_len) < 1:
        parser.error("session, round and context limits must be positive")
    if args.max_wait_seconds < 0 or args.arrival_gap_ms < 0:
        parser.error("wait and arrival limits must be nonnegative")
    if not math.isfinite(args.window_seconds) or args.window_seconds <= 0:
        parser.error("window duration must be finite and positive")
    if args.load_copies < 1:
        parser.error("load copies must be positive")
    if args.window_start is not None and timestamp_ms(args.window_start) is None:
        parser.error("window start requires a valid timestamp with timezone")
    workload = prepare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(workload, output, indent=2)
    print(json.dumps(workload["metadata"], indent=2))


if __name__ == "__main__":
    main()
