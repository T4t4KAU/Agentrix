#!/usr/bin/env python3
"""Evaluate the application TTL predictor on held-out tool durations."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "application" / "src"))

from agentrix_application import (  # noqa: E402
    OnlineHorizonTTLPredictor,
    ToolTTLContext,
)


@dataclass(frozen=True)
class TimedToolEvent:
    context: ToolTTLContext
    duration_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("synthetic", "trace"), required=True)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def synthetic_events(seed: int, samples_per_family: int = 400) -> list[TimedToolEvent]:
    rng = random.Random(seed)
    # Median and log-space sigma approximate distinct coding-agent tool regimes.
    regimes = {
        "read": (45.0, 0.30),
        "search": (220.0, 0.35),
        "network": (900.0, 0.45),
        "public_test": (4000.0, 0.40),
        "build": (8000.0, 0.50),
    }
    events = []
    for family, (median_ms, sigma) in regimes.items():
        for _ in range(samples_per_family):
            duration_ms = median_ms * math.exp(rng.gauss(0, sigma))
            events.append(
                TimedToolEvent(
                    context=ToolTTLContext(
                        tool_family=family,
                        argument_bytes=rng.randint(20, 4000),
                        kv_tokens=rng.choice((4096, 8192, 16384, 32768)),
                        pressure=rng.uniform(0.55, 0.95),
                        active_tool_sessions=rng.choice((1, 2, 4, 8, 16)),
                        shared_prefix_ratio=rng.uniform(0.25, 0.95),
                        timeout_ms=30_000 if family != "build" else 120_000,
                    ),
                    duration_ms=duration_ms,
                )
            )
    rng.shuffle(events)
    return events


def walk_objects(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_objects(child)


def trace_events(root: Path) -> list[TimedToolEvent]:
    events = []
    for path in root.glob("**/run.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for event in walk_objects(payload):
            tool = event.get("tool")
            duration = (
                event.get("latency_ms")
                if event.get("kind") == "tool"
                else event.get("wall_time_ms")
            )
            if not isinstance(tool, str) or not isinstance(duration, (int, float)):
                continue
            if not math.isfinite(duration) or duration < 0:
                continue
            arguments = event.get("arguments", {})
            argument_bytes = len(json.dumps(arguments, default=str).encode())
            events.append(
                TimedToolEvent(
                    context=ToolTTLContext(
                        tool_family=tool,
                        argument_bytes=argument_bytes,
                    ),
                    duration_ms=float(duration),
                )
            )
    return events


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def oracle_ttl(predictor: OnlineHorizonTTLPredictor, duration_ms: float) -> float:
    survived = [horizon for horizon in predictor.horizons_ms if duration_ms > horizon]
    if not survived:
        return predictor.fallback_ttl_ms
    ttl_ms = predictor.fallback_ttl_ms * predictor.min_ttl_ms / max(survived)
    return max(predictor.min_ttl_ms, min(predictor.fallback_ttl_ms, ttl_ms))


def evaluate(events: list[TimedToolEvent], train_fraction: float) -> dict[str, Any]:
    if len(events) < 20:
        raise RuntimeError(f"at least 20 events are required, got {len(events)}")
    split = int(len(events) * train_fraction)
    if not 0 < split < len(events):
        raise ValueError("train_fraction must leave non-empty train and test sets")
    train = events[:split]
    test = events[split:]
    predictor = OnlineHorizonTTLPredictor(min_training_samples=50)
    for event in train:
        predictor.observe(event.context, event.duration_ms)

    horizon_rows = []
    for index, horizon in enumerate(predictor.horizons_ms):
        correct = true_positive = false_positive = false_negative = 0
        brier_sum = 0.0
        positives = 0
        for event in test:
            probability = predictor.predict(event.context).survival_probabilities[index]
            actual = event.duration_ms > horizon
            predicted = probability >= 0.5
            correct += predicted == actual
            positives += actual
            true_positive += predicted and actual
            false_positive += predicted and not actual
            false_negative += not predicted and actual
            brier_sum += (probability - float(actual)) ** 2
        horizon_rows.append(
            {
                "horizon_ms": horizon,
                "positive_rate": positives / len(test),
                "accuracy": correct / len(test),
                "precision": safe_ratio(true_positive, true_positive + false_positive),
                "recall": safe_ratio(true_positive, true_positive + false_negative),
                "brier_score": brier_sum / len(test),
            }
        )

    exact_ttl = within_one_bucket = shorten_correct = 0
    ttl_values = sorted(
        {oracle_ttl(predictor, event.duration_ms) for event in test}
        | {predictor.fallback_ttl_ms, predictor.min_ttl_ms}
    )
    for event in test:
        prediction = predictor.predict(event.context)
        expected_ttl = oracle_ttl(predictor, event.duration_ms)
        exact_ttl += math.isclose(prediction.ttl_ms, expected_ttl)
        predicted_index = min(
            range(len(ttl_values)),
            key=lambda index: abs(ttl_values[index] - prediction.ttl_ms),
        )
        expected_index = ttl_values.index(expected_ttl)
        within_one_bucket += abs(predicted_index - expected_index) <= 1
        shorten_correct += (prediction.ttl_ms < predictor.fallback_ttl_ms) == (
            expected_ttl < predictor.fallback_ttl_ms
        )

    families: dict[str, int] = {}
    for event in events:
        family = event.context.tool_family
        families[family] = families.get(family, 0) + 1
    return {
        "total_events": len(events),
        "train_events": len(train),
        "test_events": len(test),
        "tool_families": families,
        "horizons": horizon_rows,
        "macro_horizon_accuracy": sum(row["accuracy"] for row in horizon_rows)
        / len(horizon_rows),
        "macro_brier_score": sum(row["brier_score"] for row in horizon_rows)
        / len(horizon_rows),
        "ttl_exact_accuracy": exact_ttl / len(test),
        "ttl_within_one_bucket_accuracy": within_one_bucket / len(test),
        "shorten_decision_accuracy": shorten_correct / len(test),
    }


def main() -> None:
    args = parse_args()
    if args.dataset == "synthetic":
        events = synthetic_events(args.seed)
    else:
        if args.trace_root is None:
            raise ValueError("--trace-root is required for the trace dataset")
        events = trace_events(args.trace_root)
    result = {
        "dataset": args.dataset,
        "seed": args.seed,
        **evaluate(events, args.train_fraction),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
