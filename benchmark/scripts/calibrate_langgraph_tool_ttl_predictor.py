#!/usr/bin/env python3
"""Calibrate the online TTL predictor from completed live-Agent traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentrix_application.tool_ttl_predictor import (
    OnlineHorizonTTLPredictor,
    ToolTTLContext,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-training-samples", type=int, default=6)
    args = parser.parse_args()

    predictor = OnlineHorizonTTLPredictor(
        min_training_samples=args.min_training_samples
    )
    observations = 0
    for path in args.input:
        payload = json.loads(path.read_text(encoding="utf-8"))
        starts: dict[int, dict[str, object]] = {}
        for item in payload["events"]:
            if item["kind"] == "tool_start":
                starts[int(item["round"])] = item
            elif item["kind"] == "tool_end":
                start = starts[int(item["round"])]
                duration_ms = float(item["time_ms"]) - float(start["time_ms"])
                argument = str(start["argument"])
                predictor.observe(
                    ToolTTLContext(
                        tool_family=str(start["tool"]),
                        argument_bytes=len(argument.encode()),
                        kv_tokens=int(payload.get("max_model_len", 8192)),
                        pressure=0.01,
                        active_tool_sessions=1,
                        timeout_ms=float(payload["tool_delay_ms"]) * 2,
                    ),
                    duration_ms,
                )
                observations += 1
    predictor.save(args.output)
    print(
        json.dumps(
            {
                "observations": observations,
                "sample_count": predictor.sample_count,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
