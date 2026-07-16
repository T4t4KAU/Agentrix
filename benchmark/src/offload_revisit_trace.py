from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def build_revisit_trace(source: dict[str, Any], pressure_requests: int) -> dict[str, Any]:
    if pressure_requests < 1 or pressure_requests > 25:
        raise ValueError("pressure_requests must be in [1, 25]")
    candidates = [
        event
        for event in source.get("events", [])
        if event.get("kind") == "llm" and not event.get("request", {}).get("tools")
    ]
    if not candidates:
        raise ValueError("source trace has no unconstrained LLM request")
    template = max(
        candidates,
        key=lambda event: int(event.get("usage", {}).get("prompt_tokens", 0)),
    )

    labels = ["A", *(chr(ord("B") + index) for index in range(pressure_requests)), "A"]
    events = []
    for index, label in enumerate(labels):
        request = copy.deepcopy(template["request"])
        messages = request.get("messages") or []
        if not messages:
            raise ValueError("selected request has no messages")
        messages[0]["content"] = (
            f"[OFFLOAD_PROBE_NAMESPACE:{label}]\n" + str(messages[0]["content"])
        )
        request["max_tokens"] = 1
        request.pop("tools", None)
        request.pop("tool_choice", None)
        events.append(
            {
                "kind": "llm",
                "case_id": "offload-revisit",
                "stage": "probe",
                "branch_id": None,
                "started_ms": index,
                "request": request,
            }
        )

    return {
        "schema_version": 1,
        "model": source.get("model"),
        "metadata": {
            "mode": "offload-revisit-source",
            "pressure_requests": pressure_requests,
            "request_order": labels,
            "template_prompt_tokens": int(
                template.get("usage", {}).get("prompt_tokens", 0)
            ),
        },
        "events": events,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pressure-requests", type=int, default=6)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    payload = build_revisit_trace(source, args.pressure_requests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["metadata"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
