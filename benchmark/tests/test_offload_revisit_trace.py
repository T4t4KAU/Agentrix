from __future__ import annotations

from offload_revisit_trace import build_revisit_trace


def test_build_revisit_trace_selects_longest_unconstrained_request() -> None:
    source = {
        "model": "model",
        "events": [
            {
                "kind": "llm",
                "usage": {"prompt_tokens": 100},
                "request": {"messages": [{"content": "short"}]},
            },
            {
                "kind": "llm",
                "usage": {"prompt_tokens": 200},
                "request": {
                    "messages": [{"content": "forced"}],
                    "tools": [{"type": "function"}],
                },
            },
            {
                "kind": "llm",
                "usage": {"prompt_tokens": 150},
                "request": {
                    "messages": [{"content": "long"}],
                    "max_tokens": 99,
                },
            },
        ],
    }

    trace = build_revisit_trace(source, pressure_requests=2)

    assert trace["metadata"]["request_order"] == ["A", "B", "C", "A"]
    assert trace["metadata"]["template_prompt_tokens"] == 150
    assert [event["request"]["max_tokens"] for event in trace["events"]] == [
        1,
        1,
        1,
        1,
    ]
    first = trace["events"][0]["request"]["messages"][0]["content"]
    last = trace["events"][-1]["request"]["messages"][0]["content"]
    assert first == last
    assert first.startswith("[OFFLOAD_PROBE_NAMESPACE:A]")
