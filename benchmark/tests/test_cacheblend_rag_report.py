from __future__ import annotations

import json
from pathlib import Path

from cacheblend_rag_report import (
    collect,
    render_markdown,
    response_lexical_overlap,
    response_match_rate,
    summarize_runtime_logs,
    valid_rag_tool_call_rate,
)


def _payload(wall_ms: float, latency_ms: float, content: str = "answer") -> dict:
    return {
        "metadata": {"wall_ms": wall_ms},
        "events": [
            {
                "case_id": "case",
                "stage": "planner",
                "branch_id": None,
                "source_started_ms": 0,
                "latency_ms": latency_ms,
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                "response": {"content": content},
            }
        ],
    }


def test_response_match_ignores_tool_call_ids() -> None:
    left = _payload(100, 100, "")
    right = _payload(100, 100, "")
    for payload, call_id in ((left, "call-a"), (right, "call-b")):
        payload["events"][0]["response"]["tool_calls"] = [
            {
                "id": call_id,
                "function": {"name": "rag_search", "arguments": '{"query":"x"}'},
            }
        ]

    assert response_match_rate(left, right) == 1.0


def test_runtime_log_parser_accepts_vllm_adapter_fields(tmp_path: Path) -> None:
    log = tmp_path / "measured_server.log"
    log.write_text(
        "Reqid: request-a, Total tokens 4298, Inference Engine computed tokens: 0, "
        "LMCache hit tokens: 4123, need to load: 4123\n"
        "Reqid: request-a, Total tokens 4298, Inference Engine computed tokens: 0, "
        "LMCache hit tokens: 4123, need to load: 4123\n"
        "Retrieved 4058 out of 4123 out of total 4123 tokens\n",
        encoding="utf-8",
    )

    summary = summarize_runtime_logs([log])

    assert summary["lookup_total_tokens"] == 4298
    assert summary["lookup_hit_tokens"] == 4123
    assert summary["retrieved_tokens"] == 4058


def test_quality_diagnostics() -> None:
    left = _payload(100, 100, "alpha beta beta")
    right = _payload(100, 100, "alpha beta gamma")
    tool_event = {
        "stage": "tool_select",
        "response": {
            "tool_calls": [
                {
                    "function": {
                        "name": "rag_search",
                        "arguments": '{"query":"focused evidence","top_k":3}',
                    }
                }
            ]
        },
    }

    assert response_lexical_overlap(left, right) == 2 / 3
    assert valid_rag_tool_call_rate({"events": [tool_event]}) == 1


def test_collect_and_render_cacheblend_comparison(tmp_path: Path) -> None:
    scenario = tmp_path / "incident"
    (scenario / "baseline").mkdir(parents=True)
    (scenario / "cacheblend").mkdir()
    (scenario / "trace.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "rag_reuse": {"reuse_ratio": 0.5, "reordered_pairs": 2}
                }
            }
        ),
        encoding="utf-8",
    )
    for variant, wall in (("baseline", 200), ("cacheblend", 100)):
        for repeat in (1, 2, 3):
            (scenario / variant / f"run{repeat}.json").write_text(
                json.dumps(_payload(wall, wall)), encoding="utf-8"
            )
            repeat_dir = scenario / variant / f"repeat{repeat}"
            repeat_dir.mkdir()
            (repeat_dir / "measured_server.log").write_text("", encoding="utf-8")

    report = collect(tmp_path)
    markdown = render_markdown(report)

    assert report["scenarios"]["incident"]["comparison"]["wall_speedup"] == 2
    assert "50.0%" in markdown
    assert "2.00x" in markdown
    assert report["scenarios"]["incident"]["runtime_diagnostics"][
        "production_ready"
    ]
