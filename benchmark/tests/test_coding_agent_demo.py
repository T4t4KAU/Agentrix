from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from textual.widgets import Static
from aiohttp.test_utils import TestClient, TestServer

from coding_agent_demo import (
    DemoEvent,
    SideConfig,
    SideState,
    branch_messages,
    parse_engine_metrics,
    parse_gpu_ids,
    side_state_payload,
)
from coding_agent_demo_tui import DemoApp, agent_text
from coding_agent_demo_web import make_app, normalize_route_request_id


def config() -> SideConfig:
    return SideConfig(
        side="left",
        title="Agentrix",
        base_url="http://localhost:9000/v1",
        model="model",
        gpu_ids=(0, 1, 2, 3),
        prompt_compaction=True,
        routing_label="PREFIX-AWARE DP",
        color="#2dd4bf",
    )


def test_side_state_tracks_live_and_exact_completed_tokens() -> None:
    state = SideState(config())
    common = {"side": "left", "request_id": "case:2:r1"}
    state.apply(DemoEvent(kind="queued", case_id="case", agent_id=2, **common))
    state.apply(DemoEvent(kind="started", **common))
    state.apply(DemoEvent(kind="token", text="Repository evidence", **common))

    assert state.active_requests == 1
    assert state.prefill_requests == 0
    assert state.generating_requests == 1
    assert state.live_estimated_tokens > 0

    state.apply(
        DemoEvent(
            kind="completed",
            values={"prompt_tokens": 30_000, "output_tokens": 64, "ttft_ms": 12},
            **common,
        )
    )

    assert state.displayed_tokens == 64
    assert state.completed_requests == 1
    assert state.prompt_tokens == 30_000
    assert state.ttft_p50_ms == 12


def test_completed_run_keeps_output_visible_and_freezes_throughput() -> None:
    state = SideState(config())
    common = {
        "side": "left",
        "request_id": "case:2:r1",
        "case_id": "case",
        "agent_id": 2,
    }
    state.apply(DemoEvent(kind="queued", **common))
    state.apply(DemoEvent(kind="started", **common))
    state.apply(DemoEvent(kind="token", text="Visible agent output", **common))
    state.apply(DemoEvent(kind="completed", values={"output_tokens": 10}, **common))
    state.apply(DemoEvent(side="left", kind="side_completed"))

    frozen_rate = state.output_tokens_per_s
    assert state.run_completed
    assert state.ended_at is not None
    assert state.output_tokens_per_s == frozen_rate
    assert "Visible agent output" in agent_text(state).plain
    assert side_state_payload(state)["streams"][0]["text"] == "Visible agent output"


def test_route_event_attaches_actual_dp_rank_to_request() -> None:
    state = SideState(config())
    common = {
        "side": "left",
        "request_id": "case:2:r1",
        "case_id": "case",
        "agent_id": 2,
    }
    state.apply(DemoEvent(kind="queued", **common))
    state.apply(
        DemoEvent(
            side="left",
            kind="route",
            request_id="case:2:r1",
            values={"dp_rank": 3},
        )
    )

    assert state.streams["case:2:r1"].dp_rank == 3
    assert side_state_payload(state)["routes"] == [
        {
            "request_id": "case:2:r1",
            "prefix_group": "case",
            "agent_id": 2,
            "dp_rank": 3,
            "status": "queued",
        }
    ]


def test_normalize_vllm_route_request_id_removes_subrequest_suffix() -> None:
    assert (
        normalize_route_request_id("chatcmpl-django_case:10:r1-a46fab7f")
        == "django_case:10:r1"
    )


def test_parse_engine_metrics_preserves_dp_rank() -> None:
    parsed = parse_engine_metrics(
        """
vllm:num_requests_running{engine="001",model_name="m"} 3
vllm:num_requests_running{engine="000",model_name="m"} 2
vllm:num_requests_waiting{engine="000",model_name="m"} 7
vllm:kv_cache_usage_perc{engine="001",model_name="m"} 0.25
vllm:request_queue_time_seconds_sum{engine="000",model_name="m"} 1.5
vllm:request_queue_time_seconds_count{engine="000",model_name="m"} 3
vllm:prompt_tokens_by_source_total{engine="000",source="local_compute"} 32000
vllm:prompt_tokens_by_source_total{engine="000",source="local_cache_hit"} 64000
"""
    )

    assert sorted(parsed["vllm:num_requests_running"]) == [(0, 2.0), (1, 3.0)]
    assert parsed["vllm:num_requests_waiting"] == [(0, 7.0)]
    assert parsed["vllm:kv_cache_usage_perc"] == [(1, 0.25)]
    assert parsed["vllm:request_queue_time_seconds_sum"] == [(0, 1.5)]
    assert parsed["vllm:request_queue_time_seconds_count"] == [(0, 3.0)]
    assert parsed["vllm:prompt_tokens_by_source_total"] == [(0, 32000.0)]


def test_side_state_reports_real_queue_depth_and_average() -> None:
    state = SideState(config())
    state.apply(
        DemoEvent(
            side="left",
            kind="metrics",
            values={
                "waiting_sequences": 19,
                "queue_time_sum_s": 2.4,
                "queue_time_count": 3,
                "computed_prefill_tokens": 32_000,
            },
        )
    )

    assert state.waiting_sequences == 19
    assert state.average_queue_time_ms == 800
    payload = side_state_payload(state)
    assert payload["waiting_sequences"] == 19
    assert payload["average_queue_time_ms"] == 800
    assert payload["computed_prefill_tokens"] == 32_000


def test_branch_messages_compacts_known_tool_section() -> None:
    case = {
        "shared_messages": [{"role": "system", "content": "parent"}],
        "known_prompt_sections": [
            {"segment_id": "source:a.py", "content": "same source"}
        ],
    }
    branch = {
        "private_instruction": "inspect",
        "trajectory": [
            {
                "stage": "tool",
                "tool": "repository_search",
                "tool_observation": "found",
                "tool_sections": [
                    {"segment_id": "source:a.py", "content": "same source"},
                    {"segment_id": "source:b.py", "content": "new source"},
                ],
                "instruction": "continue",
            }
        ],
    }

    compacted = branch_messages(case, branch, rounds=1, prompt_compaction=True)
    baseline = branch_messages(case, branch, rounds=1, prompt_compaction=False)

    assert "same source" not in compacted[0][-1]["content"]
    assert "new source" in compacted[0][-1]["content"]
    assert "same source" in baseline[0][-1]["content"]


def test_parse_gpu_ids() -> None:
    assert parse_gpu_ids("0, 2,3") == (0, 2, 3)


def test_demo_app_mounts_in_mock_mode(tmp_path: Path) -> None:
    case_path = tmp_path / "cases.jsonl"
    case_path.write_text(
        json.dumps(
            {
                "case_id": "demo-case",
                "shared_messages": [{"role": "system", "content": "parent"}],
                "branches": [
                    {"branch_id": index, "private_instruction": "inspect"}
                    for index in range(4)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        cases=case_path,
        case_offset=0,
        case_count=1,
        rounds=1,
        max_tokens=16,
        left_base_url="http://127.0.0.1:9000/v1",
        right_base_url="http://127.0.0.1:9001/v1",
        left_model="left",
        right_model="right",
        left_gpus="0,1,2,3",
        right_gpus="4,5,6,7",
        left_gpu_metrics_url=None,
        right_gpu_metrics_url=None,
        mock=True,
    )

    async def mount() -> None:
        app = DemoApp(args)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query_one("#left-agents", Static)
            assert app.query_one("#right-metrics", Static)
            app.exit()

    asyncio.run(mount())


def test_web_dashboard_serves_assets_and_initial_snapshot(tmp_path: Path) -> None:
    case_path = tmp_path / "cases.jsonl"
    case_path.write_text(
        json.dumps(
            {
                "case_id": "demo-case",
                "shared_messages": [{"role": "system", "content": "parent"}],
                "branches": [
                    {"branch_id": index, "private_instruction": "inspect"}
                    for index in range(4)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        cases=case_path,
        case_offset=0,
        case_count=1,
        rounds=1,
        max_tokens=16,
        left_base_url="http://127.0.0.1:9000/v1",
        right_base_url="http://127.0.0.1:9001/v1",
        left_model="left",
        right_model="right",
        left_gpus="0,1,2,3",
        right_gpus="4,5,6,7",
        left_gpu_metrics_url=None,
        right_gpu_metrics_url=None,
        mock=True,
        web_host="127.0.0.1",
        web_port=0,
        no_open_browser=True,
    )

    async def request() -> None:
        client = TestClient(TestServer(make_app(args)))
        await client.start_server()
        try:
            response = await client.get("/")
            assert response.status == 200
            assert "Live Coding Agents" in await response.text()
            socket = await client.ws_connect("/ws")
            payload = await socket.receive_json()
            assert payload["phase"] == "idle"
            assert payload["left"]["routing_label"] == "PREFIX-AWARE DP"
            await socket.close()
        finally:
            await client.close()

    asyncio.run(request())
