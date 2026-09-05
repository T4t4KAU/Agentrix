import asyncio
import io
import random
import time
from types import SimpleNamespace

import pytest
from aiohttp import web
from tracelab_replay import (
    generate,
    prepare_timeline,
    prompt_source_delta,
    replay_timeline,
    summarize,
)
from tracelab_timeline import input_ready_ms, select_timeline, timestamp_ms
from tracelab_workload import PromptBuilder, normalize_round, select_sessions


def test_normalization_uses_canonical_trace_shape():
    row = normalize_round(
        {
            "prefix_tokens": -1,
            "newly_append_tokens": 0,
            "output_tokens": None,
            "tools": [{"tool_wall_latency_ms": value} for value in (3, None, -4, 5)],
        },
        12,
    )
    assert (row["prefix_len"], row["input_len"], row["output_len"]) == (0, 1, 1)
    assert row["tool_wait_after_ms"] == 8
    assert row["source_row"] == 12


def test_prompt_reuses_real_output_and_handles_compaction():
    builder = PromptBuilder("session", 1)
    prompt = builder.build({"prefix_len": 4, "input_len": 3})
    builder.commit(prompt, [40000, 40001])
    assert builder.build({"prefix_len": 9, "input_len": 1})[:9] == prompt + [
        40000,
        40001,
    ]
    assert builder.build({"prefix_len": 3, "input_len": 1})[:3] == prompt[:3]
    extended = builder.build({"prefix_len": 12, "input_len": 2})
    assert len(extended) == 14
    assert extended[:9] == prompt + [40000, 40001]


def test_prompt_seed_is_stable_and_sessions_do_not_share_a_prefix():
    row = {"prefix_len": 32, "input_len": 8}
    first = PromptBuilder("a", 1).build(row)
    assert first == PromptBuilder("a", 1).build(row)
    assert first[:16] != PromptBuilder("b", 1).build(row)[:16]


def test_selection_preserves_lengths_and_does_not_mutate_source():
    source = {
        (provider, str(index)): [
            normalize_round(
                {
                    "round_index": i,
                    "prefix_tokens": i * 8,
                    "newly_append_tokens": 8,
                    "output_tokens": 4,
                    "tools": [{"tool_wall_latency_ms": 3}],
                },
                i,
            )
            for i in range(3)
        ]
        for provider in ("claude", "codex")
        for index in range(3)
    }
    kwargs = {
        "sessions_per_provider": 2,
        "rounds": 2,
        "max_model_len": 32,
        "max_wait_seconds": 1,
        "seed": 4,
    }
    selected, stats = select_sessions(source, **kwargs)
    assert (selected, stats) == select_sessions(source, **kwargs)
    assert len(selected) == 4
    assert stats["eligible_by_provider"] == {"claude": 3, "codex": 3}
    for session in selected:
        assert session["rounds"][0]["tool_wait_after_ms"] == 3
        assert session["rounds"][1]["tool_wait_after_ms"] == 0
        assert session["rounds"][1]["prefix_len"] == 8
    assert source["claude", "0"][1]["tool_wait_after_ms"] == 3
    with pytest.raises(ValueError):
        select_sessions(source, **(kwargs | {"max_model_len": 19}))


@pytest.mark.parametrize(
    "fault", [None, "missing_done", "missing_ids", "short_decode", "wrong_prompt"]
)
def test_stream_requires_complete_exact_token_accounting(fault):
    import json

    import aiohttp

    async def run():
        async def handler(request):
            payload = await request.json()
            assert payload["prompt"] == [123, 456]
            ids = [] if fault == "missing_ids" else [77, 88]
            if fault == "short_decode":
                ids = [77]
            chunks = [
                {"choices": [{"text": "", "token_ids": []}]},
                {"choices": [{"text": "", "token_ids": ids}]},
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 3 if fault == "wrong_prompt" else 2,
                        "completion_tokens": len(ids),
                        "prompt_tokens_details": {"cached_tokens": 1},
                    },
                },
            ]
            body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            if fault != "missing_done":
                body += "data: [DONE]\n\n"
            return web.Response(text=body, content_type="text/event-stream")

        app = web.Application()
        app.router.add_post("/v1/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            async with aiohttp.ClientSession() as client:
                call = generate(
                    client,
                    f"http://127.0.0.1:{port}",
                    {"prompt": [123, 456], "max_tokens": 2},
                )
                if fault:
                    with pytest.raises(RuntimeError):
                        await call
                else:
                    result, output = await call
                    assert output == [77, 88]
                    assert result["cached_tokens"] == 1
                    assert result["e2e_ms"] >= result["ttft_ms"] >= 0
        finally:
            await runner.cleanup()

    asyncio.run(run())


def test_summary_uses_token_weighted_cache_rate_and_excludes_initial_turns():
    rows = [
        {
            "round_index": 0,
            "prompt_tokens": 100,
            "cached_tokens": 0,
            "output_tokens": 2,
            "ttft_ms": 10,
            "e2e_ms": 20,
        },
        {
            "round_index": 1,
            "prompt_tokens": 200,
            "cached_tokens": 100,
            "output_tokens": 2,
            "ttft_ms": 4,
            "e2e_ms": 10,
        },
    ]
    result = summarize(rows, 2)
    assert result["all_cached_token_rate"] == 1 / 3
    assert result["followup_cached_token_rate"] == 0.5
    assert result["requests_per_second"] == 1
    assert result["followup_ttft_ms_p50"] == 4


def test_prompt_sources_are_deltas_summed_across_replicas():
    before = 'vllm:prompt_tokens_by_source_total{engine="0",source="local_compute"} 10'
    after = (
        'vllm:prompt_tokens_by_source_total{engine="0",source="local_compute"} 20\n'
        'vllm:prompt_tokens_by_source_total{engine="1",source="local_compute"} 30\n'
        'vllm:prompt_tokens_by_source_total{engine="1",source="external_kv_transfer"} 512'
    )
    assert prompt_source_delta(before, after) == {
        "local_compute": 40,
        "external_kv_transfer": 512,
    }


def timed_round(sequence, stamp, **overrides):
    return dict(
        normalize_round(
            {"round_index": sequence, "newly_append_tokens": 32, "output_tokens": 2},
            sequence,
        ),
        input_ready_ms=stamp,
        **overrides,
    )


def test_input_ready_uses_latest_input_before_first_model_output():
    epoch = "2026-05-29T03:00:"
    row = {
        "timing_events": [
            {"event_type": kind, "timestamp": epoch + suffix}
            for kind, suffix in (
                ("user_message", "00Z"),
                ("tool_result", "03Z"),
                ("text", "05Z"),
                ("tool_result", "07Z"),
                ("usage_report", "09Z"),
            )
        ]
    }
    assert input_ready_ms(row) == timestamp_ms(epoch + "03Z")
    assert timestamp_ms("2026-05-29T11:00:03+08:00") == input_ready_ms(row)
    assert timestamp_ms("2026-05-29T03:00:03") is None
    assert timestamp_ms("invalid") is None
    assert input_ready_ms({"timing_events": row["timing_events"][-1:]}) is None
    assert input_ready_ms({"timing_events": row["timing_events"][2:4]}) is None


def test_timeline_selects_dense_sessions_without_compressing_time():
    source = {
        ("claude", "a"): [timed_round(0, 1000), timed_round(1, 1400)],
        ("codex", "b"): [timed_round(0, 1200)],
        ("codex", "c"): [timed_round(0, 5000)],
    }
    selected, stats = select_timeline(
        source, window_seconds=1, copies=4, max_model_len=64
    )
    assert stats["window_source_sessions"] == 2
    assert stats["window_source_requests"] == 3
    assert len(selected) == 8
    assert stats["time_scale"] == 1
    assert selected[0]["arrival_time_ms"] == 0
    assert selected[0]["rounds"][1]["arrival_time_ms"] == 400
    assert selected[1]["arrival_time_ms"] == 200
    assert len({s["session_id"] for s in selected}) == 8
    assert "arrival_time_ms" not in source["claude", "a"][0]


def test_timeline_explicit_window_is_half_open():
    selected, _ = select_timeline(
        {
            ("codex", "a"): [
                timed_round(0, 999),
                timed_round(1, 1000),
                timed_round(2, 1999),
                timed_round(3, 2000),
            ]
        },
        window_seconds=1,
        copies=1,
        max_model_len=64,
        window_start_ms=1000,
    )
    assert [r["arrival_time_ms"] for r in selected[0]["rounds"]] == [0, 999]


def test_dense_window_matches_exhaustive_reference():
    rng = random.Random(17)
    source = {
        ("codex", str(session)): [
            timed_round(i, rng.randrange(20) * 100) for i in range(12)
        ]
        for session in range(8)
    }
    events = [(r["input_ready_ms"], key) for key, rows in source.items() for r in rows]

    def score(start):
        identities = [key for stamp, key in events if start <= stamp < start + 500]
        return len(set(identities)), len(identities), -start

    expected_start = max({stamp for stamp, _ in events}, key=score)
    _, stats = select_timeline(source, window_seconds=0.5, copies=1, max_model_len=64)
    assert timestamp_ms(stats["window_start_utc"]) == expected_start
    assert (stats["window_source_sessions"], stats["window_source_requests"]) == score(
        expected_start
    )[:2]


def test_timeline_rejects_missing_times_and_breaks_context_across_skipped_rows():
    source = {
        ("codex", "a"): [
            timed_round(0, 1000),
            timed_round(1, None),
            timed_round(2, 1100, prefix_len=100),
            timed_round(3, 1200),
        ]
    }
    selected, stats = select_timeline(
        source, window_seconds=1, copies=1, max_model_len=64, window_start_ms=1000
    )
    assert stats["rejected_requests"] == {
        "missing_input_ready_time": 1,
        "context_limit": 1,
    }
    assert [r["context_reset"] for r in selected[0]["rounds"]] == [True, True]
    with pytest.raises(ValueError, match="no eligible"):
        select_timeline(
            source, window_seconds=1, copies=1, max_model_len=64, window_start_ms=5000
        )


def small_timeline():
    selected, _ = select_timeline(
        {("codex", "a"): [timed_round(0, 1000), timed_round(1, 1020, prefix_len=32)]},
        window_seconds=1,
        copies=2,
        max_model_len=128,
    )
    return {"metadata": {"seed": 1}, "sessions": selected}


def test_timeline_prompts_are_fixed_reuse_history_and_isolate_load_copies():
    workload = small_timeline()
    prepared = prepare_timeline(workload)
    assert prepared == prepare_timeline(workload)
    assert prepared[0][4] != prepared[1][4]
    assert prepared[2][4][:32] == prepared[0][4]
    assert prepared[3][4][:32] == prepared[1][4]
    assert [item[0] for item in prepared] == [0, 0, 0.02, 0.02]


def test_timeline_context_reset_does_not_invent_reuse_across_a_gap():
    workload = small_timeline()
    workload["sessions"][0]["rounds"][1]["context_reset"] = True
    prepared = prepare_timeline(workload)
    assert prepared[2].prompt[:32] != prepared[0].prompt


def test_timeline_submits_next_round_without_waiting_for_previous(monkeypatch):
    all_started = asyncio.Event()
    started = []

    async def fake_generate(client, endpoint, payload):
        started.append(payload)
        if len(started) == 4:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=2)
        return {
            "ttft_ms": 40,
            "e2e_ms": 40,
            "prompt_tokens": len(payload["prompt"]),
            "output_tokens": 2,
            "cached_tokens": 0,
        }, [10, 11]

    monkeypatch.setattr("tracelab_replay.generate", fake_generate)
    args = SimpleNamespace(seed=1, model="test", base_url="unused", max_inflight=8)
    rows = asyncio.run(
        replay_timeline(
            None,
            args,
            prepare_timeline(small_timeline()),
            time.perf_counter(),
            io.StringIO(),
        )
    )
    assert max(r["submitted_s"] for r in rows) < min(r["completed_s"] for r in rows)
    assert all(r["tool_wait_ms"] == 0 for r in rows)
    summary = summarize(rows, 0.1)
    assert summary["peak_inflight_requests"] == 4
    assert summary["peak_inflight_sessions"] == 2
    assert summary["arrival_lag_ms_max"] >= 0


def test_timeline_client_limit_aborts_instead_of_queuing(monkeypatch):
    cancelled = []

    async def fake_generate(client, endpoint, payload):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    monkeypatch.setattr("tracelab_replay.generate", fake_generate)
    args = SimpleNamespace(seed=1, model="test", base_url="unused", max_inflight=1)
    with pytest.raises(RuntimeError, match="refusing to shift arrivals"):
        asyncio.run(
            replay_timeline(
                None,
                args,
                prepare_timeline(small_timeline()),
                time.perf_counter(),
                io.StringIO(),
            )
        )
    assert cancelled == [True]
