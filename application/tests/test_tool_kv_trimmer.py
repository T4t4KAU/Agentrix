import asyncio

import pytest

from agentrix_application import (
    OnlineHorizonTTLPredictor,
    ToolKVTrimmer,
    ToolKVTrimmerConfig,
    ToolTTLContext,
    VLLMToolKVClient,
)


def test_disabled_switch_does_not_start_policy() -> None:
    async def run() -> None:
        calls = []

        async def pressure() -> float:
            calls.append("pressure")
            return 1.0

        async def trim(request_id: str) -> dict[str, object]:
            calls.append(request_id)
            return {"trimmed": True}

        trimmer = ToolKVTrimmer(
            pressure,
            trim,
            ToolKVTrimmerConfig(enabled=False, grace_ms=0),
        )
        assert not trimmer.tool_started("session", "request")
        await asyncio.sleep(0)
        assert calls == []

    asyncio.run(run())


def test_trims_after_grace_when_pressure_is_high() -> None:
    async def run() -> None:
        trimmed = []

        async def pressure() -> float:
            return 0.8

        async def trim(request_id: str) -> dict[str, object]:
            trimmed.append(request_id)
            return {"trimmed": True}

        trimmer = ToolKVTrimmer(
            pressure,
            trim,
            ToolKVTrimmerConfig(enabled=True, grace_ms=0, pressure_threshold=0.7),
        )
        assert trimmer.tool_started("session", "request")
        await trimmer.wait("session")
        assert trimmed == ["request"]
        assert trimmer.stats.trim_attempts == 1
        assert trimmer.stats.trimmed_sessions == 1

    asyncio.run(run())


def test_low_pressure_keeps_kv_resident() -> None:
    async def run() -> None:
        trimmed = []

        async def pressure() -> float:
            return 0.4

        async def trim(request_id: str) -> dict[str, object]:
            trimmed.append(request_id)
            return {"trimmed": True}

        trimmer = ToolKVTrimmer(
            pressure,
            trim,
            ToolKVTrimmerConfig(enabled=True, grace_ms=0, pressure_threshold=0.7),
        )
        trimmer.tool_started("session", "request")
        await trimmer.wait("session")
        assert trimmed == []
        assert trimmer.stats.pressure_skips == 1

    asyncio.run(run())


def test_fast_tool_result_cancels_pending_trim() -> None:
    async def run() -> None:
        trimmed = []

        async def pressure() -> float:
            return 1.0

        async def trim(request_id: str) -> dict[str, object]:
            trimmed.append(request_id)
            return {"trimmed": True}

        trimmer = ToolKVTrimmer(
            pressure,
            trim,
            ToolKVTrimmerConfig(enabled=True, grace_ms=60_000),
        )
        trimmer.tool_started("session", "request")
        await trimmer.tool_finished("session")
        assert trimmed == []
        assert trimmer.stats.cancelled_before_grace == 1

    asyncio.run(run())


def test_concurrent_sessions_recheck_pressure_after_each_trim() -> None:
    async def run() -> None:
        pressure_value = 0.9
        trimmed = []

        async def pressure() -> float:
            return pressure_value

        async def trim(request_id: str) -> dict[str, object]:
            nonlocal pressure_value
            await asyncio.sleep(0)
            trimmed.append(request_id)
            pressure_value = 0.5
            return {
                "trimmed": True,
                "released_block_references": 12,
                "kv_cache_usage_before": 0.9,
                "kv_cache_usage_after": 0.5,
            }

        trimmer = ToolKVTrimmer(
            pressure,
            trim,
            ToolKVTrimmerConfig(
                enabled=True,
                grace_ms=0,
                pressure_threshold=0.7,
                post_trim_recheck_ms=0,
            ),
        )
        trimmer.tool_started("first", "request-1")
        trimmer.tool_started("second", "request-2")
        await asyncio.gather(trimmer.wait("first"), trimmer.wait("second"))

        assert len(trimmed) == 1
        assert trimmer.stats.trimmed_sessions == 1
        assert trimmer.stats.pressure_skips == 1
        assert trimmer.stats.released_block_references == 12
        assert abs(trimmer.stats.observed_usage_reduction - 0.4) < 1e-9

    asyncio.run(run())


def test_stale_finish_does_not_cancel_replacement() -> None:
    async def run() -> None:
        async def pressure() -> float:
            return 1.0

        async def trim(request_id: str) -> dict[str, object]:
            return {"trimmed": True}

        trimmer = ToolKVTrimmer(
            pressure,
            trim,
            ToolKVTrimmerConfig(enabled=True, grace_ms=60_000),
        )
        assert trimmer.tool_started("session", "old-request")
        assert trimmer.tool_started("session", "new-request")
        assert not await trimmer.tool_finished("session", "old-request")
        assert trimmer.stats.stale_finishes == 1
        assert await trimmer.tool_finished("session", "new-request")
        assert trimmer.stats.cancelled_before_grace == 1
        assert trimmer.stats.superseded_calls == 1

    asyncio.run(run())


def test_finish_waits_for_inflight_trim_instead_of_orphaning_it() -> None:
    async def run() -> None:
        trim_started = asyncio.Event()
        allow_trim_to_finish = asyncio.Event()

        async def pressure() -> float:
            return 1.0

        async def trim(request_id: str) -> dict[str, object]:
            trim_started.set()
            await allow_trim_to_finish.wait()
            return {"trimmed": True}

        trimmer = ToolKVTrimmer(
            pressure,
            trim,
            ToolKVTrimmerConfig(enabled=True, grace_ms=0, post_trim_recheck_ms=0),
        )
        trimmer.tool_started("session", "request")
        await trim_started.wait()
        finished = asyncio.create_task(trimmer.tool_finished("session", "request"))
        await asyncio.sleep(0)
        assert not finished.done()
        allow_trim_to_finish.set()
        assert await finished
        assert trimmer.stats.finished_during_trim == 1
        assert trimmer.stats.trimmed_sessions == 1

    asyncio.run(run())


def test_invalid_pressure_is_rejected() -> None:
    async def run() -> None:
        async def pressure() -> float:
            return float("nan")

        async def trim(request_id: str) -> dict[str, object]:
            raise AssertionError("trim must not be called")

        trimmer = ToolKVTrimmer(
            pressure,
            trim,
            ToolKVTrimmerConfig(enabled=True, grace_ms=0),
        )
        trimmer.tool_started("session", "request")
        await trimmer.wait("session")
        assert trimmer.stats.errors == 1
        assert trimmer.stats.trim_attempts == 0

    asyncio.run(run())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("grace_ms", float("nan")),
        ("pressure_threshold", float("inf")),
        ("post_trim_recheck_ms", float("inf")),
    ],
)
def test_non_finite_config_is_rejected(field: str, value: float) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError):
        ToolKVTrimmerConfig(**kwargs)


def test_trim_rejection_reason_is_counted() -> None:
    async def run() -> None:
        async def pressure() -> float:
            return 1.0

        async def trim(request_id: str) -> dict[str, object]:
            return {"trimmed": False, "reason": "already_trimmed"}

        trimmer = ToolKVTrimmer(
            pressure,
            trim,
            ToolKVTrimmerConfig(enabled=True, grace_ms=0, post_trim_recheck_ms=0),
        )
        trimmer.tool_started("session", "request")
        await trimmer.wait("session")
        assert trimmer.stats.trim_rejections == 1
        assert trimmer.stats.trim_rejection_reasons == {"already_trimmed": 1}

    asyncio.run(run())


def test_close_prevents_new_policy_tasks() -> None:
    async def run() -> None:
        async def pressure() -> float:
            return 1.0

        async def trim(request_id: str) -> dict[str, object]:
            return {"trimmed": True}

        trimmer = ToolKVTrimmer(pressure, trim, ToolKVTrimmerConfig(enabled=True))
        await trimmer.close()
        assert not trimmer.tool_started("session", "request")

    asyncio.run(run())


def test_ttl_predictor_shadow_mode_learns_without_changing_grace() -> None:
    async def run() -> None:
        predictor = OnlineHorizonTTLPredictor(min_training_samples=0)

        async def pressure() -> float:
            return 0.0

        async def trim(request_id: str) -> dict[str, object]:
            raise AssertionError("low pressure must not trim")

        trimmer = ToolKVTrimmer(
            pressure,
            trim,
            ToolKVTrimmerConfig(
                enabled=True,
                grace_ms=0,
                use_predicted_ttl=False,
                post_trim_recheck_ms=0,
            ),
            ttl_predictor=predictor,
        )
        context = ToolTTLContext(tool_family="public_test")
        trimmer.tool_started("session", "request", context)
        await trimmer.wait("session")
        await trimmer.tool_finished("session", "request", duration_ms=3000)
        assert predictor.sample_count == 1
        assert trimmer.stats.ttl_predictions == 1
        assert trimmer.stats.ttl_observations == 1

    asyncio.run(run())


def test_active_ttl_predictor_replaces_fixed_grace() -> None:
    async def run() -> None:
        predictor = OnlineHorizonTTLPredictor(min_training_samples=0)
        context = ToolTTLContext(tool_family="public_test")
        for _ in range(100):
            predictor.observe(context, 4000)

        async def pressure() -> float:
            return 1.0

        trimmed = []

        async def trim(request_id: str) -> dict[str, object]:
            trimmed.append(request_id)
            return {"trimmed": True}

        trimmer = ToolKVTrimmer(
            pressure,
            trim,
            ToolKVTrimmerConfig(
                enabled=True,
                grace_ms=60_000,
                use_predicted_ttl=True,
                post_trim_recheck_ms=0,
            ),
            ttl_predictor=predictor,
        )
        trimmer.tool_started("session", "request", context)
        pending = trimmer._pending["session"]
        assert pending.ttl_ms == predictor.min_ttl_ms
        await trimmer.tool_finished("session", "request", duration_ms=20)
        assert trimmed == []

    asyncio.run(run())


def test_vllm_client_reads_max_rank_pressure() -> None:
    async def run() -> None:
        client = VLLMToolKVClient("http://vllm")
        client._request = lambda path, payload: (  # type: ignore[method-assign]
            b'vllm:kv_cache_usage_perc{rank="0"} 0.25\n'
            b'vllm:kv_cache_usage_perc{rank="1"} 0.80\n'
        )

        assert await client.kv_cache_usage() == 0.8

    asyncio.run(run())


def test_vllm_client_posts_external_request_id() -> None:
    async def run() -> None:
        calls = []
        client = VLLMToolKVClient("http://vllm/")

        def request(path: str, payload: dict[str, object] | None) -> bytes:
            calls.append((path, payload))
            return b'{"request_id":"session","trimmed":true}'

        client._request = request  # type: ignore[method-assign]
        result = await client.trim("session")

        assert result["trimmed"] is True
        assert calls == [("/v1/agentrix/tool-kv/trim", {"request_id": "session"})]

    asyncio.run(run())
