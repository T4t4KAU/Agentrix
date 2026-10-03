import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import longbench_qa_runner as runner
import pytest
from agentrix_application.prefix_prefill_gate import PrefixPrefillGate


def test_document_identity_and_service_ttft_exclude_client_queue(monkeypatch):
    ticks = iter([0, 10, 11, 12])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(ticks))

    async def events():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Paris"), finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(
                model_dump=lambda: {"prompt_tokens": 100, "completion_tokens": 1}
            ),
        )

    create = AsyncMock(return_value=events())
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    case = {
        "case_id": "doc",
        "dataset": "qasper",
        "context": "A document",
        "context_sha256": "abc",
        "context_tokens": 10,
    }
    question = {"question": "Where?", "answers": ["Paris"], "source_id": "q1"}
    row = asyncio.run(
        runner.ask(client, "model", case, question, asyncio.Semaphore(1), 64, True)
    )
    assert row["ttft_seconds"] == 1
    assert row["client_queue_seconds"] == 10
    assert row["latency_seconds"] == 2
    assert row["finish_reason"] == "stop"
    assert row["f1"] == 1
    assert create.call_args.kwargs["extra_headers"] == {"X-Session-ID": "longbench-abc"}
    assert create.call_args.kwargs["messages"][0]["content"].endswith("A document")


def test_transport_failure_releases_prefill_gate():
    async def run():
        gate = PrefixPrefillGate()
        create = AsyncMock(side_effect=RuntimeError("transport failure"))
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with pytest.raises(RuntimeError, match="transport failure"):
            await runner.ask(
                client,
                "model",
                {"context": "document", "context_sha256": "abc"},
                {"question": "Where?"},
                asyncio.Semaphore(1),
                64,
                True,
                gate,
            )
        assert not gate.pending

    asyncio.run(run())
