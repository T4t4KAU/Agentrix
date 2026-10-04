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


def test_primed_fanout_scores_every_original_question_once(monkeypatch, tmp_path):
    cases = [
        {"case_id": str(i), "questions": [{"source_id": f"{i}-{j}"} for j in range(3)]}
        for i in range(2)
    ]
    monkeypatch.setattr(runner, "load_cases", lambda _: cases.copy())
    monkeypatch.setattr(
        runner, "AsyncOpenAI", lambda **_: SimpleNamespace(close=AsyncMock())
    )
    completed_first = set()
    seen = []

    async def ask(client, model, case, question, *args):
        identity = question["source_id"]
        if not identity.endswith("-0"):
            assert completed_first == {"0-0", "1-0"}
        await asyncio.sleep(0)
        if identity.endswith("-0"):
            completed_first.add(identity)
        seen.append(identity)
        return {
            "source_id": identity,
            "f1": 1,
            "exact_match": 1,
            "end_to_end_ttft_seconds": 1,
            "ttft_seconds": 1,
            "latency_seconds": 2,
            "usage": {"prompt_tokens": 100, "completion_tokens": 2},
        }

    monkeypatch.setattr(runner, "ask", ask)
    path = tmp_path / "cases.jsonl"
    path.write_text("fixture")
    args = SimpleNamespace(
        cases=path,
        base_url="http://test",
        model="model",
        timeout=10,
        coalesce_prefill=False,
        question_waves=False,
        prime_first_question=True,
        seed=42,
        concurrency=4,
        max_tokens=16,
        document_routing=True,
    )
    result = asyncio.run(runner.run(args))
    assert len(seen) == len(set(seen)) == 6
    assert result["question_count"] == 6
    assert result["prompt_tokens"] == 600
    assert [r["phase"] for r in result["results"]] == ["first"] * 2 + ["followup"] * 4
    assert set(result["phase_seconds"]) == {"first", "followup"}
