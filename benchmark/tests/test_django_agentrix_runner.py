import asyncio
import json
from pathlib import Path

import pytest

from django_agentrix_runner import (
    RequestMetric,
    WaveBarrier,
    load_cases,
    percentile,
    select_cases,
    summarize_repository_metrics,
)


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 3.0], 0.5) == 2.0
    assert percentile([], 0.5) is None


def test_wave_barrier_releases_all_parties() -> None:
    async def exercise() -> list[int]:
        barrier = WaveBarrier(3)

        async def arrive(index: int) -> int:
            await barrier.wait()
            return index

        return await asyncio.gather(*(arrive(index) for index in range(3)))

    assert asyncio.run(exercise()) == [0, 1, 2]


def test_load_cases_reads_one_repository_file(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"case_id": f"case-{index}", "repo": "django/django"})
            for index in range(2)
        ),
        encoding="utf-8",
    )

    assert [case["case_id"] for case in load_cases(path)] == ["case-0", "case-1"]


def test_load_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps({"case_id": "duplicate"}),
                json.dumps({"case_id": "duplicate"}),
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="case IDs must be unique"):
        load_cases(path)


def test_select_cases_partitions_24_cases_into_three_dp8_batches() -> None:
    cases = [{"case_id": f"case-{index}"} for index in range(24)]

    batches = [
        select_cases(cases, offset=offset, limit=8, required_count=8)
        for offset in (0, 8, 16)
    ]

    assert [len(batch) for batch in batches] == [8, 8, 8]
    assert {case["case_id"] for batch in batches for case in batch} == {
        f"case-{index}" for index in range(24)
    }


def test_summarize_repository_metrics_keeps_repository_breakdown() -> None:
    metrics = [
        RequestMetric(
            case_id="django-case",
            stage="branch_round_1",
            branch_id=0,
            round_index=1,
            started_ms=10.0,
            latency_ms=20.0,
            input_tokens=100,
            output_tokens=10,
            ttft_ms=5.0,
            tpot_ms=1.5,
        ),
        RequestMetric(
            case_id="sqlite-case",
            stage="branch_round_1",
            branch_id=0,
            round_index=1,
            started_ms=12.0,
            latency_ms=25.0,
            input_tokens=120,
            output_tokens=20,
            ttft_ms=7.0,
            tpot_ms=1.0,
        ),
    ]

    result = summarize_repository_metrics(
        metrics,
        {"django-case": "django/django", "sqlite-case": "sqlite/sqlite"},
    )

    assert result["django/django"]["input_tokens"] == 100
    assert result["sqlite/sqlite"]["output_tokens"] == 20
