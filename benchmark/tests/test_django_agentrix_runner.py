import asyncio

from django_agentrix_runner import WaveBarrier, percentile


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
