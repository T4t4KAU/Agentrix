import asyncio

from agentrix_application.prefix_prefill_gate import PrefixPrefillGate


def test_waiters_release_and_other_prefixes_progress():
    async def run():
        gate = PrefixPrefillGate()
        leader = await gate.enter("a")
        waiter = asyncio.create_task(gate.enter("a"))
        await asyncio.sleep(0)
        assert not waiter.done()
        other = await gate.enter("b")
        assert other is not None
        gate.release("a", leader)
        assert await waiter is None
        gate.release("b", other)
        assert not gate.pending
        replacement = await gate.enter("a")
        gate.release("a", leader)
        assert gate.pending["a"] is replacement
        gate.release("a", replacement)

    asyncio.run(run())


def test_cancelled_waiter_does_not_cancel_leader_and_capacity_bypasses():
    async def run():
        gate = PrefixPrefillGate(max_pending=1)
        leader = await gate.enter("a")
        assert await gate.enter("b") is None
        waiter = asyncio.create_task(gate.enter("a"))
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        assert gate.pending["a"] is leader
        gate.release("a", leader)
        assert not gate.pending

    asyncio.run(run())
