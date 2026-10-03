"""Optional in-process singleflight gate for simultaneous shared-prefix requests.

Call before acquiring a transport concurrency slot. Release at the first output
token or in a finally block on failure/cancellation. This is a dispatch hint,
not proof of KV residency: the serving engine remains responsible for reuse.
"""

import asyncio


class PrefixPrefillGate:
    """Coordinate one event loop; never retain completed prefixes or KV data.

    Keys must identify the same model, tenant and exact reusable prompt prefix.
    Multiple application processes are independent. Overflow bypasses the gate.
    """

    def __init__(self, max_pending: int = 1024):
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.max_pending = max_pending
        self.pending: dict[str, asyncio.Event] = {}

    async def enter(self, key: str) -> asyncio.Event | None:
        if not key:
            raise ValueError("prefix key must be nonempty")
        existing = self.pending.get(key)
        if existing is not None:
            await existing.wait()
            return None
        if len(self.pending) >= self.max_pending:
            return None
        event = asyncio.Event()
        self.pending[key] = event
        return event

    def release(self, key: str, event: asyncio.Event | None) -> None:
        if event is not None and self.pending.get(key) is event:
            del self.pending[key]
            event.set()
