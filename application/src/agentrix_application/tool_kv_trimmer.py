"""Application-owned policy for trimming idle tool-call KV cache."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import time
import urllib.request
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .tool_ttl_predictor import ToolTTLContext, ToolTTLPredictor

PressureReader = Callable[[], Awaitable[float]]
TrimRequest = Callable[[str], Awaitable[Mapping[str, Any]]]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


@dataclass(frozen=True)
class ToolKVTrimmerConfig:
    """Policy configuration for tool-call KV trimming."""

    enabled: bool = False
    grace_ms: float = 500.0
    pressure_threshold: float = 0.7
    post_trim_recheck_ms: float = 25.0
    use_predicted_ttl: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.grace_ms) or self.grace_ms < 0:
            raise ValueError("grace_ms must be non-negative")
        if (
            not math.isfinite(self.pressure_threshold)
            or not 0 <= self.pressure_threshold <= 1
        ):
            raise ValueError("pressure_threshold must be in [0, 1]")
        if (
            not math.isfinite(self.post_trim_recheck_ms)
            or self.post_trim_recheck_ms < 0
        ):
            raise ValueError("post_trim_recheck_ms must be non-negative")

    @classmethod
    def from_env(cls) -> "ToolKVTrimmerConfig":
        """Build configuration from the Agentrix environment variables."""

        return cls(
            enabled=_env_bool("AGENTRIX_TOOL_KV_TRIM_ENABLED", False),
            grace_ms=float(os.getenv("AGENTRIX_TOOL_KV_TRIM_GRACE_MS", "500")),
            pressure_threshold=float(
                os.getenv("AGENTRIX_TOOL_KV_TRIM_PRESSURE_THRESHOLD", "0.7")
            ),
            post_trim_recheck_ms=float(
                os.getenv("AGENTRIX_TOOL_KV_TRIM_POST_TRIM_RECHECK_MS", "25")
            ),
            use_predicted_ttl=_env_bool(
                "AGENTRIX_TOOL_KV_TRIM_USE_PREDICTED_TTL", False
            ),
        )


@dataclass
class ToolKVTrimmerStats:
    """Small in-process counters for observing policy decisions."""

    tool_calls_started: int = 0
    trim_attempts: int = 0
    trimmed_sessions: int = 0
    pressure_skips: int = 0
    cancelled_before_grace: int = 0
    cancelled_before_trim: int = 0
    finished_during_trim: int = 0
    superseded_calls: int = 0
    stale_finishes: int = 0
    trim_rejections: int = 0
    trim_rejection_reasons: dict[str, int] = field(default_factory=dict)
    ttl_predictions: int = 0
    ttl_fallbacks: int = 0
    ttl_observations: int = 0
    ttl_prediction_errors: int = 0
    predicted_ttl_ms_sum: float = 0.0
    released_block_references: int = 0
    observed_usage_reduction: float = 0.0
    last_pressure: float | None = None
    errors: int = 0


@dataclass
class _PendingTrim:
    request_id: str
    task: asyncio.Task[None] | None = None
    phase: str = "grace"
    started_at: float = field(default_factory=time.monotonic)
    ttl_ms: float = 0.0
    ttl_context: ToolTTLContext | None = None


class ToolKVTrimmer:
    """Trim a resumable vLLM session when a tool call waits under pressure.

    The class deliberately owns no vLLM types. The application supplies one
    async pressure reader and one async trim function, which keeps the policy
    independently testable and makes the vLLM integration replaceable.
    """

    def __init__(
        self,
        pressure_reader: PressureReader,
        trim_request: TrimRequest,
        config: ToolKVTrimmerConfig | None = None,
        ttl_predictor: ToolTTLPredictor | None = None,
    ) -> None:
        self.config = config or ToolKVTrimmerConfig.from_env()
        self.pressure_reader = pressure_reader
        self.trim_request = trim_request
        self.ttl_predictor = ttl_predictor
        self.stats = ToolKVTrimmerStats()
        self._pending: dict[str, _PendingTrim] = {}
        self._trim_lock = asyncio.Lock()
        self._closed = False

    def tool_started(
        self,
        session_id: str,
        request_id: str,
        ttl_context: ToolTTLContext | None = None,
    ) -> bool:
        """Schedule a trim decision for a tool call.

        Returns ``False`` when the feature switch is off. A second tool call
        for the same session replaces its earlier pending decision.
        """

        if not self.config.enabled or self._closed:
            return False
        previous = self._pending.get(session_id)
        if previous is not None and previous.task is not None:
            if not previous.task.done() and previous.phase == "trimming":
                # An HTTP trim cannot be cancelled once its worker thread has
                # started. Let the caller finish the old lifecycle first.
                return False
            if not previous.task.done():
                previous.task.cancel()
                self.stats.superseded_calls += 1
        self.stats.tool_calls_started += 1
        ttl_ms = self.config.grace_ms
        if self.ttl_predictor is not None and ttl_context is not None:
            try:
                prediction = self.ttl_predictor.predict(ttl_context)
                self.stats.ttl_predictions += 1
                self.stats.predicted_ttl_ms_sum += prediction.ttl_ms
                if prediction.used_fallback:
                    self.stats.ttl_fallbacks += 1
                if self.config.use_predicted_ttl:
                    ttl_ms = prediction.ttl_ms
            except Exception:
                self.stats.ttl_prediction_errors += 1
        pending = _PendingTrim(
            request_id=request_id,
            ttl_ms=ttl_ms,
            ttl_context=ttl_context,
        )
        pending.task = asyncio.create_task(self._run(session_id, pending))
        self._pending[session_id] = pending
        return True

    async def tool_finished(
        self,
        session_id: str,
        request_id: str | None = None,
        *,
        duration_ms: float | None = None,
    ) -> bool:
        """Finish one tool lifecycle and wait out any in-flight trim.

        Supplying ``request_id`` prevents a late completion from an older tool
        call from cancelling a newer call for the same session.
        """

        pending = self._pending.get(session_id)
        if pending is None or pending.task is None:
            return False
        if request_id is not None and request_id != pending.request_id:
            self.stats.stale_finishes += 1
            return False
        task = pending.task
        if not task.done() and pending.phase != "trimming":
            task.cancel()
            if pending.phase == "grace":
                self.stats.cancelled_before_grace += 1
            else:
                self.stats.cancelled_before_trim += 1
        elif not task.done():
            self.stats.finished_during_trim += 1
        with contextlib.suppress(asyncio.CancelledError):
            await task
        if duration_ms is None:
            duration_ms = (time.monotonic() - pending.started_at) * 1000
        self._observe_ttl(pending, duration_ms)
        if self._pending.get(session_id) is pending:
            self._pending.pop(session_id, None)
        return True

    async def close(self) -> None:
        """Cancel all outstanding timers."""

        self._closed = True
        pending_items = list(self._pending.values())
        for pending in pending_items:
            if (
                pending.task is not None
                and not pending.task.done()
                and pending.phase != "trimming"
            ):
                pending.task.cancel()
        for pending in pending_items:
            if pending.task is None:
                continue
            with contextlib.suppress(asyncio.CancelledError):
                await pending.task
        self._pending.clear()

    async def wait(self, session_id: str) -> None:
        """Wait for one session's pending policy decision, mainly for tests."""

        pending = self._pending.get(session_id)
        if pending is not None and pending.task is not None:
            await pending.task

    async def _run(self, session_id: str, pending: _PendingTrim) -> None:
        try:
            await asyncio.sleep(pending.ttl_ms / 1000)
            # Serializing the read/decision/trim transaction prevents all tool
            # calls whose grace timers expire together from acting on the same
            # stale high-pressure sample.
            async with self._trim_lock:
                pending.phase = "checking"
                pressure = float(await self.pressure_reader())
                if not math.isfinite(pressure) or not 0 <= pressure <= 1:
                    raise ValueError(f"invalid KV-cache pressure: {pressure!r}")
                self.stats.last_pressure = pressure
                if pressure < self.config.pressure_threshold:
                    self.stats.pressure_skips += 1
                    return
                self.stats.trim_attempts += 1
                pending.phase = "trimming"
                result = await self.trim_request(pending.request_id)
                if bool(result.get("trimmed")):
                    self.stats.trimmed_sessions += 1
                    self._record_trim_result(result)
                    if self.config.post_trim_recheck_ms:
                        await asyncio.sleep(self.config.post_trim_recheck_ms / 1000)
                else:
                    self.stats.trim_rejections += 1
                    reason = str(result.get("reason", "unknown"))
                    self.stats.trim_rejection_reasons[reason] = (
                        self.stats.trim_rejection_reasons.get(reason, 0) + 1
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            self.stats.errors += 1
        finally:
            pending.phase = "done"

    def _observe_ttl(self, pending: _PendingTrim, duration_ms: float) -> None:
        if self.ttl_predictor is None or pending.ttl_context is None:
            return
        try:
            self.ttl_predictor.observe(pending.ttl_context, duration_ms)
            self.stats.ttl_observations += 1
        except Exception:
            self.stats.ttl_prediction_errors += 1

    def _record_trim_result(self, result: Mapping[str, Any]) -> None:
        released = result.get("released_block_references")
        if isinstance(released, (int, float)) and released >= 0:
            self.stats.released_block_references += int(released)
        before = result.get("kv_cache_usage_before")
        after = result.get("kv_cache_usage_after")
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            reduction = float(before) - float(after)
            if math.isfinite(reduction) and reduction > 0:
                self.stats.observed_usage_reduction += reduction


class VLLMToolKVClient:
    """Minimal HTTP adapter for vLLM metrics and the Agentrix trim hook."""

    def __init__(self, base_url: str, timeout_s: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def kv_cache_usage(self) -> float:
        """Return the maximum live KV-cache usage reported by vLLM ranks."""

        body = await asyncio.to_thread(self._request, "/metrics", None)
        usages = []
        for line in body.decode().splitlines():
            if line.startswith("vllm:kv_cache_usage_perc"):
                fields = line.split()
                if len(fields) >= 2:
                    usages.append(float(fields[1]))
        if not usages:
            raise RuntimeError("vLLM metrics did not contain KV-cache usage")
        return max(usages)

    async def trim(self, request_id: str) -> Mapping[str, Any]:
        """Ask vLLM to release one idle resumable request's GPU blocks."""

        body = await asyncio.to_thread(
            self._request,
            "/v1/agentrix/tool-kv/trim",
            {"request_id": request_id},
        )
        result = json.loads(body)
        if not isinstance(result, dict):
            raise RuntimeError("vLLM trim response was not an object")
        return result

    def _request(self, path: str, payload: dict[str, Any] | None) -> bytes:
        data = None if payload is None else json.dumps(payload).encode()
        headers = {} if data is None else {"Content-Type": "application/json"}
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return response.read()
