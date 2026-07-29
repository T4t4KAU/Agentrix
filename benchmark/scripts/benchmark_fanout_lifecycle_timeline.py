#!/usr/bin/env python3
"""Capture branch-aware KV HOT/COOLING/COLD handling on one GPU."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import statistics
import subprocess
import time
from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import pynvml

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    _LifecycleMetricName,
    _TransferMetricName,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.kv_offload.base import (
    OffloadEvictionMetadata,
    OffloadKey,
    ReqContext,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.metrics.loggers import StatLoggerBase

from vllm import SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_LOGGER = "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler"
PROFILE_PREFIX = "Fanout offload profile:"
PROFILE_FIELD = re.compile(r"([a-z_]+)=([^\s]+)")
STARTED: float | None = None
WALL_ZERO: float | None = None
TRACE_PATH: Path | None = None
MEMORY_ROOT_BLOCKS = 0
MEMORY_KEY_INFO: dict[str, dict[str, Any]] = {}
MEMORY_EVICTION_METADATA: dict[str, OffloadEvictionMetadata] = {}


def elapsed_ms() -> float:
    return (time.perf_counter() - (STARTED or time.perf_counter())) * 1000


def trace_elapsed_ms() -> float:
    return (time.time() - (WALL_ZERO or time.time())) * 1000


def write_trace(kind: str, payload: dict[str, Any]) -> None:
    if TRACE_PATH is None:
        return
    row = {"kind": kind, **payload}
    with TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def key_id(key: OffloadKey) -> str:
    return hashlib.sha256(repr(key).encode()).hexdigest()[:16]


def lifecycle_name(value: int | None) -> str:
    return {1: "cold", 2: "cooling", 3: "hot"}.get(value, "unobserved")


def metadata_dict(
    metadata: OffloadEvictionMetadata | None,
) -> dict[str, Any]:
    if metadata is None:
        return {
            "lifecycle": "unobserved",
            "fanout": 1,
            "reuse_score": 0,
            "prefix_position": 1.0,
            "residency_value": 0,
        }
    return {
        "lifecycle": lifecycle_name(metadata.lifecycle_value),
        "fanout": metadata.fanout,
        "reuse_score": metadata.reuse_score,
        "prefix_position": metadata.prefix_position,
        "residency_value": metadata.residency_value,
    }


def classify_memory_key(
    request_id: str,
    logical_block_index: int | None = None,
) -> dict[str, Any]:
    branch_match = re.search(r"(?:wave1|wave2|revisit)-(\d+)", request_id)
    if branch_match is not None:
        branch = int(branch_match.group(1))
        if logical_block_index is not None and logical_block_index < MEMORY_ROOT_BLOCKS:
            return {
                "category": "shared_prefix",
                "branch": None,
                "source_request": request_id,
            }
        return {
            "category": "branch",
            "branch": branch,
            "source_request": request_id,
        }
    return {
        "category": "pressure",
        "branch": None,
        "source_request": request_id,
    }


def register_memory_key(
    raw_key: bytes,
    *,
    request_id: str,
    logical_block_index: int | None = None,
) -> dict[str, Any]:
    offload_key = OffloadKey(bytes(raw_key))
    identifier = key_id(offload_key)
    candidate = classify_memory_key(request_id, logical_block_index)
    previous = MEMORY_KEY_INFO.get(identifier)
    priority = {"pressure": 0, "branch": 1, "shared_prefix": 2}
    if (
        previous is None
        or priority[candidate["category"]] > priority[previous["category"]]
    ):
        MEMORY_KEY_INFO[identifier] = candidate
    return {"key": identifier, **MEMORY_KEY_INFO[identifier]}


def serialize_memory_key(
    raw_key: bytes,
    *,
    request_id: str,
) -> dict[str, Any]:
    offload_key = OffloadKey(bytes(raw_key))
    identifier = key_id(offload_key)
    info = MEMORY_KEY_INFO.get(identifier)
    metadata = MEMORY_EVICTION_METADATA.get(identifier)
    if info is None:
        info = classify_memory_key(request_id)
        if metadata is not None and (
            metadata.fanout >= 4 or metadata.reuse_score >= MEMORY_ROOT_BLOCKS // 2
        ):
            info = {
                "category": "shared_prefix",
                "branch": None,
                "source_request": request_id,
            }
        MEMORY_KEY_INFO[identifier] = info
    return {
        "key": identifier,
        **info,
        **metadata_dict(metadata),
    }


def install_cpu_eviction_observer() -> None:
    original_update = CPUOffloadingManager.update_eviction_metadata
    original_prepare = CPUOffloadingManager.prepare_store
    original_complete_store = CPUOffloadingManager.complete_store
    original_prepare_load = CPUOffloadingManager.prepare_load
    original_complete_load = CPUOffloadingManager.complete_load
    original_cache_full_blocks = BlockPool.cache_full_blocks
    original_evict_gpu_block = BlockPool._maybe_evict_cached_block

    def update_eviction_metadata(
        self: CPUOffloadingManager,
        metadata: Mapping[OffloadKey, OffloadEvictionMetadata],
        *,
        replace: bool = False,
    ) -> None:
        shadow = getattr(self, "_agentrix_lifecycle_metadata", {})
        if replace:
            shadow = dict(metadata)
        else:
            shadow.update(metadata)
        self._agentrix_lifecycle_metadata = shadow
        if replace:
            MEMORY_EVICTION_METADATA.clear()
        MEMORY_EVICTION_METADATA.update(
            {key_id(key): value for key, value in metadata.items()}
        )
        if WALL_ZERO is not None:
            counts = Counter(
                lifecycle_name(item.lifecycle_value) for item in metadata.values()
            )
            write_trace(
                "lifecycle",
                {
                    "elapsed_ms": trace_elapsed_ms(),
                    "hot_blocks": counts["hot"],
                    "cooling_blocks": counts["cooling"],
                    "cold_blocks": counts["cold"],
                    "observed_blocks": len(metadata),
                },
            )
        original_update(self, metadata, replace=replace)

    def prepare_store(
        self: CPUOffloadingManager,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ):
        output = original_prepare(self, keys, req_context)
        if output is None or WALL_ZERO is None:
            return output
        shadow = getattr(self, "_agentrix_lifecycle_metadata", {})
        evicted = []
        for key in output.evicted_keys:
            evicted.append(
                {
                    "key": key_id(key),
                    **metadata_dict(shadow.get(key)),
                }
            )
        if output.keys_to_store or evicted:
            counts = Counter(item["lifecycle"] for item in evicted)
            keys_by_lifecycle = {
                lifecycle: [
                    item["key"] for item in evicted if item["lifecycle"] == lifecycle
                ]
                for lifecycle in (
                    "hot",
                    "cooling",
                    "cold",
                    "unobserved",
                )
            }
            row = {
                "elapsed_ms": trace_elapsed_ms(),
                "event": "cpu_store",
                "policy": type(self._policy).__name__,
                "stored_blocks": len(output.keys_to_store),
                "evicted_blocks": len(evicted),
                "evicted_hot": counts["hot"],
                "evicted_cooling": counts["cooling"],
                "evicted_cold": counts["cold"],
                "evicted_unobserved": counts["unobserved"],
                "evicted_keys_by_lifecycle": keys_by_lifecycle,
            }
            write_trace("cpu", row)
        if output.evicted_keys:
            write_trace(
                "memory",
                {
                    "elapsed_ms": trace_elapsed_ms(),
                    "event": "cpu_evict",
                    "request_id": req_context.req_id,
                    "keys": [
                        serialize_memory_key(
                            key,
                            request_id=req_context.req_id,
                        )
                        for key in output.evicted_keys
                    ],
                },
            )
        return output

    def complete_store(
        self: CPUOffloadingManager,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        original_complete_store(self, keys, req_context, success)
        if not success or WALL_ZERO is None:
            return
        stored_keys = [
            key
            for key in keys
            if (block := self._policy.get(key)) is not None and block.is_ready
        ]
        if stored_keys:
            write_trace(
                "memory",
                {
                    "elapsed_ms": trace_elapsed_ms(),
                    "event": "cpu_store_complete",
                    "request_id": req_context.req_id,
                    "keys": [
                        serialize_memory_key(
                            key,
                            request_id=req_context.req_id,
                        )
                        for key in stored_keys
                    ],
                },
            )

    def prepare_load(
        self: CPUOffloadingManager,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ):
        output = original_prepare_load(self, keys, req_context)
        if WALL_ZERO is not None and keys:
            write_trace(
                "memory",
                {
                    "elapsed_ms": trace_elapsed_ms(),
                    "event": "cpu_load_start",
                    "request_id": req_context.req_id,
                    "keys": [
                        serialize_memory_key(
                            key,
                            request_id=req_context.req_id,
                        )
                        for key in keys
                    ],
                },
            )
        return output

    def complete_load(
        self: CPUOffloadingManager,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        original_complete_load(self, keys, req_context)
        if WALL_ZERO is not None and keys:
            write_trace(
                "memory",
                {
                    "elapsed_ms": trace_elapsed_ms(),
                    "event": "cpu_load_complete",
                    "request_id": req_context.req_id,
                    "keys": [
                        serialize_memory_key(
                            key,
                            request_id=req_context.req_id,
                        )
                        for key in keys
                    ],
                },
            )

    def cache_full_blocks(
        self: BlockPool,
        request: Any,
        blocks: list[Any],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
        block_mask: list[bool] | None = None,
    ) -> None:
        original_cache_full_blocks(
            self,
            request,
            blocks,
            num_cached_blocks,
            num_full_blocks,
            block_size,
            kv_cache_group_id,
            block_mask,
        )
        if WALL_ZERO is None or num_cached_blocks >= num_full_blocks:
            return
        cached = []
        for logical_index in range(num_cached_blocks, num_full_blocks):
            block = blocks[logical_index]
            if block.is_null or block.block_hash is None:
                continue
            info = register_memory_key(
                block.block_hash,
                request_id=request.request_id,
                logical_block_index=logical_index,
            )
            cached.append(
                {
                    **info,
                    "block_id": block.block_id,
                    "logical_block_index": logical_index,
                }
            )
        if cached:
            write_trace(
                "memory",
                {
                    "elapsed_ms": trace_elapsed_ms(),
                    "event": "gpu_cache",
                    "request_id": request.request_id,
                    "keys": cached,
                },
            )

    def evict_gpu_block(self: BlockPool, block: Any) -> bool:
        block_hash = block.block_hash
        block_id = block.block_id
        request_id = "gpu_allocator"
        key_info = (
            serialize_memory_key(block_hash, request_id=request_id)
            if block_hash is not None
            else None
        )
        evicted = original_evict_gpu_block(self, block)
        if evicted and WALL_ZERO is not None and key_info is not None:
            write_trace(
                "memory",
                {
                    "elapsed_ms": trace_elapsed_ms(),
                    "event": "gpu_evict",
                    "request_id": request_id,
                    "keys": [{**key_info, "block_id": block_id}],
                },
            )
        return evicted

    CPUOffloadingManager.update_eviction_metadata = update_eviction_metadata
    CPUOffloadingManager.prepare_store = prepare_store
    CPUOffloadingManager.complete_store = complete_store
    CPUOffloadingManager.prepare_load = prepare_load
    CPUOffloadingManager.complete_load = complete_load
    BlockPool.cache_full_blocks = cache_full_blocks
    BlockPool._maybe_evict_cached_block = evict_gpu_block


@dataclass(frozen=True)
class KVSample:
    timestamp: float
    usage: float
    running: int
    waiting: int
    tool_waiting: int
    connector: dict[str, float]
    fork_kind: str
    fork_active_ctas: int
    fork_shared_ctas: int
    fork_singleton_ctas: int
    fork_shared_queries: int
    fork_max_aggregated_queries: int


class TimelineLogger(StatLoggerBase):
    instances: ClassVar[list[TimelineLogger]] = []

    def __init__(self, vllm_config: Any, engine_index: int = 0) -> None:
        self.vllm_config = vllm_config
        self.engine_index = engine_index
        self.samples: list[KVSample] = []
        self.instances.append(self)

    def record(
        self,
        scheduler_stats: Any | None,
        iteration_stats: Any | None,
        mm_cache_stats: Any | None = None,
        engine_idx: int = 0,
    ) -> None:
        del iteration_stats, mm_cache_stats, engine_idx
        if scheduler_stats is None:
            return
        connector: dict[str, float] = {}
        raw = getattr(scheduler_stats, "kv_connector_stats", None)
        if raw is not None and hasattr(raw, "reduce"):
            raw = raw.reduce()
        elif isinstance(raw, Mapping) and {"types", "data"} <= raw.keys():
            raw = OffloadingConnectorStats(data=dict(raw)).reduce()
        if isinstance(raw, Mapping):
            connector = {
                str(key): float(value)
                for key, value in raw.items()
                if isinstance(value, (int, float))
            }
        fork_stats = getattr(scheduler_stats, "fork_execution_stats", None)
        if fork_stats is None:
            fork_stats = ("", 0, 0, 0, 0, 0, 0)
        self.samples.append(
            KVSample(
                timestamp=time.perf_counter(),
                usage=float(scheduler_stats.kv_cache_usage),
                running=int(scheduler_stats.num_running_reqs),
                waiting=int(scheduler_stats.num_waiting_reqs),
                tool_waiting=int(scheduler_stats.num_skipped_waiting_reqs),
                connector=connector,
                fork_kind=str(fork_stats[0]),
                fork_active_ctas=int(fork_stats[2]),
                fork_shared_ctas=int(fork_stats[3]),
                fork_singleton_ctas=int(fork_stats[4]),
                fork_shared_queries=int(fork_stats[5]),
                fork_max_aggregated_queries=int(fork_stats[6]),
            )
        )

    def log_engine_initialized(self) -> None:
        return


class ProfileHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.samples: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if PROFILE_PREFIX not in message or WALL_ZERO is None:
            return
        fields: dict[str, Any] = {}
        for key, value in PROFILE_FIELD.findall(message):
            if key in {"pressure", "layerwise_load"}:
                fields[key] = value
            elif "." in value:
                fields[key] = float(value)
            else:
                fields[key] = int(value)
        fields["elapsed_ms"] = trace_elapsed_ms()
        self.samples.append(fields)
        write_trace("profile", fields)


@dataclass
class LiveBranch:
    request_id: str
    ready: asyncio.Event
    generated_tokens: int = 0
    task: asyncio.Task[BranchResult] | None = None


@dataclass(frozen=True)
class BranchResult:
    request_id: str
    submitted_ms: float
    first_token_ms: float
    completed_ms: float
    output_tokens: int


def token_ids(seed: int, length: int, vocab_size: int) -> list[int]:
    usable = max(1, vocab_size - 1024)
    return [
        1024 + ((seed * 104729 + position * 8191) % usable)
        for position in range(length)
    ]


async def collect_live_branch(
    engine: AsyncLLM,
    *,
    branch: LiveBranch,
    prompt: list[int],
    warmup_tokens: int,
    sampling_params: SamplingParams,
    token_samples: list[dict[str, Any]],
) -> BranchResult:
    submitted_ms = elapsed_ms()
    first_token_ms: float | None = None
    async for output in engine.generate(
        {"prompt_token_ids": prompt},
        sampling_params,
        branch.request_id,
    ):
        now_ms = elapsed_ms()
        delta = sum(len(candidate.token_ids) for candidate in output.outputs)
        if not delta:
            continue
        if first_token_ms is None:
            first_token_ms = now_ms
        branch.generated_tokens += delta
        token_samples.append(
            {
                "elapsed_ms": now_ms,
                "phase": "live_branches",
                "request_id": branch.request_id,
                "tokens": delta,
            }
        )
        if branch.generated_tokens >= warmup_tokens:
            branch.ready.set()
    completed_ms = elapsed_ms()
    branch.ready.set()
    return BranchResult(
        request_id=branch.request_id,
        submitted_ms=submitted_ms,
        first_token_ms=first_token_ms or completed_ms,
        completed_ms=completed_ms,
        output_tokens=branch.generated_tokens,
    )


def launch_live_branches(
    engine: AsyncLLM,
    *,
    count: int,
    root: list[int],
    suffix_tokens: int,
    vocab_size: int,
    warmup_tokens: int,
    output_tokens: int,
    token_samples: list[dict[str, Any]],
) -> list[LiveBranch]:
    sampling_params = SamplingParams(
        max_tokens=output_tokens,
        temperature=0,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
    )
    branches = []
    for index in range(count):
        branch = LiveBranch(
            request_id=f"wave1-{index}",
            ready=asyncio.Event(),
        )
        prompt = root + token_ids(10_000 + index, suffix_tokens, vocab_size)
        branch.task = asyncio.create_task(
            collect_live_branch(
                engine,
                branch=branch,
                prompt=prompt,
                warmup_tokens=warmup_tokens,
                sampling_params=sampling_params,
                token_samples=token_samples,
            )
        )
        branches.append(branch)
    return branches


async def run_cold_pressure(
    engine: AsyncLLM,
    *,
    count: int,
    prompt_tokens: int,
    vocab_size: int,
    sampling_params: SamplingParams,
    events: list[dict[str, Any]],
    label: str = "pressure",
    seed_base: int = 100_000,
) -> None:
    event(events, f"{label}_started", pressure_requests=count)
    for index in range(count):
        async for _ in engine.generate(
            {
                "prompt_token_ids": token_ids(
                    seed_base + index, prompt_tokens, vocab_size
                )
            },
            sampling_params,
            f"{label}-{index}",
        ):
            pass
        event(
            events,
            f"{label}_request_completed",
            pressure_completed=index + 1,
        )
    event(events, f"{label}_completed", pressure_requests=count)


async def wait_live_ready(branches: list[LiveBranch]) -> None:
    await asyncio.wait_for(
        asyncio.gather(*(branch.ready.wait() for branch in branches)),
        timeout=240,
    )


async def wait_live_tokens(branches: list[LiveBranch], target: int) -> None:
    async def wait_one(branch: LiveBranch) -> None:
        assert branch.task is not None
        while branch.generated_tokens < target and not branch.task.done():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(
        asyncio.gather(*(wait_one(branch) for branch in branches)),
        timeout=240,
    )


async def stop_live_branches(engine: AsyncLLM, branches: list[LiveBranch]) -> None:
    tasks = []
    for branch in branches:
        assert branch.task is not None
        tasks.append(branch.task)
    active = [
        branch
        for branch in branches
        if branch.task is not None and not branch.task.done()
    ]
    if active:
        await engine.abort([branch.request_id for branch in active], internal=False)
    for branch in active:
        assert branch.task is not None
        branch.task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def pulse(
    engine: AsyncLLM,
    *,
    count: int,
    seed: int,
    vocab_size: int,
    sampling_params: SamplingParams,
) -> None:
    for index in range(count):
        async for _ in engine.generate(
            {"prompt_token_ids": token_ids(seed + index, 32, vocab_size)},
            sampling_params,
            f"pulse-{seed}-{index}",
        ):
            pass


async def sample_gpu(
    samples: list[dict[str, Any]],
    stop: asyncio.Event,
    *,
    interval_seconds: float,
) -> None:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    try:
        while not stop.is_set():
            timestamp = elapsed_ms()
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                pcie_rx_kib_s = pynvml.nvmlDeviceGetPcieThroughput(
                    handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                )
                pcie_tx_kib_s = pynvml.nvmlDeviceGetPcieThroughput(
                    handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                )
            except pynvml.NVMLError:
                pcie_rx_kib_s = 0
                pcie_tx_kib_s = 0
            try:
                power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
            except pynvml.NVMLError:
                power_watts = 0
            samples.append(
                {
                    "elapsed_ms": timestamp,
                    "gpu_utilization": int(utilization.gpu),
                    "memory_utilization": int(utilization.memory),
                    "memory_used_mib": memory.used / 1024**2,
                    "pcie_rx_mib_s": pcie_rx_kib_s / 1024,
                    "pcie_tx_mib_s": pcie_tx_kib_s / 1024,
                    "power_watts": power_watts,
                }
            )
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
            except TimeoutError:
                pass
    finally:
        pynvml.nvmlShutdown()


async def run_generation_wave(
    engine: AsyncLLM,
    *,
    label: str,
    branches: int,
    root: list[int],
    suffix_tokens: int,
    vocab_size: int,
    output_tokens: int,
    events: list[dict[str, Any]],
    token_samples: list[dict[str, Any]],
    active_base: int = 0,
    index_start: int = 0,
    distractors: int = 0,
) -> list[BranchResult]:
    sampling_params = SamplingParams(
        max_tokens=output_tokens,
        temperature=0,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
    )
    indices = range(index_start, index_start + branches)
    active = {f"{label}-{index}" for index in indices}
    event(
        events,
        f"{label}_started",
        active_branches=active_base + branches,
    )

    async def generate_branch(index: int) -> BranchResult:
        request_id = f"{label}-{index}"
        submitted_ms = elapsed_ms()
        first_token_ms: float | None = None
        generated = 0
        prompt = root + token_ids(400_000 + index, suffix_tokens, vocab_size)
        async for output in engine.generate(
            {"prompt_token_ids": prompt},
            sampling_params,
            request_id,
        ):
            now_ms = elapsed_ms()
            delta = sum(len(candidate.token_ids) for candidate in output.outputs)
            if delta:
                if first_token_ms is None:
                    first_token_ms = now_ms
                generated += delta
                token_samples.append(
                    {
                        "elapsed_ms": now_ms,
                        "phase": label,
                        "request_id": request_id,
                        "tokens": delta,
                    }
                )
        completed_ms = elapsed_ms()
        active.discard(request_id)
        event(
            events,
            f"{label}_branch_completed",
            active_branches=active_base + len(active),
            request_id=request_id,
        )
        return BranchResult(
            request_id=request_id,
            submitted_ms=submitted_ms,
            first_token_ms=first_token_ms or completed_ms,
            completed_ms=completed_ms,
            output_tokens=generated,
        )

    async def generate_distractor(index: int) -> None:
        prompt = token_ids(800_000 + index, suffix_tokens * 2, vocab_size)
        async for _ in engine.generate(
            {"prompt_token_ids": prompt},
            sampling_params,
            f"{label}-private-{index}",
        ):
            pass

    tasks: list[asyncio.Task[BranchResult | None]] = []
    for offset, index in enumerate(indices):
        tasks.append(asyncio.create_task(generate_branch(index)))
        if offset < distractors:
            tasks.append(asyncio.create_task(generate_distractor(index)))
    gathered = await asyncio.gather(*tasks)
    results = [result for result in gathered if result is not None]
    event(
        events,
        f"{label}_completed",
        active_branches=active_base,
        output_tokens=sum(result.output_tokens for result in results),
    )
    return results


def latest_usage() -> float:
    samples = [
        logger.samples[-1].usage
        for logger in TimelineLogger.instances
        if logger.samples
    ]
    return max(samples, default=0.0)


def event(events: list[dict[str, Any]], name: str, **details: Any) -> None:
    events.append(
        {
            "elapsed_ms": elapsed_ms(),
            "event": name,
            "kv_cache_usage": latest_usage(),
            **details,
        }
    )


def gpu_description() -> str:
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()


def summarize_wave(results: list[BranchResult]) -> dict[str, float | int]:
    started_ms = min(result.submitted_ms for result in results)
    completed_ms = max(result.completed_ms for result in results)
    elapsed_seconds = max((completed_ms - started_ms) / 1000, 1e-9)
    ttft = [result.first_token_ms - result.submitted_ms for result in results]
    return {
        "output_tokens": sum(result.output_tokens for result in results),
        "completed_branches": len(results),
        "elapsed_ms": completed_ms - started_ms,
        "goodput_tokens_per_second": (
            sum(result.output_tokens for result in results) / elapsed_seconds
        ),
        "branch_turns_per_second": len(results) / elapsed_seconds,
        "ttft_p50_ms": statistics.median(ttft),
        "ttft_p95_ms": statistics.quantiles(ttft, n=20, method="inclusive")[18],
    }


def summarize_interval(
    events: list[dict[str, Any]],
    token_samples: list[dict[str, Any]],
    *,
    started_event: str,
    completed_event: str,
) -> dict[str, float | int]:
    started_ms = next(
        float(row["elapsed_ms"]) for row in events if row["event"] == started_event
    )
    completed_ms = next(
        float(row["elapsed_ms"]) for row in events if row["event"] == completed_event
    )
    output_tokens = sum(
        int(sample["tokens"])
        for sample in token_samples
        if started_ms <= float(sample["elapsed_ms"]) <= completed_ms
    )
    elapsed_ms = max(completed_ms - started_ms, 1e-6)
    return {
        "output_tokens": output_tokens,
        "elapsed_ms": elapsed_ms,
        "goodput_tokens_per_second": output_tokens / elapsed_ms * 1000,
    }


def summarize_operator_interval(
    events: list[dict[str, Any]],
    operator_events: list[dict[str, Any]],
    *,
    started_event: str,
    completed_event: str,
) -> dict[str, float | int]:
    started_ms = next(
        float(row["elapsed_ms"]) for row in events if row["event"] == started_event
    )
    completed_ms = next(
        float(row["elapsed_ms"]) for row in events if row["event"] == completed_event
    )
    interval = [
        row
        for row in operator_events
        if started_ms <= float(row["elapsed_ms"]) <= completed_ms
    ]
    cohort_sizes = [int(row["max_aggregated_queries"]) for row in interval]
    active_cohort_sizes = [size for size in cohort_sizes if size >= 2]
    shared_queries = [int(row["shared_queries"]) for row in interval]
    return {
        "steps": len(interval),
        "max_aggregated_queries": max(cohort_sizes, default=0),
        "mean_aggregated_queries": (
            statistics.fmean(cohort_sizes) if cohort_sizes else 0.0
        ),
        "mean_active_cohort_queries": (
            statistics.fmean(active_cohort_sizes)
            if active_cohort_sizes
            else 0.0
        ),
        "mean_shared_queries": (
            statistics.fmean(shared_queries) if shared_queries else 0.0
        ),
        "shared_step_fraction": (
            sum(size >= 2 for size in cohort_sizes) / len(cohort_sizes)
            if cohort_sizes
            else 0.0
        ),
    }


def unique_evicted_keys(
    cpu_events: list[dict[str, Any]], lifecycle: str | None = None
) -> int:
    keys: set[str] = set()
    for row in cpu_events:
        grouped = row.get("evicted_keys_by_lifecycle")
        if grouped is not None:
            if lifecycle is None:
                for lifecycle_keys in grouped.values():
                    keys.update(lifecycle_keys)
            else:
                keys.update(grouped.get(lifecycle, ()))
            continue
        for item in row.get("evictions", ()):
            if lifecycle is None or item["lifecycle"] == lifecycle:
                keys.add(item["key"])
    return len(keys)


def labeled_metric(
    totals: Mapping[str, float],
    metric_name: str,
    label: str,
) -> float:
    return float(totals.get(f"{metric_name}:{(label,)!r}", 0.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/hwx/Documents/models/Qwen3-0.6B"),
    )
    parser.add_argument("--policy", choices=("lru", "cohort_lru"), required=True)
    parser.add_argument(
        "--gpu-lifecycle-eviction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--root-tokens", type=int, default=4096)
    parser.add_argument("--suffix-tokens", type=int, default=128)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--survivors", type=int, default=2)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--live-output-tokens", type=int, default=1024)
    parser.add_argument("--cooling-pulses", type=int, default=8)
    parser.add_argument("--cold-pulses", type=int, default=20)
    parser.add_argument("--hot-prefix-cooldown-steps", type=int, default=16)
    parser.add_argument("--pressure-sessions", type=int, default=6)
    parser.add_argument("--post-finish-pressure-sessions", type=int, default=6)
    parser.add_argument("--pressure-prompt-tokens", type=int, default=2048)
    parser.add_argument("--wave-output-tokens", type=int, default=128)
    parser.add_argument("--revisit-output-tokens", type=int, default=32)
    parser.add_argument("--gpu-sample-ms", type=int, default=200)
    parser.add_argument("--num-gpu-blocks", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--cpu-cache-gib", type=float, default=0.75)
    parser.add_argument("--revisit-distractors", type=int, default=0)
    parser.add_argument(
        "--attention-backend",
        choices=("FLASH_ATTN", "FORK_ATTN"),
        default="FLASH_ATTN",
    )
    parser.add_argument(
        "--fork-query-join",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    global MEMORY_ROOT_BLOCKS, STARTED, TRACE_PATH, WALL_ZERO
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if not 0 < args.survivors < args.branches:
        raise ValueError("survivors must be between zero and branches")

    trace_path = args.output.with_suffix(".trace.jsonl")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.unlink(missing_ok=True)
    TRACE_PATH = trace_path
    WALL_ZERO = time.time()
    install_cpu_eviction_observer()
    MEMORY_KEY_INFO.clear()
    MEMORY_EVICTION_METADATA.clear()
    TimelineLogger.instances.clear()
    profile_handler = ProfileHandler()
    logging.getLogger(PROFILE_LOGGER).addHandler(profile_handler)

    extra_config = {
        "cpu_bytes_to_use": int(args.cpu_cache_gib * 1024**3),
        "eviction_policy": args.policy,
        "fanout_offload": True,
        "fanout_gpu_lifecycle_eviction": args.gpu_lifecycle_eviction,
        "fanout_admission_window": (
            max(16, args.branches + args.revisit_distractors)
            if args.fork_query_join
            else 0
        ),
        "fanout_preemption_enabled": args.fork_query_join,
        "fanout_gpu_hotset_enabled": args.fork_query_join,
        "fanout_join_max_deferral_steps": int(args.fork_query_join),
        "fanout_arrival_wait_steps": 2 if args.fork_query_join else 0,
        "fanout_profile": True,
        "fanout_budget_blocks": 512,
        "fanout_allow_hot_prefix_backup": True,
        "fanout_hot_prefix_min_fanout": 4,
        "fanout_hot_prefix_min_reuse_blocks": 128,
        "fanout_hot_prefix_min_residency_steps": 4,
        "fanout_hot_prefix_cooldown_steps": (args.hot_prefix_cooldown_steps),
        "fanout_high_pressure_threshold": 0.90,
        "fanout_critical_pressure_threshold": 0.97,
    }
    # The experiment uses the vLLM default 16-token full-attention block.
    # Set this before engine startup so an EngineCore child inherits it.
    MEMORY_ROOT_BLOCKS = args.root_tokens // 16
    engine_args = AsyncEngineArgs(
        model=str(args.model),
        dtype="float16",
        attention_backend=args.attention_backend,
        enforce_eager=True,
        gpu_memory_utilization=0.75,
        num_gpu_blocks_override=args.num_gpu_blocks,
        max_model_len=max(
            6144,
            args.root_tokens
            + args.suffix_tokens
            + max(args.live_output_tokens, args.wave_output_tokens)
            + 32,
            args.pressure_prompt_tokens + 32,
        ),
        max_num_seqs=max(args.max_num_seqs, args.branches),
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        async_scheduling=False,
        disable_log_stats=False,
        enable_log_requests=False,
        kv_transfer_config={
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_load_failure_policy": "recompute",
            "kv_connector_extra_config": extra_config,
        },
        generation_config="vllm",
    )
    engine = AsyncLLM.from_engine_args(engine_args, stat_loggers=[TimelineLogger])
    MEMORY_ROOT_BLOCKS = args.root_tokens // int(
        engine.vllm_config.cache_config.block_size
    )
    gpu_stop = asyncio.Event()
    gpu_task: asyncio.Task[None] | None = None
    try:
        tokenizer = engine.get_tokenizer()
        vocab_size = int(tokenizer.vocab_size)
        sampling = SamplingParams(
            max_tokens=1,
            temperature=0,
            ignore_eos=True,
            output_kind=RequestOutputKind.DELTA,
        )
        root = token_ids(42, args.root_tokens, vocab_size)
        events: list[dict[str, Any]] = []
        token_samples: list[dict[str, Any]] = []
        gpu_samples: list[dict[str, Any]] = []
        STARTED = time.perf_counter()
        timeline_zero_ms = trace_elapsed_ms()
        gpu_task = asyncio.create_task(
            sample_gpu(
                gpu_samples,
                gpu_stop,
                interval_seconds=args.gpu_sample_ms / 1000,
            )
        )
        event(
            events,
            "experiment_started",
            policy=args.policy,
            gpu_lifecycle_eviction=args.gpu_lifecycle_eviction,
        )

        first_wave = launch_live_branches(
            engine,
            count=args.branches,
            root=root,
            suffix_tokens=args.suffix_tokens,
            vocab_size=vocab_size,
            warmup_tokens=args.warmup_output_tokens,
            output_tokens=args.live_output_tokens,
            token_samples=token_samples,
        )
        await wait_live_ready(first_wave)
        event(events, "wave1_running", active_branches=args.branches)
        event(events, "root_hot_checkpoint", active_branches=args.branches)

        retiring = first_wave[args.survivors :]
        survivors = first_wave[: args.survivors]
        await stop_live_branches(engine, retiring)
        event(events, "fanout_dropped", active_branches=args.survivors)
        await wait_live_tokens(
            survivors,
            args.warmup_output_tokens + min(args.cooling_pulses, 8),
        )
        event(events, "cooling_checkpoint", active_branches=args.survivors)

        await run_cold_pressure(
            engine,
            count=args.pressure_sessions,
            prompt_tokens=args.pressure_prompt_tokens,
            vocab_size=vocab_size,
            sampling_params=sampling,
            events=events,
        )
        event(
            events,
            "pressure_checkpoint",
            pressure_sessions=args.pressure_sessions,
            active_branches=args.survivors,
        )
        await pulse(
            engine,
            count=8,
            seed=320_000,
            vocab_size=vocab_size,
            sampling_params=sampling,
        )

        wave2_results = await run_generation_wave(
            engine,
            label="wave2",
            branches=args.branches - args.survivors,
            root=root,
            suffix_tokens=args.suffix_tokens,
            vocab_size=vocab_size,
            output_tokens=args.wave_output_tokens,
            events=events,
            token_samples=token_samples,
            active_base=args.survivors,
            index_start=args.survivors,
        )
        event(
            events,
            "root_reheated",
            active_branches=args.survivors,
        )
        await stop_live_branches(engine, survivors)
        event(events, "all_shared_branches_finished", active_branches=0)
        await run_cold_pressure(
            engine,
            count=args.post_finish_pressure_sessions,
            prompt_tokens=args.pressure_prompt_tokens,
            vocab_size=vocab_size,
            sampling_params=sampling,
            events=events,
            label="post_finish_pressure",
            seed_base=200_000,
        )
        await pulse(
            engine,
            count=args.cold_pulses,
            seed=340_000,
            vocab_size=vocab_size,
            sampling_params=sampling,
        )
        event(events, "cold_checkpoint", active_branches=0)

        revisit_results = await run_generation_wave(
            engine,
            label="revisit",
            branches=args.branches,
            root=root,
            suffix_tokens=args.suffix_tokens,
            vocab_size=vocab_size,
            output_tokens=args.revisit_output_tokens,
            events=events,
            token_samples=token_samples,
            distractors=args.revisit_distractors,
        )
        event(events, "experiment_completed")
        gpu_stop.set()
        await gpu_task

        trace_rows = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in trace_rows:
            row["elapsed_ms"] = float(row["elapsed_ms"]) - timeline_zero_ms
        profile_samples = [
            {key: value for key, value in row.items() if key != "kind"}
            for row in trace_rows
            if row["kind"] == "profile"
        ]
        cpu_events = [
            {key: value for key, value in row.items() if key != "kind"}
            for row in trace_rows
            if row["kind"] == "cpu"
        ]
        lifecycle_samples = [
            {key: value for key, value in row.items() if key != "kind"}
            for row in trace_rows
            if row["kind"] == "lifecycle"
        ]
        memory_events = [
            {key: value for key, value in row.items() if key != "kind"}
            for row in trace_rows
            if row["kind"] == "memory"
        ]
        kv_samples = [
            {
                **asdict(sample),
                "elapsed_ms": (sample.timestamp - STARTED) * 1000,
            }
            for logger in TimelineLogger.instances
            for sample in logger.samples
            if sample.timestamp >= STARTED
        ]
        for row in kv_samples:
            row.pop("timestamp")
        operator_events = [
            {
                "elapsed_ms": sample["elapsed_ms"],
                "event": "fork_plan",
                "kind": sample["fork_kind"],
                "active_queries": sample["running"],
                "shared_queries": sample["fork_shared_queries"],
                "max_aggregated_queries": sample[
                    "fork_max_aggregated_queries"
                ],
                "shared_ctas": sample["fork_shared_ctas"],
                "singleton_ctas": sample["fork_singleton_ctas"],
                "active_ctas": sample["fork_active_ctas"],
            }
            for sample in kv_samples
            if sample["fork_kind"]
        ]
        connector_totals: Counter[str] = Counter()
        for sample in kv_samples:
            connector_totals.update(sample["connector"])
        return {
            "schema_version": 1,
            "metadata": {
                "policy": args.policy,
                "gpu_lifecycle_eviction": args.gpu_lifecycle_eviction,
                "attention_backend": args.attention_backend,
                "fork_query_join": args.fork_query_join,
                "model": str(args.model),
                "gpu": gpu_description(),
                "root_tokens": args.root_tokens,
                "suffix_tokens": args.suffix_tokens,
                "branches": args.branches,
                "survivors": args.survivors,
                "warmup_output_tokens": args.warmup_output_tokens,
                "live_output_tokens": args.live_output_tokens,
                "cooling_pulses": args.cooling_pulses,
                "cold_pulses": args.cold_pulses,
                "hot_prefix_cooldown_steps": (args.hot_prefix_cooldown_steps),
                "pressure_sessions": args.pressure_sessions,
                "post_finish_pressure_sessions": (args.post_finish_pressure_sessions),
                "pressure_prompt_tokens": args.pressure_prompt_tokens,
                "wave_output_tokens": args.wave_output_tokens,
                "revisit_output_tokens": args.revisit_output_tokens,
                "gpu_sample_ms": args.gpu_sample_ms,
                "num_gpu_blocks": args.num_gpu_blocks,
                "max_num_seqs": args.max_num_seqs,
                "cpu_cache_gib": args.cpu_cache_gib,
                "revisit_distractors": args.revisit_distractors,
                "block_size_tokens": int(engine.vllm_config.cache_config.block_size),
                "lifecycle_config": extra_config,
            },
            "events": events,
            "profile_samples": profile_samples,
            "lifecycle_samples": lifecycle_samples,
            "memory_events": sorted(
                memory_events,
                key=lambda item: item["elapsed_ms"],
            ),
            "operator_events": sorted(
                operator_events,
                key=lambda item: item["elapsed_ms"],
            ),
            "kv_samples": sorted(kv_samples, key=lambda sample: sample["elapsed_ms"]),
            "cpu_events": sorted(cpu_events, key=lambda item: item["elapsed_ms"]),
            "token_samples": sorted(token_samples, key=lambda item: item["elapsed_ms"]),
            "gpu_samples": sorted(gpu_samples, key=lambda item: item["elapsed_ms"]),
            "wave_results": {
                "wave2": [asdict(result) for result in wave2_results],
                "revisit": [asdict(result) for result in revisit_results],
            },
            "connector_totals": dict(connector_totals),
            "summary": {
                "peak_kv_usage": max(
                    (sample["usage"] for sample in kv_samples), default=0
                ),
                "profile_steps": len(profile_samples),
                "fork_operator_steps": len(operator_events),
                "fork_max_aggregated_queries": max(
                    (
                        int(event["max_aggregated_queries"])
                        for event in operator_events
                    ),
                    default=0,
                ),
                "fork_mean_aggregated_queries": statistics.fmean(
                    int(event["max_aggregated_queries"])
                    for event in operator_events
                )
                if operator_events
                else 0.0,
                "cpu_evicted_blocks": sum(
                    event["evicted_blocks"] for event in cpu_events
                ),
                "cpu_evicted_hot": sum(event["evicted_hot"] for event in cpu_events),
                "cpu_evicted_cooling": sum(
                    event["evicted_cooling"] for event in cpu_events
                ),
                "cpu_evicted_cold": sum(event["evicted_cold"] for event in cpu_events),
                "cpu_evicted_unobserved": sum(
                    event["evicted_unobserved"] for event in cpu_events
                ),
                "cpu_evicted_unique_blocks": unique_evicted_keys(cpu_events),
                "cpu_evicted_unique_hot": unique_evicted_keys(cpu_events, "hot"),
                "cpu_evicted_unique_cooling": unique_evicted_keys(
                    cpu_events, "cooling"
                ),
                "cpu_evicted_unique_cold": unique_evicted_keys(cpu_events, "cold"),
                "load_operations": int(
                    connector_totals.get(
                        f"{_TransferMetricName.LOAD_SIZE}_count",
                        0,
                    )
                ),
                "load_bytes": int(
                    connector_totals.get(_TransferMetricName.LOAD_BYTES, 0)
                ),
                "reload_blocks_hot": int(
                    labeled_metric(
                        connector_totals,
                        _LifecycleMetricName.RELOAD_BLOCKS,
                        "hot",
                    )
                ),
                "reload_blocks_cooling": int(
                    labeled_metric(
                        connector_totals,
                        _LifecycleMetricName.RELOAD_BLOCKS,
                        "cooling",
                    )
                ),
                "reload_blocks_cold": int(
                    labeled_metric(
                        connector_totals,
                        _LifecycleMetricName.RELOAD_BLOCKS,
                        "cold",
                    )
                ),
                "gpu_evicted_hot": int(
                    labeled_metric(
                        connector_totals,
                        _LifecycleMetricName.GPU_EVICTED_BLOCKS,
                        "hot",
                    )
                ),
                "gpu_evicted_cooling": int(
                    labeled_metric(
                        connector_totals,
                        _LifecycleMetricName.GPU_EVICTED_BLOCKS,
                        "cooling",
                    )
                ),
                "gpu_evicted_cold": int(
                    labeled_metric(
                        connector_totals,
                        _LifecycleMetricName.GPU_EVICTED_BLOCKS,
                        "cold",
                    )
                ),
                "coalesced_load_waits": int(
                    connector_totals.get(
                        _LifecycleMetricName.COALESCED_LOAD_WAITS,
                        0,
                    )
                ),
                "max_hot_blocks": max(
                    (sample["hot_blocks"] for sample in lifecycle_samples),
                    default=0,
                ),
                "max_cooling_blocks": max(
                    (sample["cooling_blocks"] for sample in lifecycle_samples),
                    default=0,
                ),
                "max_cold_blocks": max(
                    (sample["cold_blocks"] for sample in lifecycle_samples),
                    default=0,
                ),
                "wave2": summarize_wave(wave2_results),
                "pressure_interval": summarize_interval(
                    events,
                    token_samples,
                    started_event="pressure_started",
                    completed_event="pressure_completed",
                ),
                "post_finish_pressure_interval": summarize_interval(
                    events,
                    token_samples,
                    started_event="post_finish_pressure_started",
                    completed_event="post_finish_pressure_completed",
                ),
                "wave2_effective": summarize_interval(
                    events,
                    token_samples,
                    started_event="wave2_started",
                    completed_event="wave2_completed",
                ),
                "revisit": summarize_wave(revisit_results),
                "revisit_operator": summarize_operator_interval(
                    events,
                    operator_events,
                    started_event="revisit_started",
                    completed_event="revisit_completed",
                ),
            },
        }
    finally:
        gpu_stop.set()
        if gpu_task is not None and not gpu_task.done():
            await gpu_task
        logging.getLogger(PROFILE_LOGGER).removeHandler(profile_handler)
        engine.shutdown()
        await asyncio.sleep(0.1)


def main() -> int:
    args = parse_args()
    payload = asyncio.run(main_async(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
