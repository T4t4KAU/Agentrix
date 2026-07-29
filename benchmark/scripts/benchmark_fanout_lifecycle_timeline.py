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

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
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


def install_cpu_eviction_observer() -> None:
    original_update = CPUOffloadingManager.update_eviction_metadata
    original_prepare = CPUOffloadingManager.prepare_store

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
        return output

    CPUOffloadingManager.update_eviction_metadata = update_eviction_metadata
    CPUOffloadingManager.prepare_store = prepare_store


@dataclass(frozen=True)
class KVSample:
    timestamp: float
    usage: float
    running: int
    waiting: int
    tool_waiting: int
    connector: dict[str, float]


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
        if isinstance(raw, Mapping):
            connector = {
                str(key): float(value)
                for key, value in raw.items()
                if isinstance(value, (int, float))
            }
        self.samples.append(
            KVSample(
                timestamp=time.perf_counter(),
                usage=float(scheduler_stats.kv_cache_usage),
                running=int(scheduler_stats.num_running_reqs),
                waiting=int(scheduler_stats.num_waiting_reqs),
                tool_waiting=int(scheduler_stats.num_skipped_waiting_reqs),
                connector=connector,
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
) -> None:
    event(events, "pressure_started", pressure_requests=count)
    for index in range(count):
        async for _ in engine.generate(
            {"prompt_token_ids": token_ids(100_000 + index, prompt_tokens, vocab_size)},
            sampling_params,
            f"pressure-{index}",
        ):
            pass
        event(
            events,
            "pressure_request_completed",
            pressure_completed=index + 1,
        )
    event(events, "pressure_completed", pressure_requests=count)


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

    results = await asyncio.gather(*(generate_branch(index) for index in indices))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/hwx/Documents/models/Qwen3-0.6B"),
    )
    parser.add_argument("--policy", choices=("lru", "cohort_lru"), required=True)
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
    parser.add_argument("--pressure-prompt-tokens", type=int, default=2048)
    parser.add_argument("--wave-output-tokens", type=int, default=128)
    parser.add_argument("--revisit-output-tokens", type=int, default=32)
    parser.add_argument("--gpu-sample-ms", type=int, default=200)
    parser.add_argument("--num-gpu-blocks", type=int, default=1024)
    parser.add_argument("--cpu-cache-gib", type=float, default=0.75)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    global STARTED, TRACE_PATH, WALL_ZERO
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
    TimelineLogger.instances.clear()
    profile_handler = ProfileHandler()
    logging.getLogger(PROFILE_LOGGER).addHandler(profile_handler)

    extra_config = {
        "cpu_bytes_to_use": int(args.cpu_cache_gib * 1024**3),
        "eviction_policy": args.policy,
        "fanout_offload": True,
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
    engine_args = AsyncEngineArgs(
        model=str(args.model),
        dtype="float16",
        attention_backend="FLASH_ATTN",
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
        max_num_seqs=max(32, args.branches + args.pressure_sessions + 8),
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
        event(events, "experiment_started", policy=args.policy)

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
        connector_totals: Counter[str] = Counter()
        for sample in kv_samples:
            connector_totals.update(sample["connector"])
        return {
            "schema_version": 1,
            "metadata": {
                "policy": args.policy,
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
                "pressure_prompt_tokens": args.pressure_prompt_tokens,
                "wave_output_tokens": args.wave_output_tokens,
                "revisit_output_tokens": args.revisit_output_tokens,
                "gpu_sample_ms": args.gpu_sample_ms,
                "num_gpu_blocks": args.num_gpu_blocks,
                "cpu_cache_gib": args.cpu_cache_gib,
                "block_size_tokens": int(engine.vllm_config.cache_config.block_size),
                "lifecycle_config": extra_config,
            },
            "events": events,
            "profile_samples": profile_samples,
            "lifecycle_samples": lifecycle_samples,
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
                "wave2_effective": summarize_interval(
                    events,
                    token_samples,
                    started_event="wave2_started",
                    completed_event="wave2_completed",
                ),
                "revisit": summarize_wave(revisit_results),
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
