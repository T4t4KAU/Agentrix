#!/usr/bin/env python3
"""Measure total vLLM KV-cache occupancy with Agentrix TTL trimming."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import statistics
import subprocess
import sys
import time
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
APPLICATION_SRC = REPO_ROOT / "application" / "src"
sys.path.insert(0, str(APPLICATION_SRC))

from agentrix_application.tool_kv_trimmer import (  # noqa: E402
    ToolKVTrimmer,
    ToolKVTrimmerConfig,
)
from agentrix_application.tool_ttl_predictor import (  # noqa: E402
    OnlineHorizonTTLPredictor,
    ToolTTLContext,
)
from vllm import SamplingParams  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.engine.protocol import StreamingInput  # noqa: E402
from vllm.outputs import RequestOutput  # noqa: E402
from vllm.sampling_params import RequestOutputKind  # noqa: E402
from vllm.v1.engine.async_llm import AsyncLLM  # noqa: E402
from vllm.v1.metrics.loggers import StatLoggerBase  # noqa: E402


@dataclass(frozen=True)
class KVSample:
    timestamp: float
    usage: float
    running: int
    waiting: int
    tool_waiting: int


class KVUsageLogger(StatLoggerBase):
    """Capture every scheduler KV occupancy update in the frontend."""

    instances: list["KVUsageLogger"] = []

    def __init__(self, vllm_config: Any, engine_index: int = 0) -> None:
        self.engine_index = engine_index
        self.vllm_config = vllm_config
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
        self.samples.append(
            KVSample(
                timestamp=time.perf_counter(),
                usage=float(scheduler_stats.kv_cache_usage),
                running=int(scheduler_stats.num_running_reqs),
                waiting=int(scheduler_stats.num_waiting_reqs),
                tool_waiting=int(scheduler_stats.num_skipped_waiting_reqs),
            )
        )

    def log_engine_initialized(self) -> None:
        return


@dataclass
class Session:
    request_id: str
    resume: asyncio.Event
    waiting: asyncio.Event
    task: asyncio.Task[list[RequestOutput]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/hwx/Documents/models/Qwen3-0.6B"),
    )
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--primary", type=int, default=4)
    parser.add_argument("--secondary", type=int, default=4)
    parser.add_argument("--secondary-arrival-ms", type=float, default=300.0)
    parser.add_argument("--tool-duration-ms", type=float, default=1200.0)
    parser.add_argument("--fixed-ttl-ms", type=float, default=500.0)
    parser.add_argument("--predicted-ttl-ms", type=float, default=100.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--nvml-interval-ms", type=float, default=50.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "benchmark/results/investigation_20260718/tool_kv_ttl_memory.json",
    )
    return parser.parse_args()


def physical_gpu_memory_mib() -> float | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return max(float(line.strip()) for line in result.stdout.splitlines())
    except (ValueError, TypeError):
        return None


async def monitor_physical_memory(
    stop: asyncio.Event,
    samples: list[tuple[float, float]],
    interval_ms: float,
) -> None:
    while not stop.is_set():
        value = await asyncio.to_thread(physical_gpu_memory_mib)
        if value is not None:
            samples.append((time.perf_counter(), value))
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval_ms / 1000)


def prompt_token_ids(
    *, session_number: int, length: int, vocab_size: int
) -> list[int]:
    """Create valid, deterministic prompts with no shared full cache blocks."""

    usable = max(1, vocab_size - 1024)
    return [
        1024 + ((session_number * 104729 + position * 8191) % usable)
        for position in range(length)
    ]


async def collect_session(
    engine: AsyncLLM,
    request_id: str,
    prompt_ids: list[int],
    continuation_ids: list[int],
    resume: asyncio.Event,
    waiting: asyncio.Event,
    sampling_params: SamplingParams,
) -> list[RequestOutput]:
    async def inputs() -> AsyncGenerator[StreamingInput, None]:
        yield StreamingInput(prompt=prompt_ids)
        await resume.wait()
        yield StreamingInput(prompt=continuation_ids)

    outputs: list[RequestOutput] = []
    async for output in engine.generate(inputs(), sampling_params, request_id):
        outputs.append(output)
        if not output.finished:
            waiting.set()
    return outputs


def launch_sessions(
    engine: AsyncLLM,
    *,
    mode: str,
    repeat: int,
    cohort: str,
    count: int,
    number_offset: int,
    prompt_tokens: int,
    vocab_size: int,
    sampling_params: SamplingParams,
) -> list[Session]:
    sessions = []
    for index in range(count):
        number = number_offset + index
        request_id = f"ttl-memory-{mode}-{repeat}-{cohort}-{index}"
        resume = asyncio.Event()
        waiting = asyncio.Event()
        prompt_ids = prompt_token_ids(
            session_number=number,
            length=prompt_tokens,
            vocab_size=vocab_size,
        )
        continuation_ids = prompt_token_ids(
            session_number=number + 1_000_000,
            length=16,
            vocab_size=vocab_size,
        )
        task = asyncio.create_task(
            collect_session(
                engine,
                request_id,
                prompt_ids,
                continuation_ids,
                resume,
                waiting,
                sampling_params,
            )
        )
        sessions.append(Session(request_id, resume, waiting, task))
    return sessions


async def wait_until_tool_waiting(sessions: list[Session]) -> None:
    await asyncio.wait_for(
        asyncio.gather(*(session.waiting.wait() for session in sessions)),
        timeout=120,
    )


def latest_usage(logger: KVUsageLogger) -> float:
    return logger.samples[-1].usage if logger.samples else 0.0


def time_window_samples(
    samples: list[KVSample], start: float, end: float
) -> list[KVSample]:
    before = [sample for sample in samples if sample.timestamp <= start]
    within = [sample for sample in samples if start < sample.timestamp < end]
    after = [sample for sample in samples if sample.timestamp >= end]
    result = []
    initial_usage = before[-1].usage if before else 0.0
    result.append(KVSample(start, initial_usage, 0, 0, 0))
    result.extend(within)
    final_usage = within[-1].usage if within else initial_usage
    if after and after[0].timestamp == end:
        final_usage = after[0].usage
    result.append(KVSample(end, final_usage, 0, 0, 0))
    return result


def usage_at(samples: list[KVSample], timestamp: float) -> float:
    prior = [sample.usage for sample in samples if sample.timestamp <= timestamp]
    return prior[-1] if prior else 0.0


def summarize_window(
    samples: list[KVSample], start: float, end: float, arrival: float
) -> dict[str, float | int]:
    window = time_window_samples(samples, start, end)
    auc = 0.0
    for left, right in zip(window, window[1:]):
        auc += left.usage * (right.timestamp - left.timestamp)
    peak = max(sample.usage for sample in window)
    return {
        "sample_count": len(window),
        "initial_primary_usage": usage_at(window, start),
        "usage_before_secondary": usage_at(window, arrival - 0.001),
        "usage_after_secondary": usage_at(window, arrival + 0.050),
        "peak_kv_usage": peak,
        "mean_kv_usage": auc / max(end - start, 1e-9),
        "kv_usage_area_seconds": auc,
    }


def make_predictor(predicted_ttl_ms: float, fixed_ttl_ms: float):
    predictor = OnlineHorizonTTLPredictor(
        confidence=0.7,
        fallback_ttl_ms=fixed_ttl_ms,
        min_ttl_ms=predicted_ttl_ms,
        min_training_samples=50,
    )
    # Seed with representative long tool calls. This activates the same online
    # model used in application code without depending on an external trace.
    training_context = ToolTTLContext(
        tool_family="memory_benchmark",
        argument_bytes=4096,
        kv_tokens=4096,
        pressure=0.5,
        active_tool_sessions=4,
        timeout_ms=10_000,
    )
    for _ in range(100):
        predictor.observe(training_context, 5000.0)
    return predictor


async def run_mode(
    engine: AsyncLLM,
    logger: KVUsageLogger,
    args: argparse.Namespace,
    mode: str,
    repeat: int,
    vocab_size: int,
) -> dict[str, Any]:
    reset_ok = await engine.reset_prefix_cache()
    await asyncio.sleep(0.1)
    sample_index = len(logger.samples)
    physical_samples: list[tuple[float, float]] = []
    monitor_stop = asyncio.Event()
    monitor_task = asyncio.create_task(
        monitor_physical_memory(
            monitor_stop, physical_samples, args.nvml_interval_ms
        )
    )
    sampling_params = SamplingParams(
        max_tokens=1,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
        temperature=0.0,
    )
    unique_offset = repeat * 100_000 + {"disabled": 0, "fixed": 20_000,
                                       "predicted": 40_000}[mode]
    primary = launch_sessions(
        engine,
        mode=mode,
        repeat=repeat,
        cohort="primary",
        count=args.primary,
        number_offset=unique_offset,
        prompt_tokens=args.prompt_tokens,
        vocab_size=vocab_size,
        sampling_params=sampling_params,
    )
    await wait_until_tool_waiting(primary)
    start = time.perf_counter()

    trim_results: list[dict[str, Any]] = []
    trimmer: ToolKVTrimmer | None = None
    ttl_prediction_ms: float | None = None
    contexts: dict[str, ToolTTLContext] = {}
    if mode != "disabled":
        predictor = make_predictor(
            args.predicted_ttl_ms, args.fixed_ttl_ms
        )
        context = ToolTTLContext(
            tool_family="memory_benchmark",
            argument_bytes=4096,
            kv_tokens=args.prompt_tokens,
            pressure=latest_usage(logger),
            active_tool_sessions=args.primary,
            timeout_ms=10_000,
        )
        ttl_prediction_ms = predictor.predict(context).ttl_ms

        async def trim_request(request_id: str) -> dict[str, Any]:
            result = dict(await engine.trim_tool_kv(request_id))
            result["completed_ms"] = (time.perf_counter() - start) * 1000
            trim_results.append(result)
            return result

        async def pressure_reader() -> float:
            return latest_usage(logger)

        trimmer = ToolKVTrimmer(
            pressure_reader=pressure_reader,
            trim_request=trim_request,
            config=ToolKVTrimmerConfig(
                enabled=True,
                grace_ms=args.fixed_ttl_ms,
                pressure_threshold=0.01,
                post_trim_recheck_ms=0,
                use_predicted_ttl=mode == "predicted",
            ),
            ttl_predictor=predictor,
        )
        for session in primary:
            contexts[session.request_id] = context
            assert trimmer.tool_started(
                session.request_id,
                session.request_id,
                ttl_context=context,
            )

    target_arrival = start + args.secondary_arrival_ms / 1000
    await asyncio.sleep(max(0.0, target_arrival - time.perf_counter()))
    arrival = time.perf_counter()
    secondary = launch_sessions(
        engine,
        mode=mode,
        repeat=repeat,
        cohort="secondary",
        count=args.secondary,
        number_offset=unique_offset + 10_000,
        prompt_tokens=args.prompt_tokens,
        vocab_size=vocab_size,
        sampling_params=sampling_params,
    )
    await wait_until_tool_waiting(secondary)

    target_end = start + args.tool_duration_ms / 1000
    await asyncio.sleep(max(0.0, target_end - time.perf_counter()))
    end = time.perf_counter()
    if trimmer is not None:
        await asyncio.gather(
            *(
                trimmer.tool_finished(
                    session.request_id,
                    session.request_id,
                    duration_ms=args.tool_duration_ms,
                )
                for session in primary
            )
        )

    resume_start = time.perf_counter()
    for session in primary + secondary:
        session.resume.set()
    await asyncio.wait_for(
        asyncio.gather(*(session.task for session in primary + secondary)),
        timeout=180,
    )
    resume_ms = (time.perf_counter() - resume_start) * 1000
    if trimmer is not None:
        await trimmer.close()
    monitor_stop.set()
    await monitor_task

    mode_samples = logger.samples[sample_index:]
    result: dict[str, Any] = {
        "mode": mode,
        "repeat": repeat,
        "prefix_cache_reset": reset_ok,
        "tool_window_ms": (end - start) * 1000,
        "secondary_arrival_ms": (arrival - start) * 1000,
        "resume_all_sessions_ms": resume_ms,
        "predicted_ttl_ms": ttl_prediction_ms,
        "physical_memory_mib_start": (
            physical_samples[0][1] if physical_samples else None
        ),
        "physical_memory_mib_peak": (
            max(value for _, value in physical_samples)
            if physical_samples
            else None
        ),
        "trim_results": trim_results,
        "window": summarize_window(mode_samples, start, end, arrival),
        "timeline": [
            {
                "elapsed_ms": (sample.timestamp - start) * 1000,
                "usage": sample.usage,
                "running": sample.running,
                "waiting": sample.waiting,
                "tool_waiting": sample.tool_waiting,
            }
            for sample in mode_samples
            if start - 0.1 <= sample.timestamp <= end + 0.1
        ],
    }
    if trimmer is not None:
        result["trimmer_stats"] = asdict(trimmer.stats)
    return result


def infer_total_blocks(results: list[dict[str, Any]]) -> float | None:
    estimates = []
    for mode in results:
        for trim in mode["trim_results"]:
            released = trim.get("released_block_references")
            before = trim.get("kv_cache_usage_before")
            after = trim.get("kv_cache_usage_after")
            if not isinstance(released, (int, float)):
                continue
            if not isinstance(before, (int, float)) or not isinstance(
                after, (int, float)
            ):
                continue
            delta = float(before) - float(after)
            if delta > 0:
                estimates.append(float(released) / delta)
    return statistics.median(estimates) if estimates else None


def kv_block_bytes(engine: AsyncLLM) -> int | None:
    config = engine.model_config.hf_text_config
    try:
        layers = int(config.num_hidden_layers)
        kv_heads = int(config.num_key_value_heads)
        attention_heads = int(config.num_attention_heads)
        head_dim = int(getattr(config, "head_dim", config.hidden_size // attention_heads))
        block_size = int(engine.vllm_config.cache_config.block_size)
        dtype_bytes = 2 if str(engine.model_config.dtype) in {
            "torch.float16",
            "torch.bfloat16",
        } else 4
    except (AttributeError, TypeError, ValueError):
        return None
    return block_size * layers * 2 * kv_heads * head_dim * dtype_bytes


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    KVUsageLogger.instances.clear()
    max_model_len = max(8192, args.prompt_tokens + 64)
    engine_args = AsyncEngineArgs(
        model=str(args.model),
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=args.primary + args.secondary + 4,
        enable_prefix_caching=True,
        disable_log_stats=True,
        enable_log_requests=False,
    )
    engine = AsyncLLM.from_engine_args(
        engine_args,
        stat_loggers=[KVUsageLogger],
    )
    try:
        if not KVUsageLogger.instances:
            raise RuntimeError("custom KV logger was not initialized")
        logger = KVUsageLogger.instances[0]
        tokenizer = engine.get_tokenizer()
        vocab_size = int(tokenizer.vocab_size)
        results = []
        for repeat in range(args.repeats):
            for mode in ("disabled", "fixed", "predicted"):
                print(f"running mode={mode} repeat={repeat}", flush=True)
                result = await run_mode(
                    engine, logger, args, mode, repeat, vocab_size
                )
                results.append(result)
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "peak": result["window"]["peak_kv_usage"],
                            "area": result["window"]["kv_usage_area_seconds"],
                            "physical_peak_mib": result[
                                "physical_memory_mib_peak"
                            ],
                            "trimmed": len(result["trim_results"]),
                        }
                    ),
                    flush=True,
                )

        total_blocks = infer_total_blocks(results)
        block_bytes = kv_block_bytes(engine)
        for result in results:
            window = result["window"]
            if total_blocks is not None and block_bytes is not None:
                pool_gib = total_blocks * block_bytes / 2**30
                window["peak_effective_kv_gib"] = (
                    window["peak_kv_usage"] * pool_gib
                )
                window["mean_effective_kv_gib"] = (
                    window["mean_kv_usage"] * pool_gib
                )
                window["effective_kv_gib_seconds"] = (
                    window["kv_usage_area_seconds"] * pool_gib
                )
        metadata = {
            "model": str(args.model),
            "gpu": subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "prompt_tokens": args.prompt_tokens,
            "primary_sessions": args.primary,
            "secondary_sessions": args.secondary,
            "tool_duration_ms": args.tool_duration_ms,
            "secondary_arrival_ms": args.secondary_arrival_ms,
            "fixed_ttl_ms": args.fixed_ttl_ms,
            "predicted_ttl_ms": args.predicted_ttl_ms,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": max_model_len,
            "block_size_tokens": engine.vllm_config.cache_config.block_size,
            "estimated_total_gpu_blocks": total_blocks,
            "estimated_block_bytes": block_bytes,
            "estimated_kv_pool_gib": (
                total_blocks * block_bytes / 2**30
                if total_blocks is not None and block_bytes is not None
                else None
            ),
        }
        return {"metadata": metadata, "runs": results}
    finally:
        engine.shutdown()
        await asyncio.sleep(0.1)


def main() -> None:
    args = parse_args()
    payload = asyncio.run(main_async(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
