#!/usr/bin/env python3
"""Run one arm of the equal-GPU-KV Agentrix full-stack LongBench experiment."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import statistics
import subprocess
import time
from collections import Counter
from collections.abc import AsyncGenerator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

from agentrix_application.prompt_compactor import (
    ToolResultBackingStore,
    ToolResultCompactionConfig,
    compact_tool_results,
)
from agentrix_application.tool_kv_trimmer import (
    ToolKVTrimmer,
    ToolKVTrimmerConfig,
)
from agentrix_application.tool_ttl_predictor import ToolTTLContext
from full_stack_agent import (
    AgentAction,
    BM25Index,
    Passage,
    build_passages,
    parse_action,
    render_search_results,
)
from longbench_qa import load_cases, score_answer
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import StreamingInput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerBase

from vllm import SamplingParams


@dataclass(frozen=True)
class KVSample:
    timestamp: float
    engine_index: int
    usage: float
    running: int
    waiting: int
    tool_waiting: int


class FullStackLogger(StatLoggerBase):
    instances: ClassVar[list[FullStackLogger]] = []

    def __init__(self, vllm_config: Any, engine_index: int = 0) -> None:
        self.vllm_config = vllm_config
        self.engine_index = engine_index
        self.samples: list[KVSample] = []
        self.connector_totals: Counter[str] = Counter()
        self.fork_observed_steps = 0
        self.fork_active_steps = 0
        self.fork_shared_ctas = 0
        self.fork_singleton_ctas = 0
        self.instances.append(self)

    def record(
        self,
        scheduler_stats: Any | None,
        iteration_stats: Any | None,
        mm_cache_stats: Any | None = None,
        engine_idx: int = 0,
    ) -> None:
        del iteration_stats, mm_cache_stats
        if scheduler_stats is None:
            return
        self.samples.append(
            KVSample(
                timestamp=time.perf_counter(),
                engine_index=engine_idx,
                usage=float(scheduler_stats.kv_cache_usage),
                running=int(scheduler_stats.num_running_reqs),
                waiting=int(scheduler_stats.num_waiting_reqs),
                tool_waiting=int(scheduler_stats.num_skipped_waiting_reqs),
            )
        )
        raw = getattr(scheduler_stats, "kv_connector_stats", None)
        if raw is not None and hasattr(raw, "reduce"):
            raw = raw.reduce()
        elif isinstance(raw, Mapping) and {"types", "data"} <= raw.keys():
            raw = OffloadingConnectorStats(data=dict(raw)).reduce()
        if isinstance(raw, Mapping):
            for key, value in raw.items():
                if isinstance(value, (int, float)):
                    self.connector_totals[str(key)] += value
        fork_stats = getattr(scheduler_stats, "fork_execution_stats", None)
        if fork_stats is not None:
            _, _, _, shared_ctas, singleton_ctas = fork_stats
            self.fork_observed_steps += 1
            if shared_ctas > 0:
                self.fork_active_steps += 1
            self.fork_shared_ctas += int(shared_ctas)
            self.fork_singleton_ctas += int(singleton_ctas)

    def log_engine_initialized(self) -> None:
        return


@dataclass
class Turn:
    text: str
    first_token_seconds: float
    latency_seconds: float
    cached_tokens: int


class StreamingAgentSession:
    """One resumable vLLM request with explicit tool-wait boundaries."""

    _END = object()

    def __init__(
        self,
        engine: AsyncLLM,
        request_id: str,
        sampling_params: SamplingParams,
    ) -> None:
        self.engine = engine
        self.request_id = request_id
        self.default_sampling_params = sampling_params
        self.inputs: asyncio.Queue[StreamingInput | object] = asyncio.Queue()
        self.turns: asyncio.Queue[Turn | BaseException] = asyncio.Queue()
        self._turn_started = 0.0
        self._collector = asyncio.create_task(self._collect())

    async def _input_stream(self) -> AsyncGenerator[StreamingInput, None]:
        while True:
            item = await self.inputs.get()
            if item is self._END:
                return
            assert isinstance(item, StreamingInput)
            yield item

    async def _collect(self) -> None:
        text_parts: list[str] = []
        first_token: float | None = None
        cached_tokens = 0
        try:
            async for output in self.engine.generate(
                self._input_stream(),
                self.default_sampling_params,
                self.request_id,
            ):
                if output.num_cached_tokens is not None:
                    cached_tokens = max(cached_tokens, output.num_cached_tokens)
                if not output.outputs:
                    continue
                completion = output.outputs[0]
                if completion.text:
                    first_token = first_token or time.perf_counter()
                    text_parts.append(completion.text)
                if completion.finish_reason is not None:
                    ended = time.perf_counter()
                    await self.turns.put(
                        Turn(
                            text="".join(text_parts).strip(),
                            first_token_seconds=(
                                (first_token or ended) - self._turn_started
                            ),
                            latency_seconds=ended - self._turn_started,
                            cached_tokens=cached_tokens,
                        )
                    )
                    text_parts = []
                    first_token = None
                    cached_tokens = 0
        except BaseException as error:  # noqa: BLE001
            await self.turns.put(error)

    async def generate_turn(
        self,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams | None = None,
        *,
        timeout_s: float = 600,
    ) -> Turn:
        self._turn_started = time.perf_counter()
        await self.inputs.put(
            StreamingInput(
                prompt=prompt_token_ids,
                sampling_params=sampling_params,
            )
        )
        result = await asyncio.wait_for(self.turns.get(), timeout=timeout_s)
        if isinstance(result, BaseException):
            raise result
        return result

    async def close(self) -> None:
        await self.inputs.put(self._END)
        await asyncio.wait_for(self._collector, timeout=120)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=("baseline", "agentrix", "full"), required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--question-limit", type=int, default=68)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--data-parallel-size", type=int, default=4)
    parser.add_argument("--num-gpu-blocks-override", type=int, default=2600)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--tool-rounds", type=int, default=4)
    parser.add_argument("--tool-delay-ms", type=float, default=800)
    parser.add_argument("--trim-grace-ms", type=float, default=100)
    parser.add_argument("--action-tokens", type=int, default=32)
    parser.add_argument("--answer-tokens", type=int, default=96)
    parser.add_argument("--offload-cpu-gib", type=float, default=16)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def continuation_ids(tokenizer: Any, instruction: str) -> list[int]:
    suffix = (
        "<|im_end|>\n<|im_start|>user\n"
        + instruction
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    return normalize_token_ids(tokenizer.encode(suffix, add_special_tokens=False))


def chat_ids(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return normalize_token_ids(encoded)


def normalize_token_ids(encoded: Any) -> list[int]:
    """Normalize Transformers list, tensor, and BatchEncoding outputs."""

    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if (
        isinstance(encoded, list)
        and len(encoded) == 1
        and isinstance(encoded[0], list)
    ):
        encoded = encoded[0]
    if not isinstance(encoded, list) or any(
        not isinstance(token_id, int) for token_id in encoded
    ):
        raise TypeError("tokenizer output did not contain a flat integer list")
    return encoded


def system_prompt(context: str) -> str:
    return (
        "You are a document research agent. The workspace document is included "
        "below, but its passage IDs are intentionally hidden. You must use the "
        "document_search and read_passage tools to obtain citable opaque IDs. "
        "Follow the requested action syntax exactly. Never invent a passage ID.\n\n"
        "WORKSPACE DOCUMENT:\n"
        + context
    )


def first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def choose_action(
    generated: str,
    *,
    expected: str,
    question: str,
    passages_by_id: dict[str, Passage],
    last_matches: list[tuple[Passage, float]],
) -> tuple[AgentAction, bool]:
    parsed = parse_action(generated)
    if (
        parsed is not None
        and parsed.kind == expected
        and (expected != "read" or parsed.value.upper() in passages_by_id)
    ):
        value = parsed.value.upper() if expected == "read" else parsed.value
        return AgentAction(expected, value), True
    if expected == "search":
        return AgentAction("search", question), False
    passage_id = (
        last_matches[0][0].passage_id
        if last_matches
        else next(iter(passages_by_id))
    )
    return AgentAction("read", passage_id), False


def router_stats(engine: AsyncLLM) -> dict[str, Any]:
    router = getattr(engine.engine_core, "prefix_router", None)
    if router is None:
        return {"active": False}
    fields = (
        "route_count",
        "routing_policy",
        "prefix_route_count",
        "affinity_route_count",
        "first_turn_balance_count",
        "followup_affinity_count",
        "followup_rebalance_count",
        "session_overload_rebalance_count",
        "session_cache_miss_rebalance_count",
        "session_id_route_count",
        "session_prefix_fallback_count",
        "unknown_turn_route_count",
        "graph_bound_route_count",
        "arrival_wave_count",
        "ordinary_bypass_route_count",
        "cohort_locked_route_count",
        "reload_intent_count",
        "reload_local_count",
        "reload_rebalanced_count",
        "reload_committed_count",
        "reload_failed_count",
        "reload_saved_tokens",
    )
    result = {"active": True}
    for field in fields:
        value = getattr(router, field, None)
        if isinstance(value, (int, float, str, bool)) or value is None:
            result[field] = value
    counts = getattr(router, "rank_route_counts", None)
    if counts is not None:
        result["rank_route_counts"] = list(counts)
    result["average_route_us"] = getattr(router, "average_route_us", None)
    return result


def kv_summary() -> dict[str, Any]:
    samples = [
        sample for logger in FullStackLogger.instances for sample in logger.samples
    ]
    connector = Counter()
    fork = Counter()
    for logger in FullStackLogger.instances:
        connector.update(logger.connector_totals)
        fork.update(
            {
                "observed_steps": logger.fork_observed_steps,
                "active_steps": logger.fork_active_steps,
                "shared_ctas": logger.fork_shared_ctas,
                "singleton_ctas": logger.fork_singleton_ctas,
            }
        )
    return {
        "sample_count": len(samples),
        "peak_usage_fraction": max((sample.usage for sample in samples), default=0),
        "mean_usage_fraction": (
            statistics.fmean(sample.usage for sample in samples) if samples else 0
        ),
        "peak_running_requests": max(
            (sample.running for sample in samples), default=0
        ),
        "peak_tool_waiting_requests": max(
            (sample.tool_waiting for sample in samples), default=0
        ),
        "connector_counters": dict(connector),
        "fork_execution": dict(fork),
    }


async def gpu_memory_sampler(
    stop: asyncio.Event, samples: list[dict[str, Any]]
) -> None:
    while not stop.is_set():
        result = await asyncio.to_thread(
            subprocess.run,
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            values = []
            for line in result.stdout.splitlines():
                try:
                    index, used = (part.strip() for part in line.split(",", 1))
                    values.append({"gpu": int(index), "used_mib": float(used)})
                except ValueError:
                    continue
            samples.append({"time": time.time(), "gpus": values})
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=0.2)


async def run_task(
    *,
    engine: AsyncLLM,
    tokenizer: Any,
    case: dict[str, Any],
    question: dict[str, Any],
    index: BM25Index,
    passages_by_id: dict[str, Passage],
    semaphore: asyncio.Semaphore,
    trimmer: ToolKVTrimmer | None,
    compaction_enabled: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    async with semaphore:
        task_started = time.perf_counter()
        logical_id = f"{case['case_id']}-{question['source_id']}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt(case["context"])},
            {
                "role": "user",
                "content": (
                    f"Question: {question['question']}\n"
                    "Begin with exactly: SEARCH: <concise query>"
                ),
            },
        ]
        action_params = SamplingParams(
            max_tokens=args.action_tokens,
            temperature=0,
            seed=args.seed,
            ignore_eos=True,
            output_kind=RequestOutputKind.DELTA,
        )
        answer_params = SamplingParams(
            max_tokens=args.answer_tokens,
            temperature=0,
            seed=args.seed,
            ignore_eos=True,
            output_kind=RequestOutputKind.DELTA,
        )
        session_number = 0
        session = StreamingAgentSession(
            engine, f"{logical_id}-s{session_number}", action_params
        )
        turns: list[Turn] = []
        tool_events: list[dict[str, Any]] = []
        compaction_reports: list[dict[str, Any]] = []
        backing_store = ToolResultBackingStore()
        turn = await session.generate_turn(chat_ids(tokenizer, messages))
        turns.append(turn)
        generated = turn.text
        last_matches: list[tuple[Passage, float]] = []
        read_ids: set[str] = set()

        for round_index in range(args.tool_rounds):
            expected = "search" if round_index % 2 == 0 else "read"
            action, model_action_valid = choose_action(
                generated,
                expected=expected,
                question=question["question"],
                passages_by_id=passages_by_id,
                last_matches=last_matches,
            )
            call_id = f"{logical_id}-tool-{round_index}"
            request_id = session.request_id
            tool_started = time.perf_counter()
            trim_scheduled = False
            if trimmer is not None:
                trim_scheduled = trimmer.tool_started(
                    logical_id,
                    request_id,
                    ttl_context=ToolTTLContext(
                        tool_family=action.kind,
                        argument_bytes=len(action.value.encode()),
                        kv_tokens=case["context_tokens"],
                        pressure=0.0,
                        active_tool_sessions=args.concurrency,
                        timeout_ms=10_000,
                    ),
                )
            await asyncio.sleep(args.tool_delay_ms / 1000)
            if action.kind == "search":
                last_matches = index.search(action.value, top_k=4)
                tool_result = render_search_results(last_matches)
                arguments = {"query": action.value, "top_k": 4}
                next_instruction = (
                    "Select one returned passage. Reply exactly: READ: <passage_id>"
                )
            else:
                passage = passages_by_id[action.value]
                read_ids.add(passage.passage_id)
                tool_result = passage.text
                arguments = {"passage_id": passage.passage_id}
                if round_index + 1 == args.tool_rounds:
                    next_instruction = (
                        "Cite one passage you read, then give the shortest answer. "
                        "Reply exactly: FINAL: <passage_id> || <short answer>"
                    )
                else:
                    next_instruction = (
                        "Find additional evidence. Reply exactly: SEARCH: <query>"
                    )
            if trimmer is not None:
                await trimmer.tool_finished(
                    logical_id,
                    request_id,
                    duration_ms=(time.perf_counter() - tool_started) * 1000,
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": generated,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": (
                                    "document_search"
                                    if action.kind == "search"
                                    else "read_passage"
                                ),
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": tool_result,
                }
            )
            messages.append({"role": "user", "content": next_instruction})
            tool_events.append(
                {
                    "round": round_index,
                    "tool": action.kind,
                    "argument": action.value,
                    "model_action_valid": model_action_valid,
                    "trim_scheduled": trim_scheduled,
                    "duration_ms": (time.perf_counter() - tool_started) * 1000,
                    "result_chars": len(tool_result),
                }
            )

            is_last = round_index + 1 == args.tool_rounds
            if is_last and compaction_enabled:
                await session.close()
                compacted = compact_tool_results(
                    messages,
                    config=ToolResultCompactionConfig(
                        enabled=True,
                        min_chars=256,
                        min_age_turns=1,
                        recoverable_tools=("document_search", "read_passage"),
                        resource_argument_names=("query", "passage_id"),
                    ),
                    backing_store=backing_store,
                )
                messages = compacted.messages
                compaction_reports.append(asdict(compacted.report))
                session_number += 1
                session = StreamingAgentSession(
                    engine, f"{logical_id}-s{session_number}", answer_params
                )
                turn = await session.generate_turn(
                    chat_ids(tokenizer, messages), answer_params
                )
            else:
                turn = await session.generate_turn(
                    continuation_ids(
                        tokenizer, f"TOOL RESULT:\n{tool_result}\n\n{next_instruction}"
                    ),
                    answer_params if is_last else action_params,
                )
            turns.append(turn)
            generated = turn.text

        await session.close()
        final_line = first_line(generated)
        final_payload = final_line.removeprefix("FINAL:").strip()
        final_parts = [part.strip() for part in final_payload.split("||", 1)]
        id_first = (
            len(final_parts) == 2 and final_parts[0] in passages_by_id
        )
        prediction = (
            final_parts[1]
            if id_first
            else final_parts[0] if final_parts else final_payload
        )
        cited_ids = {
            passage_id
            for passage_id in passages_by_id
            if passage_id in generated
        }
        quality = score_answer(prediction, question["answers"])
        valid_citation = bool(cited_ids & read_ids)
        return {
            "case_id": case["case_id"],
            "dataset": case["dataset"],
            "source_id": question["source_id"],
            "question": question["question"],
            "answers": question["answers"],
            "prediction": prediction,
            **quality,
            "success": quality["f1"] >= 0.5 and valid_citation,
            "agent_completed": final_line.startswith("FINAL:") and valid_citation,
            "valid_citation": valid_citation,
            "cited_passage_ids": sorted(cited_ids),
            "read_passage_ids": sorted(read_ids),
            "tool_events": tool_events,
            "tool_call_valid_rate": statistics.fmean(
                event["model_action_valid"] for event in tool_events
            ),
            "compaction_reports": compaction_reports,
            "compaction_saved_chars": sum(
                report["before_chars"] - report["after_chars"]
                for report in compaction_reports
            ),
            "turn_count": len(turns),
            "mean_ttft_seconds": statistics.fmean(
                turn.first_token_seconds for turn in turns
            ),
            "latency_seconds": time.perf_counter() - task_started,
            "cached_tokens": sum(turn.cached_tokens for turn in turns),
        }


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if args.tool_rounds < 2 or args.tool_rounds % 2:
        raise ValueError("tool_rounds must be an even integer >= 2")
    cases = load_cases(args.cases)
    manifest_sha = hashlib.sha256(args.cases.read_bytes()).hexdigest()
    tasks: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (case, question)
        for case in cases
        for question in case["questions"]
    ][: args.question_limit]
    FullStackLogger.instances.clear()
    optimized = args.arm != "baseline"
    full = args.arm == "full"
    kv_transfer_config = None
    if full:
        kv_transfer_config = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_load_failure_policy": "recompute",
            "kv_connector_extra_config": {
                "cpu_bytes_to_use": int(args.offload_cpu_gib * 1024**3),
                "fanout_offload": True,
                "fanout_profile": True,
                "fanout_budget_blocks": 256,
                "fanout_allow_hot_prefix_backup": True,
            },
        }
    engine_args = AsyncEngineArgs(
        model=str(args.model),
        dtype="bfloat16",
        attention_backend="FORK_ATTN" if optimized else "FLASH_ATTN",
        data_parallel_size=args.data_parallel_size,
        data_parallel_backend="mp",
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        async_scheduling=False,
        enforce_eager=True,
        disable_log_stats=False,
        enable_log_requests=False,
        kv_transfer_config=kv_transfer_config,
        generation_config="vllm",
    )
    engine = AsyncLLM.from_engine_args(engine_args, stat_loggers=[FullStackLogger])
    stop_sampler = asyncio.Event()
    memory_samples: list[dict[str, Any]] = []
    sampler_task = asyncio.create_task(
        gpu_memory_sampler(stop_sampler, memory_samples)
    )
    trimmer: ToolKVTrimmer | None = None
    trim_results: list[dict[str, Any]] = []
    try:
        tokenizer = engine.get_tokenizer()

        async def pressure_reader() -> float:
            return max(
                (
                    logger.samples[-1].usage
                    for logger in FullStackLogger.instances
                    if logger.samples
                ),
                default=0.0,
            )

        async def trim_request(request_id: str) -> Mapping[str, Any]:
            result = dict(await engine.trim_tool_kv(request_id))
            trim_results.append(result)
            return result

        if full:
            trimmer = ToolKVTrimmer(
                pressure_reader=pressure_reader,
                trim_request=trim_request,
                config=ToolKVTrimmerConfig(
                    enabled=True,
                    grace_ms=args.trim_grace_ms,
                    pressure_threshold=0.0,
                    post_trim_recheck_ms=0,
                    use_predicted_ttl=False,
                ),
            )
        indexes: dict[str, BM25Index] = {}
        passages: dict[str, dict[str, Passage]] = {}
        for case in cases:
            built = build_passages(case["context"])
            indexes[case["case_id"]] = BM25Index(built)
            passages[case["case_id"]] = {
                passage.passage_id: passage for passage in built
            }
        semaphore = asyncio.Semaphore(args.concurrency)
        started = time.perf_counter()
        pending_tasks = [
            asyncio.create_task(
                run_task(
                    engine=engine,
                    tokenizer=tokenizer,
                    case=case,
                    question=question,
                    index=indexes[case["case_id"]],
                    passages_by_id=passages[case["case_id"]],
                    semaphore=semaphore,
                    trimmer=trimmer,
                    compaction_enabled=full,
                    args=args,
                )
            )
            for case, question in tasks
        ]
        results = []
        for completed_index, pending in enumerate(
            asyncio.as_completed(pending_tasks), start=1
        ):
            result = await pending
            results.append(result)
            print(
                "agent task completed "
                f"{completed_index}/{len(pending_tasks)} "
                f"case={result['case_id']} source={result['source_id']}",
                flush=True,
            )
        wall = time.perf_counter() - started
        if trimmer is not None:
            await trimmer.close()
        block_size = int(engine.vllm_config.cache_config.block_size)
        metadata = {
            "arm": args.arm,
            "model": str(args.model),
            "dtype": "bfloat16",
            "attention_backend": "FORK_ATTN" if optimized else "FLASH_ATTN",
            "trimmer": full,
            "offload": full,
            "router": optimized,
            "compaction": full,
            "forkattention": optimized,
            "data_parallel_size": args.data_parallel_size,
            "num_gpu_blocks_override": args.num_gpu_blocks_override,
            "total_gpu_blocks": (
                args.num_gpu_blocks_override * args.data_parallel_size
            ),
            "block_size_tokens": block_size,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "enforce_eager": True,
            "async_scheduling": False,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "case_manifest_sha256": manifest_sha,
            "question_limit": len(tasks),
            "tool_rounds": args.tool_rounds,
            "tool_delay_ms": args.tool_delay_ms,
            "action_tokens": args.action_tokens,
            "answer_tokens": args.answer_tokens,
            "trim_grace_ms": args.trim_grace_ms,
            "offload_cpu_gib": args.offload_cpu_gib if full else 0,
        }
        return {
            "schema_version": 1,
            "metadata": metadata,
            "wall_seconds": wall,
            "tasks_per_second": len(results) / wall,
            "mean_exact_match": statistics.fmean(
                result["exact_match"] for result in results
            ),
            "mean_f1": statistics.fmean(result["f1"] for result in results),
            "success_rate": statistics.fmean(
                result["success"] for result in results
            ),
            "agent_completion_rate": statistics.fmean(
                result["agent_completed"] for result in results
            ),
            "valid_citation_rate": statistics.fmean(
                result["valid_citation"] for result in results
            ),
            "tool_call_valid_rate": statistics.fmean(
                result["tool_call_valid_rate"] for result in results
            ),
            "mean_task_latency_seconds": statistics.fmean(
                result["latency_seconds"] for result in results
            ),
            "mean_turn_ttft_seconds": statistics.fmean(
                result["mean_ttft_seconds"] for result in results
            ),
            "compaction_saved_chars": sum(
                result["compaction_saved_chars"] for result in results
            ),
            "trimmer_stats": asdict(trimmer.stats) if trimmer is not None else {},
            "trim_results": trim_results,
            "router_stats": router_stats(engine),
            "kv": kv_summary(),
            "gpu_memory_samples": memory_samples,
            "results": results,
        }
    finally:
        stop_sampler.set()
        with contextlib.suppress(Exception):
            await sampler_task
        engine.shutdown()
        await asyncio.sleep(0.1)


def main() -> int:
    args = parse_args()
    result = asyncio.run(main_async(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {
                "results", "gpu_memory_samples", "trim_results"
            }},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
