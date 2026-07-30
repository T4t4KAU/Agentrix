#!/usr/bin/env python3
"""Run the live HotpotQA StateGraph+Send Agent on an in-process vLLM engine.

Unlike the ordinary OpenAI-compatible runner, branch requests stay resumable
while their tools execute.  This lets one measured run exercise ForkAttention,
prompt compaction, tool-KV TTL prediction, and KV offload on the same online
LangGraph workflow.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agentrix_application.tool_kv_trimmer import ToolKVTrimmer, ToolKVTrimmerConfig
from agentrix_application.tool_ttl_predictor import (
    OnlineHorizonTTLPredictor,
    ToolTTLContext,
)
from benchmark_full_stack_agent import (
    FullStackLogger,
    StreamingAgentSession,
    gpu_memory_sampler,
    normalize_token_ids,
)
from hotpot import (
    HotpotExample,
    load_hotpot,
)
from hotpot import (
    evaluate_predictions as evaluate_hotpot_predictions,
)
from langgraph_runner import (
    HotpotRAG,
    TraceRecorder,
    _hotpot_final_output,
    _normalize_supporting_facts,
    build_graph,
    compact_rag_results,
    format_rag_results,
    summarize_prompt_compaction,
    summarize_rag_reuse,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

from vllm import SamplingParams


def _message(content: str, tool_calls: list[Any] | None = None) -> Any:
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def _parse_tool_calls(text: str) -> list[Any]:
    """Parse the Qwen tool-call envelope without changing fallback semantics."""

    candidates = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if not candidates:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            candidates = [stripped]
    calls: list[Any] = []
    for index, candidate in enumerate(candidates):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        name = value.get("name")
        arguments = value.get("arguments", {})
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            continue
        calls.append(
            SimpleNamespace(
                id=f"online-tool-{index}",
                function=SimpleNamespace(
                    name=name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                ),
            )
        )
    return calls


def _chat_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    if tools:
        kwargs["tools"] = tools
    return normalize_token_ids(tokenizer.apply_chat_template(messages, **kwargs))


def _reflection_ids(tokenizer: Any, messages: list[dict[str, Any]]) -> list[int]:
    tool_message = messages[-2]
    user_message = messages[-1]
    suffix = (
        "<|im_end|>\n"
        "<|im_start|>tool\n"
        + str(tool_message["content"])
        + "<|im_end|>\n"
        "<|im_start|>user\n"
        + str(user_message["content"])
        + "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
    return normalize_token_ids(tokenizer.encode(suffix, add_special_tokens=False))


class InProcessAgentRuntime:
    """Runtime contract consumed by ``langgraph_runner.build_graph``."""

    def __init__(
        self,
        *,
        engine: AsyncLLM,
        tokenizer: Any,
        rag: HotpotRAG,
        recorder: TraceRecorder,
        concurrency: int,
        prompt_compaction: bool,
        tool_delay_ms: float,
        ttl_mode: str,
        ttl_ms: float,
        pressure_threshold: float,
        predictor: OnlineHorizonTTLPredictor | None,
        seed: int,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.rag = rag
        self.recorder = recorder
        self.semaphore = asyncio.Semaphore(concurrency)
        self.rag_format = "plain"
        self.prompt_compaction = prompt_compaction
        self.workload = "hotpot"
        self.tool_delay_ms = tool_delay_ms
        self.seed = seed
        self.sessions: dict[tuple[str, int], StreamingAgentSession] = {}
        self.session_prompt_tokens: dict[tuple[str, int], int] = {}
        self.trim_results: list[dict[str, Any]] = []
        self.prediction_events: list[dict[str, Any]] = []
        self.tool_call_outputs = 0
        self.valid_tool_calls = 0
        self.predictor = predictor
        self.trimmer: ToolKVTrimmer | None = None
        if ttl_mode != "disabled":
            self.trimmer = ToolKVTrimmer(
                pressure_reader=self._pressure,
                trim_request=self._trim,
                config=ToolKVTrimmerConfig(
                    enabled=True,
                    grace_ms=ttl_ms,
                    pressure_threshold=pressure_threshold,
                    post_trim_recheck_ms=0,
                    use_predicted_ttl=ttl_mode == "predicted",
                ),
                ttl_predictor=predictor,
            )

    async def _pressure(self) -> float:
        return max(
            (
                logger.samples[-1].usage
                for logger in FullStackLogger.instances
                if logger.samples
            ),
            default=0.0,
        )

    async def _trim(self, request_id: str) -> dict[str, Any]:
        started_ms = (time.perf_counter() - self.recorder.started) * 1000
        result = dict(await self.engine.trim_tool_kv(request_id))
        recorded = {"started_ms": started_ms, "request_id": request_id, **result}
        self.trim_results.append(recorded)
        await self.recorder.add({"kind": "kv_trim", **recorded})
        return result

    def _params(self, max_tokens: int) -> SamplingParams:
        return SamplingParams(
            max_tokens=max_tokens,
            temperature=0,
            seed=self.seed,
            output_kind=RequestOutputKind.DELTA,
        )

    async def complete(
        self,
        *,
        case_id: str,
        stage: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        branch_id: int | None = None,
    ) -> Any:
        del tool_choice
        key = (case_id, branch_id) if branch_id is not None else None
        started_ms = (time.perf_counter() - self.recorder.started) * 1000
        started = time.perf_counter()
        params = self._params(max_tokens)
        prompt_tokens = 0

        async with self.semaphore:
            if stage == "branch_reflect" and key is not None:
                session = self.sessions.pop(key)
                prompt_tokens = self.session_prompt_tokens.pop(key)
                try:
                    turn = await session.generate_turn(
                        _reflection_ids(self.tokenizer, messages),
                        params,
                    )
                finally:
                    await session.close()
            else:
                prompt_ids = _chat_ids(self.tokenizer, messages, tools)
                prompt_tokens = len(prompt_ids)
                request_id = (
                    f"hotpot-{case_id}-branch-{branch_id}"
                    if key is not None
                    else f"hotpot-{case_id}-{stage}-{time.monotonic_ns()}"
                )
                session = StreamingAgentSession(self.engine, request_id, params)
                keep_open = stage == "tool_select" and key is not None
                try:
                    turn = await session.generate_turn(prompt_ids, params)
                except BaseException:
                    await session.close()
                    raise
                if keep_open:
                    self.sessions[key] = session
                    self.session_prompt_tokens[key] = prompt_tokens
                else:
                    await session.close()

        text = turn.text.strip()
        parsed_calls = _parse_tool_calls(text) if tools else []
        if tools:
            self.tool_call_outputs += 1
            if parsed_calls:
                self.valid_tool_calls += 1
        response = _message(text, parsed_calls)
        await self.recorder.add(
            {
                "kind": "llm",
                "case_id": case_id,
                "stage": stage,
                "branch_id": branch_id,
                "started_ms": started_ms,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "ttft_ms": turn.first_token_seconds * 1000,
                "request": {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "tools": tools,
                },
                "response": {"content": text},
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "cached_tokens": turn.cached_tokens,
                },
            }
        )
        return response

    async def run_tool(
        self,
        case_id: str,
        branch_id: int,
        name: str,
        arguments: dict[str, Any],
        delay_ms: float = 0,
        known_results: list[dict[str, Any]] | None = None,
    ) -> str:
        key = (case_id, branch_id)
        session = self.sessions[key]
        query = str(arguments.get("query") or "")
        top_k = max(1, min(int(arguments.get("top_k") or 3), 10))
        started_ms = (time.perf_counter() - self.recorder.started) * 1000
        started = time.perf_counter()
        actual_delay_ms = delay_ms if delay_ms > 0 else self.tool_delay_ms
        trim_scheduled = False
        prediction_payload = None
        context = ToolTTLContext(
            tool_family=name,
            argument_bytes=len(json.dumps(arguments).encode()),
            kv_tokens=self.session_prompt_tokens.get(key, 0),
            pressure=await self._pressure(),
            active_tool_sessions=len(self.sessions),
            shared_prefix_ratio=0.9,
            timeout_ms=actual_delay_ms * 2,
        )
        if self.predictor is not None:
            prediction = self.predictor.predict(context)
            prediction_payload = {
                "kind": "ttl_prediction",
                "case_id": case_id,
                "branch_id": branch_id,
                "stage": "tool",
                "started_ms": started_ms,
                "ttl_ms": prediction.ttl_ms,
                "trained_samples": prediction.trained_samples,
                "used_fallback": prediction.used_fallback,
                "survival_probabilities": list(prediction.survival_probabilities),
            }
            self.prediction_events.append(prediction_payload)
            await self.recorder.add(prediction_payload)
        if self.trimmer is not None:
            trim_scheduled = self.trimmer.tool_started(
                f"{case_id}:{branch_id}",
                session.request_id,
                ttl_context=context,
            )
        if actual_delay_ms > 0:
            await asyncio.sleep(actual_delay_ms / 1000)
        if name == "paragraph_search":
            result: Any = self.rag.search(query, top_k, case_id=case_id)
        else:
            result = {"error": f"unsupported tool: {name}"}
        if self.trimmer is not None:
            await self.trimmer.tool_finished(
                f"{case_id}:{branch_id}",
                session.request_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        compaction = None
        if isinstance(result, list) and self.prompt_compaction and known_results:
            compacted = compact_rag_results(
                result,
                known_results=known_results,
                rag_format=self.rag_format,
            )
            content = compacted.text
            compaction = asdict(compacted.report)
            compaction["saved_chars"] = compacted.report.saved_chars
        elif isinstance(result, list):
            content = format_rag_results(result, rag_format=self.rag_format)
        else:
            content = json.dumps(result, ensure_ascii=False)
        await self.recorder.add(
            {
                "kind": "tool",
                "case_id": case_id,
                "branch_id": branch_id,
                "stage": "tool",
                "tool": name,
                "arguments": arguments,
                "started_ms": started_ms,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "result": result,
                "compaction": compaction,
                "trim_scheduled": trim_scheduled,
                "ttl_prediction": prediction_payload,
            }
        )
        return content

    async def close(self) -> None:
        if self.trimmer is not None:
            await self.trimmer.close()
        sessions = list(self.sessions.values())
        self.sessions.clear()
        self.session_prompt_tokens.clear()
        await asyncio.gather(
            *(session.close() for session in sessions),
            return_exceptions=True,
        )


def _load_cases(
    hotpot_path: Path,
    manifest: Path,
    *,
    sample_index: int,
    cases: int,
) -> tuple[list[HotpotExample], list[HotpotExample]]:
    all_examples = load_hotpot(hotpot_path)
    by_id = {example.example_id: example for example in all_examples}
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][sample_index : sample_index + cases]
    selected = [by_id[record["id"]] for record in records]
    rag_examples: list[HotpotExample] = []
    for record, example in zip(records, selected, strict=True):
        paragraphs = list(example.context)
        seen = {paragraph.title for paragraph in paragraphs}
        for donor_id in record.get("distractor_ids", []):
            donor = by_id[str(donor_id)]
            for paragraph in donor.context:
                if paragraph.title not in seen:
                    paragraphs.append(paragraph)
                    seen.add(paragraph.title)
        rag_examples.append(replace(example, context=tuple(paragraphs)))
    return selected, rag_examples


def _kv_payload(started: float) -> dict[str, Any]:
    samples = [
        {
            **asdict(sample),
            "time_ms": (sample.timestamp - started) * 1000,
        }
        for logger in FullStackLogger.instances
        for sample in logger.samples
    ]
    connector: Counter[str] = Counter()
    fork: Counter[str] = Counter()
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
        "samples": samples,
        "peak_usage_fraction": max(
            (sample["usage"] for sample in samples), default=0.0
        ),
        "mean_usage_fraction": (
            statistics.fmean(sample["usage"] for sample in samples)
            if samples
            else 0.0
        ),
        "connector_counters": dict(connector),
        "fork_execution": dict(fork),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    selected, rag_examples = _load_cases(
        args.hotpot_path,
        args.hotpot_case_file,
        sample_index=args.sample_index,
        cases=args.cases,
    )
    predictor: OnlineHorizonTTLPredictor | None
    if args.predictor_state is not None:
        predictor = OnlineHorizonTTLPredictor.load(args.predictor_state)
    elif args.ttl_mode in {"fixed", "predicted"}:
        predictor = OnlineHorizonTTLPredictor(
            min_training_samples=args.predictor_min_samples,
            fallback_ttl_ms=args.ttl_ms,
            min_ttl_ms=args.predicted_min_ttl_ms,
        )
    else:
        predictor = None
    kv_transfer_config = None
    if args.offload_cpu_gib > 0:
        kv_transfer_config = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_load_failure_policy": "recompute",
            "kv_connector_extra_config": {
                "cpu_bytes_to_use": int(args.offload_cpu_gib * 1024**3),
                "block_size": 16,
                "eviction_policy": "cohort_lru",
                "offload_prompt_only": True,
                "fanout_offload": True,
                "fanout_profile": True,
                "fanout_allow_hot_prefix_backup": True,
            },
        }
    FullStackLogger.instances.clear()
    engine_args = AsyncEngineArgs(
        model=str(args.model),
        dtype="bfloat16",
        attention_backend=args.attention_backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_gpu_blocks_override=(
            args.num_gpu_blocks if args.num_gpu_blocks > 0 else None
        ),
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        async_scheduling=False,
        enforce_eager=args.enforce_eager,
        disable_log_stats=False,
        enable_log_requests=False,
        kv_transfer_config=kv_transfer_config,
        generation_config="vllm",
    )
    engine_started = time.perf_counter()
    engine = AsyncLLM.from_engine_args(engine_args, stat_loggers=[FullStackLogger])
    tokenizer = engine.get_tokenizer()
    recorder = TraceRecorder(str(args.model))
    runtime = InProcessAgentRuntime(
        engine=engine,
        tokenizer=tokenizer,
        rag=HotpotRAG(rag_examples),
        recorder=recorder,
        concurrency=args.concurrency,
        prompt_compaction=args.prompt_compaction,
        tool_delay_ms=args.tool_delay_ms,
        ttl_mode=args.ttl_mode,
        ttl_ms=args.ttl_ms,
        pressure_threshold=args.pressure_threshold,
        predictor=predictor,
        seed=args.seed,
    )
    stop_sampler = asyncio.Event()
    gpu_samples: list[dict[str, Any]] = []
    sampler = asyncio.create_task(gpu_memory_sampler(stop_sampler, gpu_samples))
    periodic_kv_samples: list[dict[str, Any]] = []

    async def sample_kv_periodically() -> None:
        while not stop_sampler.is_set():
            latest = [
                logger.samples[-1]
                for logger in FullStackLogger.instances
                if logger.samples
            ]
            periodic_kv_samples.append(
                {
                    "time_ms": (time.perf_counter() - engine_started) * 1000,
                    "usage": max(
                        (sample.usage for sample in latest),
                        default=0.0,
                    ),
                    "running": sum(sample.running for sample in latest),
                    "waiting": sum(sample.waiting for sample in latest),
                    "tool_waiting": sum(
                        sample.tool_waiting for sample in latest
                    ),
                }
            )
            await asyncio.sleep(0.05)

    kv_sampler = asyncio.create_task(sample_kv_periodically())
    outputs: list[dict[str, Any]] = []
    graph_started = time.perf_counter()

    async def invoke(example: HotpotExample) -> dict[str, Any]:
        graph = build_graph(
            runtime,
            args.branches,
            {
                "planner": args.planner_tokens,
                "tool_select": args.tool_tokens,
                "reflect": args.reflect_tokens,
                "reduce": args.reduce_tokens,
            },
            bootstrap_chunks=args.bootstrap_chunks,
            bootstrap_max_chars=args.bootstrap_max_chars,
            tool_delay_ms=args.tool_delay_ms,
            workload="hotpot",
            branch_min=args.branches,
            branch_max=args.branches,
            tool_delay_profile=args.tool_delay_profile,
            seed=args.seed,
        )
        return await graph.ainvoke(
            {
                "case_id": example.example_id,
                "task": "HotpotQA question: " + example.question,
                "context_query": example.question,
                "branches": args.branches,
                "branch_roles": [],
                "branch_outputs": [],
            }
        )

    try:
        semaphore = asyncio.Semaphore(args.case_concurrency)

        async def bounded(example: HotpotExample) -> dict[str, Any]:
            async with semaphore:
                return await invoke(example)

        outputs = await asyncio.gather(*(bounded(example) for example in selected))
    finally:
        await runtime.close()
        stop_sampler.set()
        await sampler
        await kv_sampler
        engine.shutdown()

    wall_s = time.perf_counter() - graph_started
    answer_map: dict[str, str] = {}
    fact_map: dict[str, list[list[Any]]] = {}
    output_records = []
    for output in outputs:
        final = output.get("answer")
        if not isinstance(final, dict):
            final = _hotpot_final_output(str(final or ""))
        case_id = output["case_id"]
        answer_map[case_id] = str(final.get("answer") or "")
        fact_map[case_id] = _normalize_supporting_facts(
            final.get("supporting_facts", [])
        )
        output_records.append(
            {
                "case_id": case_id,
                "answer": answer_map[case_id],
                "supporting_facts": fact_map[case_id],
                "branch_count": len(output.get("branch_outputs", [])),
            }
        )
    predictions = {"answer": answer_map, "sp": fact_map}
    metadata = {
        "mode": "live_in_process_resumable",
        "model": str(args.model),
        "attention_backend": args.attention_backend,
        "data_parallel_size": 1,
        "cases": args.cases,
        "branches": args.branches,
        "case_concurrency": args.case_concurrency,
        "prompt_compaction": args.prompt_compaction,
        "ttl_mode": args.ttl_mode,
        "ttl_ms": args.ttl_ms,
        "pressure_threshold": args.pressure_threshold,
        "offload_cpu_gib": args.offload_cpu_gib,
        "prefix_caching": True,
        "chunked_prefill": True,
        "enforce_eager": args.enforce_eager,
        "max_model_len": args.max_model_len,
        "num_gpu_blocks": args.num_gpu_blocks or None,
        "tool_delay_ms": args.tool_delay_ms,
        "tool_delay_profile": args.tool_delay_profile,
        "wall_s": wall_s,
        "engine_startup_s": graph_started - engine_started,
        "hotpot_sha256": hashlib.sha256(args.hotpot_path.read_bytes()).hexdigest(),
    }
    metadata["rag_reuse"] = summarize_rag_reuse(recorder.events)
    metadata["prompt_compaction_report"] = summarize_prompt_compaction(
        recorder.events
    )
    if predictor is not None and args.predictor_output is not None:
        predictor.save(args.predictor_output)
    payload = recorder.payload(metadata)
    payload.update(
        {
            "outputs": output_records,
            "predictions": predictions,
            "evaluation": evaluate_hotpot_predictions(selected, predictions),
            "kv": _kv_payload(engine_started),
            "periodic_kv_samples": periodic_kv_samples,
            "gpu_samples": gpu_samples,
            "trim_results": runtime.trim_results,
            "trimmer_stats": (
                asdict(runtime.trimmer.stats) if runtime.trimmer is not None else None
            ),
            "predictor": (
                {
                    "sample_count": predictor.sample_count,
                    "state_output": (
                        str(args.predictor_output)
                        if args.predictor_output is not None
                        else None
                    ),
                }
                if predictor is not None
                else None
            ),
            "online_tool_call_valid_rate": (
                runtime.valid_tool_calls / runtime.tool_call_outputs
                if runtime.tool_call_outputs
                else 0.0
            ),
        }
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--hotpot-path", type=Path, required=True)
    parser.add_argument("--hotpot-case-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--cases", type=int, default=1)
    parser.add_argument("--case-concurrency", type=int, default=1)
    parser.add_argument("--branches", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--attention-backend",
        choices=("FLASH_ATTN", "FORK_ATTN"),
        required=True,
    )
    parser.add_argument("--prompt-compaction", action="store_true")
    parser.add_argument(
        "--ttl-mode",
        choices=("disabled", "fixed", "predicted"),
        default="disabled",
    )
    parser.add_argument("--ttl-ms", type=float, default=500)
    parser.add_argument("--predicted-min-ttl-ms", type=float, default=100)
    parser.add_argument("--pressure-threshold", type=float, default=0)
    parser.add_argument("--predictor-state", type=Path)
    parser.add_argument("--predictor-output", type=Path)
    parser.add_argument("--predictor-min-samples", type=int, default=10)
    parser.add_argument("--offload-cpu-gib", type=float, default=0)
    parser.add_argument("--tool-delay-ms", type=float, default=2000)
    parser.add_argument(
        "--tool-delay-profile",
        choices=("synchronized", "fixed", "lognormal"),
        default="lognormal",
    )
    parser.add_argument("--bootstrap-chunks", type=int, default=40)
    parser.add_argument("--bootstrap-max-chars", type=int, default=100000)
    parser.add_argument("--planner-tokens", type=int, default=96)
    parser.add_argument("--tool-tokens", type=int, default=64)
    parser.add_argument("--reflect-tokens", type=int, default=128)
    parser.add_argument("--reduce-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument(
        "--num-gpu-blocks",
        type=int,
        default=0,
        help="Override the GPU KV block count; zero lets vLLM profile capacity.",
    )
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.ttl_mode == "predicted" and args.predictor_state is None:
        parser.error("--predictor-state is required for predicted TTL")
    if args.cases < 1 or args.case_concurrency < 1 or args.branches < 2:
        parser.error("cases/case-concurrency must be positive and branches >= 2")
    return args


def main() -> None:
    args = parse_args()
    payload = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "evaluation": payload["evaluation"],
                "fork_execution": payload["kv"]["fork_execution"],
                "trim_results": len(payload["trim_results"]),
                "trimmer_stats": payload["trimmer_stats"],
                "compaction": payload["metadata"]["prompt_compaction_report"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
