#!/usr/bin/env python3
"""Run a small live LangGraph agent while tracing tool-KV lifetime.

The task fixtures are deterministic, but the model chooses each search/read
action.  Tools execute against real files and the vLLM request remains
resumable while the LangGraph tool node is running.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypedDict

from agentrix_application.tool_kv_trimmer import ToolKVTrimmer, ToolKVTrimmerConfig
from agentrix_application.tool_ttl_predictor import (
    OnlineHorizonTTLPredictor,
    ToolTTLContext,
)
from benchmark_full_stack_agent import (
    FullStackLogger,
    StreamingAgentSession,
    choose_action,
    continuation_ids,
    gpu_memory_sampler,
    normalize_token_ids,
)
from full_stack_agent import BM25Index, Passage, render_search_results
from langgraph.graph import END, START, StateGraph
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

from vllm import SamplingParams

CASE_TEXT = {
    "incident_report.txt": """
At 09:42 the checkout API began returning intermittent 503 responses.  The
application deployment was unchanged.  Database latency and error rates
remained normal.  Worker logs repeatedly reported `catalog lookup timeout`.
The affected workers were all in availability zone west-b.
""".strip(),
    "cache_metrics.txt": """
catalog-cache west-a: hit_rate=0.94 p99_ms=18 connections=41
catalog-cache west-b: hit_rate=0.07 p99_ms=2410 connections=200
catalog-cache west-c: hit_rate=0.93 p99_ms=21 connections=39
At 09:39 west-b reached its configured connection limit of 200.
""".strip(),
    "change_log.txt": """
09:15 routine certificate rotation completed.
09:36 a configuration rollout changed catalog-cache pool_size from 240 to 200
for west-b only.  Rollback command: deployctl rollback catalog-cache west-b.
10:05 no other infrastructure changes were recorded.
""".strip(),
}


class AgentState(TypedDict, total=False):
    round: int
    generated: str
    tool_result: str
    next_instruction: str
    done: bool


def event(events: list[dict[str, Any]], started: float, kind: str, **values: Any) -> None:
    events.append(
        {
            "time_ms": (time.perf_counter() - started) * 1000,
            "kind": kind,
            **values,
        }
    )


def chat_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    return normalize_token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


async def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    workspace = args.workspace
    workspace.mkdir(parents=True, exist_ok=True)
    for name, content in CASE_TEXT.items():
        (workspace / name).write_text(content + "\n", encoding="utf-8")

    passages = [
        Passage(
            passage_id=path.name.upper(),
            text=path.read_text(encoding="utf-8"),
        )
        for path in sorted(workspace.iterdir())
        if path.is_file()
    ]
    passages_by_id = {item.passage_id: item for item in passages}
    index = BM25Index(passages)

    FullStackLogger.instances.clear()
    engine_args = AsyncEngineArgs(
        model=str(args.model),
        dtype="bfloat16",
        attention_backend=args.attention_backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_gpu_blocks_override=args.num_gpu_blocks,
        max_model_len=args.max_model_len,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        enforce_eager=True,
        disable_log_stats=False,
        enable_log_requests=False,
        generation_config="vllm",
    )
    event(events, started, "engine_start")
    engine = AsyncLLM.from_engine_args(engine_args, stat_loggers=[FullStackLogger])
    tokenizer = engine.get_tokenizer()
    params = SamplingParams(
        max_tokens=args.action_tokens,
        temperature=0,
        seed=args.seed,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
    )
    session = StreamingAgentSession(engine, "langgraph-s2-session", params)

    trim_results: list[dict[str, Any]] = []

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
        event(events, started, "trim_result", **result)
        return result

    predictor = (
        OnlineHorizonTTLPredictor.load(args.predictor_state)
        if args.mode == "predicted"
        else None
    )
    trimmer = (
        ToolKVTrimmer(
            pressure_reader=pressure_reader,
            trim_request=trim_request,
            config=ToolKVTrimmerConfig(
                enabled=True,
                grace_ms=args.ttl_ms,
                pressure_threshold=0.0,
                post_trim_recheck_ms=0,
                use_predicted_ttl=args.mode == "predicted",
            ),
            ttl_predictor=predictor,
        )
        if args.mode != "baseline"
        else None
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
                    "time_ms": (time.perf_counter() - started) * 1000,
                    "usage": max((sample.usage for sample in latest), default=0.0),
                    "running": sum(sample.running for sample in latest),
                    "waiting": sum(sample.waiting for sample in latest),
                    "tool_waiting": sum(sample.tool_waiting for sample in latest),
                }
            )
            await asyncio.sleep(0.05)

    kv_sampler = asyncio.create_task(sample_kv_periodically())
    last_matches: list[tuple[Passage, float]] = []
    read_ids: set[str] = set()

    async def agent_node(state: AgentState) -> AgentState:
        round_index = state.get("round", 0)
        event(events, started, "agent_start", round=round_index)
        if round_index == 0:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Investigate the incident using tools. Reply with exactly one "
                        "line. Allowed forms: SEARCH: <query>, READ: <filename>, or "
                        "FINAL: <root cause and remediation>. Read evidence before final."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Find the root cause of the checkout 503 incident and give the "
                        "specific remediation. Start with SEARCH."
                    ),
                },
            ]
            turn = await session.generate_turn(chat_ids(tokenizer, messages))
        else:
            turn = await session.generate_turn(
                continuation_ids(
                    tokenizer,
                    f"TOOL RESULT:\n{state['tool_result']}\n\n"
                    f"{state['next_instruction']}",
                )
            )
        generated = turn.text.strip()
        is_final = generated.startswith("FINAL:")
        event(
            events,
            started,
            "agent_end",
            round=round_index,
            output=generated,
            ttft_ms=turn.first_token_seconds * 1000,
            latency_ms=turn.latency_seconds * 1000,
            cached_tokens=turn.cached_tokens,
        )
        return {"generated": generated, "done": is_final}

    async def tool_node(state: AgentState) -> AgentState:
        nonlocal last_matches
        round_index = state.get("round", 0)
        expected = "search" if not last_matches else "read"
        action, valid = choose_action(
            state["generated"],
            expected=expected,
            question="checkout 503 root cause remediation",
            passages_by_id=passages_by_id,
            last_matches=last_matches,
        )
        event(
            events,
            started,
            "tool_start",
            round=round_index,
            tool=action.kind,
            argument=action.value,
            model_action_valid=valid,
        )
        scheduled = False
        ttl_context = ToolTTLContext(
            tool_family=action.kind,
            argument_bytes=len(action.value.encode()),
            kv_tokens=args.max_model_len,
            pressure=await pressure_reader(),
            active_tool_sessions=1,
            timeout_ms=int(args.tool_delay_ms * 2),
        )
        if predictor is not None:
            prediction = predictor.predict(ttl_context)
            event(
                events,
                started,
                "ttl_prediction",
                round=round_index,
                tool=action.kind,
                ttl_ms=prediction.ttl_ms,
                trained_samples=prediction.trained_samples,
                used_fallback=prediction.used_fallback,
                survival_probabilities=list(prediction.survival_probabilities),
            )
        if trimmer is not None:
            scheduled = trimmer.tool_started(
                "langgraph-s2",
                session.request_id,
                ttl_context=ttl_context,
            )
        await asyncio.sleep(args.tool_delay_ms / 1000)
        if action.kind == "search":
            last_matches = index.search(action.value, top_k=3)
            result = render_search_results(last_matches)
            instruction = "Reply exactly READ: <one returned filename>."
        else:
            passage = passages_by_id[action.value]
            read_ids.add(passage.passage_id)
            result = passage.text
            instruction = (
                "If evidence is sufficient reply FINAL: <root cause and remediation>. "
                "Otherwise reply SEARCH: <query>."
            )
        if trimmer is not None:
            await trimmer.tool_finished(
                "langgraph-s2",
                session.request_id,
                duration_ms=args.tool_delay_ms,
            )
        event(
            events,
            started,
            "tool_end",
            round=round_index,
            tool=action.kind,
            result_chars=len(result),
            trim_scheduled=scheduled,
        )
        return {
            "round": round_index + 1,
            "tool_result": result,
            "next_instruction": instruction,
        }

    def route(state: AgentState) -> str:
        if state.get("done") or state.get("round", 0) >= args.max_rounds:
            return END
        return "tool"

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tool", tool_node)
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", route, {"tool": "tool", END: END})
    graph_builder.add_edge("tool", "agent")
    graph = graph_builder.compile()

    final_state: AgentState = {}
    try:
        final_state = await graph.ainvoke({"round": 0})
    finally:
        if trimmer is not None:
            await trimmer.close()
        await session.close()
        stop_sampler.set()
        await sampler
        await kv_sampler
        engine.shutdown()

    origin = started
    kv_samples = [
        {
            **asdict(sample),
            "time_ms": (sample.timestamp - origin) * 1000,
        }
        for logger in FullStackLogger.instances
        for sample in logger.samples
    ]
    return {
        "schema_version": 1,
        "case": "S2_live_slow_incident_investigation",
        "mode": args.mode,
        "model": str(args.model),
        "request_id": session.request_id,
        "tool_delay_ms": args.tool_delay_ms,
        "ttl_ms": args.ttl_ms if trimmer else None,
        "predictor_state": (
            str(args.predictor_state) if args.predictor_state is not None else None
        ),
        "completed": bool(final_state.get("done")),
        "final_output": final_state.get("generated", ""),
        "read_files": sorted(read_ids),
        "events": events,
        "kv_samples": periodic_kv_samples,
        "scheduler_kv_samples": kv_samples,
        "gpu_samples": gpu_samples,
        "trim_results": trim_results,
        "trimmer_stats": asdict(trimmer.stats) if trimmer else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("baseline", "ttl", "predicted"), required=True
    )
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--tool-delay-ms", type=float, default=2000)
    parser.add_argument("--ttl-ms", type=float, default=500)
    parser.add_argument("--predictor-state", type=Path)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--action-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--num-gpu-blocks", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "predicted" and args.predictor_state is None:
        parser.error("--predictor-state is required in predicted mode")
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
                "case": payload["case"],
                "mode": payload["mode"],
                "completed": payload["completed"],
                "events": len(payload["events"]),
                "kv_samples": len(payload["kv_samples"]),
                "trim_results": len(payload["trim_results"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
