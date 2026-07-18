from __future__ import annotations

import argparse
import asyncio
import json
import operator
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from openai import AsyncOpenAI

from agentrix_application import PromptSection, compact_prompt_delta


class CaseState(TypedDict, total=False):
    case: dict[str, Any]
    branch_results: Annotated[list[dict[str, Any]], operator.add]


class BranchState(TypedDict):
    case_id: str
    shared_messages: list[dict[str, str]]
    branch: dict[str, Any]
    known_prompt_sections: list[dict[str, Any]]


@dataclass
class RequestMetric:
    case_id: str
    stage: str
    branch_id: int | None
    round_index: int | None
    started_ms: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    ttft_ms: float | None
    tpot_ms: float | None


class WaveBarrier:
    def __init__(self, parties: int) -> None:
        self.parties = parties
        self.arrived = 0
        self.condition = asyncio.Condition()

    async def wait(self) -> None:
        async with self.condition:
            self.arrived += 1
            if self.arrived >= self.parties:
                self.condition.notify_all()
                return
            await self.condition.wait_for(lambda: self.arrived >= self.parties)


class Runtime:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        case_count: int,
        round_parties: dict[int, int],
    ) -> None:
        self.client = client
        self.model = model
        self.started = time.perf_counter()
        self.metrics: list[RequestMetric] = []
        self.metrics_lock = asyncio.Lock()
        self.bootstrap_barrier = WaveBarrier(case_count)
        self.round_barriers = {
            index: WaveBarrier(parties) for index, parties in round_parties.items()
        }

    async def wait_for_round(self, round_index: int) -> None:
        await self.round_barriers[round_index].wait()

    async def complete(
        self,
        *,
        case_id: str,
        stage: str,
        branch_id: int | None,
        round_index: int | None,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> tuple[str, RequestMetric]:
        started_ms = (time.perf_counter() - self.started) * 1000
        started = time.perf_counter()
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        parts: list[str] = []
        first_content_at: float | None = None
        usage = None
        async for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content or ""
            if content:
                if first_content_at is None:
                    first_content_at = time.perf_counter()
                parts.append(content)
        completed = time.perf_counter()
        if usage is None:
            raise RuntimeError("streaming response omitted token usage")
        latency_ms = (completed - started) * 1000
        ttft_ms = (
            (first_content_at - started) * 1000
            if first_content_at is not None
            else None
        )
        tpot_ms = None
        if ttft_ms is not None and usage.completion_tokens > 1:
            tpot_ms = (latency_ms - ttft_ms) / (usage.completion_tokens - 1)
        metric = RequestMetric(
            case_id=case_id,
            stage=stage,
            branch_id=branch_id,
            round_index=round_index,
            started_ms=started_ms,
            latency_ms=latency_ms,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
        )
        async with self.metrics_lock:
            self.metrics.append(metric)
        return "".join(parts), metric


def build_graph(
    runtime: Runtime,
    branch_output_tokens: int,
    rounds: int,
    trajectory_mode: str,
    prompt_compaction: bool = False,
):
    async def bootstrap(state: CaseState) -> dict[str, Any]:
        case = state["case"]
        # This one-token request establishes the natural parent/cohort owner
        # without introducing backend-dependent generated text into the branch
        # prefix. All branches still use the immutable frozen parent artifact.
        await runtime.complete(
            case_id=case["case_id"],
            stage="bootstrap",
            branch_id=None,
            round_index=None,
            messages=case["shared_messages"],
            max_tokens=1,
        )
        await runtime.bootstrap_barrier.wait()
        return {}

    def fanout(state: CaseState) -> list[Send]:
        case = state["case"]
        return [
            Send(
                "branch",
                {
                    "case_id": case["case_id"],
                    "shared_messages": case["shared_messages"],
                    "branch": branch,
                    "known_prompt_sections": case.get("known_prompt_sections", []),
                },
            )
            for branch in case["branches"]
        ]

    async def branch(state: BranchState) -> dict[str, Any]:
        branch_spec = state["branch"]
        trajectory = branch_spec.get("trajectory") or [
            {"stage": "triage", "instruction": branch_spec["private_instruction"]}
        ]
        messages = list(state["shared_messages"])
        round_results = []
        compaction_totals = {
            "input_sections": 0,
            "output_sections": 0,
            "removed_duplicate_sections": 0,
            "before_chars": 0,
            "after_chars": 0,
        }
        known_sections = [
            PromptSection(**section) for section in state["known_prompt_sections"]
        ]
        for round_index, turn in enumerate(trajectory[:rounds], start=1):
            await runtime.wait_for_round(round_index)
            observation = turn.get("tool_observation")
            tool_sections = [
                PromptSection(**section) for section in turn.get("tool_sections", [])
            ]
            if tool_sections:
                if prompt_compaction:
                    compacted = compact_prompt_delta(
                        tool_sections, known_sections=known_sections
                    )
                    section_text = compacted.text
                    report = compacted.report
                    compaction_totals["input_sections"] += report.input_sections
                    compaction_totals["output_sections"] += report.output_sections
                    compaction_totals["removed_duplicate_sections"] += (
                        report.removed_duplicate_sections
                    )
                    compaction_totals["before_chars"] += report.before_chars
                    compaction_totals["after_chars"] += report.after_chars
                else:
                    section_text = "\n\n".join(
                        section.render() for section in tool_sections
                    )
                    compaction_totals["input_sections"] += len(tool_sections)
                    compaction_totals["output_sections"] += len(tool_sections)
                    compaction_totals["before_chars"] += len(section_text)
                    compaction_totals["after_chars"] += len(section_text)
                observation = (
                    f"{observation}\n\n{section_text}" if section_text else observation
                )
            if observation:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool `{turn['tool']}` returned:\n{observation}\n\n"
                            f"Next instruction:\n{turn['instruction']}"
                        ),
                    }
                )
            else:
                messages.append({"role": "user", "content": turn["instruction"]})
            text, metric = await runtime.complete(
                case_id=state["case_id"],
                stage=f"branch_round_{round_index}",
                branch_id=int(branch_spec["branch_id"]),
                round_index=round_index,
                messages=messages,
                max_tokens=branch_output_tokens,
            )
            round_results.append(
                {
                    "round_index": round_index,
                    "stage": turn["stage"],
                    "tool": turn.get("tool"),
                    "text": text,
                    "input_tokens": metric.input_tokens,
                    "output_tokens": metric.output_tokens,
                }
            )
            assistant_text = (
                text
                if trajectory_mode == "live"
                else "Prior-round analysis is retained in the parent trace."
            )
            messages.append({"role": "assistant", "content": assistant_text})
        return {
            "branch_results": [
                {
                    "case_id": state["case_id"],
                    "branch_id": int(branch_spec["branch_id"]),
                    "text": round_results[-1]["text"],
                    "input_tokens": sum(x["input_tokens"] for x in round_results),
                    "output_tokens": sum(x["output_tokens"] for x in round_results),
                    "rounds": round_results,
                    "compaction": compaction_totals,
                }
            ]
        }

    graph = StateGraph(CaseState)
    graph.add_node("bootstrap", bootstrap)
    graph.add_node("branch", branch)
    graph.add_edge(START, "bootstrap")
    graph.add_conditional_edges("bootstrap", fanout, ["branch"])
    graph.add_edge("branch", END)
    return graph.compile()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    case_ids = [str(case["case_id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be unique within the case file")
    return cases


def select_cases(
    cases: list[dict[str, Any]],
    *,
    offset: int,
    limit: int,
    required_count: int,
) -> list[dict[str, Any]]:
    if offset < 0 or limit < 0 or required_count < 0:
        raise ValueError("case selection values must be non-negative")
    selected = cases[offset:]
    if limit:
        selected = selected[:limit]
    if required_count and len(selected) != required_count:
        raise ValueError(f"selected {len(selected)} cases, expected {required_count}")
    return selected


def summarize_repository_metrics(
    branches: list[RequestMetric], case_repositories: dict[str, str]
) -> dict[str, dict[str, Any]]:
    summaries = {}
    for repository in sorted(set(case_repositories.values())):
        current = [
            metric
            for metric in branches
            if case_repositories[metric.case_id] == repository
        ]
        started_ms = min((metric.started_ms for metric in current), default=0.0)
        ended_ms = max(
            (metric.started_ms + metric.latency_ms for metric in current),
            default=started_ms,
        )
        wall_ms = ended_ms - started_ms
        output_tokens = sum(metric.output_tokens for metric in current)
        ttfts = [metric.ttft_ms for metric in current if metric.ttft_ms is not None]
        tpots = [metric.tpot_ms for metric in current if metric.tpot_ms is not None]
        summaries[repository] = {
            "request_count": len(current),
            "wall_time_ms": wall_ms,
            "input_tokens": sum(metric.input_tokens for metric in current),
            "output_tokens": output_tokens,
            "output_tokens_per_s": (output_tokens * 1000 / wall_ms if wall_ms else 0.0),
            "ttft_p50_ms": percentile(ttfts, 0.50),
            "ttft_p95_ms": percentile(ttfts, 0.95),
            "tpot_p50_ms": percentile(tpots, 0.50),
        }
    return summaries


async def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = select_cases(
        load_cases(args.cases),
        offset=args.case_offset,
        limit=args.case_limit,
        required_count=args.required_case_count,
    )
    client = AsyncOpenAI(api_key="local", base_url=args.base_url, timeout=900)
    total_branches = sum(len(case["branches"]) for case in cases)
    round_parties = {
        round_index: sum(
            min(len(branch.get("trajectory", [])) or 1, args.rounds) >= round_index
            for case in cases
            for branch in case["branches"]
        )
        for round_index in range(1, args.rounds + 1)
    }
    round_parties = {key: value for key, value in round_parties.items() if value}
    runtime = Runtime(client, args.model, len(cases), round_parties)
    graph = build_graph(
        runtime,
        args.branch_output_tokens,
        args.rounds,
        args.trajectory_mode,
        args.prompt_compaction,
    )
    started = time.perf_counter()
    states = await asyncio.gather(
        *(graph.ainvoke({"case": case, "branch_results": []}) for case in cases)
    )
    wall_ms = (time.perf_counter() - started) * 1000
    await client.close()
    metrics = sorted(runtime.metrics, key=lambda item: item.started_ms)
    branches = [
        metric for metric in metrics if metric.stage.startswith("branch_round_")
    ]
    if branches:
        branch_start = min(metric.started_ms for metric in branches)
        branch_end = max(metric.started_ms + metric.latency_ms for metric in branches)
        branch_wall_ms = branch_end - branch_start
    else:
        branch_wall_ms = 0.0
    output_tokens = sum(metric.output_tokens for metric in branches)
    ttfts = [metric.ttft_ms for metric in branches if metric.ttft_ms is not None]
    tpots = [metric.tpot_ms for metric in branches if metric.tpot_ms is not None]
    repositories = sorted({case.get("repo", "unknown") for case in cases})
    case_repositories = {
        str(case["case_id"]): str(case.get("repo", "unknown")) for case in cases
    }
    round_metrics = {}
    for round_index in range(1, args.rounds + 1):
        current = [x for x in branches if x.round_index == round_index]
        started_ms = min((x.started_ms for x in current), default=0.0)
        ended_ms = max(
            (x.started_ms + x.latency_ms for x in current), default=started_ms
        )
        current_ttft = [x.ttft_ms for x in current if x.ttft_ms is not None]
        current_tpot = [x.tpot_ms for x in current if x.tpot_ms is not None]
        round_metrics[str(round_index)] = {
            "request_count": len(current),
            "wall_time_ms": ended_ms - started_ms,
            "input_tokens": sum(x.input_tokens for x in current),
            "output_tokens": sum(x.output_tokens for x in current),
            "ttft_p50_ms": percentile(current_ttft, 0.50),
            "tpot_p50_ms": percentile(current_tpot, 0.50),
        }
    return {
        "schema_version": 2,
        "workload": "repository_agentrix_long_prefix_fanout",
        "experiment_variant": args.experiment_variant,
        "attention_backend": args.attention_backend,
        "dp_policy": args.dp_policy,
        "case_file": str(args.cases),
        "case_offset": args.case_offset,
        "trajectory_mode": args.trajectory_mode,
        "prompt_compaction": args.prompt_compaction,
        "round_count": args.rounds,
        "round_request_counts": round_parties,
        "repositories": repositories,
        "case_count": len(cases),
        "branch_count": total_branches,
        "branch_request_count": len(branches),
        "shared_parent_tokens_harness": [
            case["shared_parent_tokens"] for case in cases
        ],
        "wall_time_ms": wall_ms,
        "branch_wall_time_ms": branch_wall_ms,
        "branch_input_tokens": sum(metric.input_tokens for metric in branches),
        "branch_output_tokens": output_tokens,
        "branch_output_tokens_per_s": (
            output_tokens * 1000 / branch_wall_ms if branch_wall_ms else 0.0
        ),
        "branch_ttft_ms": {
            "p50": percentile(ttfts, 0.50),
            "p95": percentile(ttfts, 0.95),
            "mean": statistics.fmean(ttfts) if ttfts else None,
        },
        "branch_tpot_ms": {
            "p50": percentile(tpots, 0.50),
            "p95": percentile(tpots, 0.95),
            "mean": statistics.fmean(tpots) if tpots else None,
        },
        "round_metrics": round_metrics,
        "repository_metrics": summarize_repository_metrics(branches, case_repositories),
        "requests": [asdict(metric) for metric in metrics],
        "branch_results": [
            result for state in states for result in state.get("branch_results", [])
        ],
        "compaction": {
            key: sum(
                result.get("compaction", {}).get(key, 0)
                for state in states
                for result in state.get("branch_results", [])
            )
            for key in (
                "input_sections",
                "output_sections",
                "removed_duplicate_sections",
                "before_chars",
                "after_chars",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run repository Agentrix LangGraph cases"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:9000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--experiment-variant", default="unspecified")
    parser.add_argument("--attention-backend", default="unspecified")
    parser.add_argument("--dp-policy", default="unspecified")
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--required-case-count", type=int, default=0)
    parser.add_argument("--branch-output-tokens", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--trajectory-mode", choices=("live", "replay"), default="live")
    parser.add_argument("--prompt-compaction", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in {"requests", "branch_results"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
