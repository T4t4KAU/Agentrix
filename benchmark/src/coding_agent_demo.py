from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import subprocess
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import tiktoken
from openai import AsyncOpenAI

from agentrix_application import PromptSection, compact_prompt_delta


EventSink = Callable[["DemoEvent"], Awaitable[None]]
TOKENIZER = tiktoken.get_encoding("o200k_base")


@dataclass(frozen=True)
class SideConfig:
    side: str
    title: str
    base_url: str
    model: str
    gpu_ids: tuple[int, ...]
    prompt_compaction: bool
    routing_label: str
    color: str
    gpu_metrics_url: str | None = None


@dataclass(frozen=True)
class DemoEvent:
    side: str
    kind: str
    request_id: str | None = None
    case_id: str | None = None
    agent_id: int | None = None
    text: str = ""
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStream:
    request_id: str
    case_id: str
    agent_id: int
    status: str = "queued"
    text: str = ""
    estimated_tokens: int = 0
    dp_rank: int | None = None


@dataclass
class SideState:
    config: SideConfig
    started_at: float = field(default_factory=time.monotonic)
    ended_at: float | None = None
    completed_tokens: int = 0
    completed_requests: int = 0
    expected_requests: int = 0
    failed_requests: int = 0
    run_completed: bool = False
    active_requests: int = 0
    queued_requests: int = 0
    prompt_tokens: int = 0
    hbm_used_mib: list[float] = field(default_factory=list)
    hbm_total_mib: list[float] = field(default_factory=list)
    rank_loads: list[tuple[int, int]] = field(default_factory=list)
    kv_usage: list[float] = field(default_factory=list)
    waiting_sequences: int = 0
    queue_time_sum_s: float = 0.0
    queue_time_count: int = 0
    queue_time_baseline_sum_s: float = 0.0
    queue_time_baseline_count: int = 0
    computed_prefill_tokens: int = 0
    computed_prefill_baseline: int = 0
    streams: dict[str, AgentStream] = field(default_factory=dict)
    display_order: deque[str] = field(default_factory=lambda: deque(maxlen=24))
    recent_ttft_ms: deque[float] = field(default_factory=lambda: deque(maxlen=64))

    @property
    def live_estimated_tokens(self) -> int:
        return sum(
            stream.estimated_tokens
            for stream in self.streams.values()
            if stream.status in {"prefill", "generating"}
        )

    @property
    def displayed_tokens(self) -> int:
        return self.completed_tokens + self.live_estimated_tokens

    @property
    def prefill_requests(self) -> int:
        return sum(stream.status == "prefill" for stream in self.streams.values())

    @property
    def generating_requests(self) -> int:
        return sum(stream.status == "generating" for stream in self.streams.values())

    @property
    def output_tokens_per_s(self) -> float:
        elapsed = max((self.ended_at or time.monotonic()) - self.started_at, 1e-6)
        return self.displayed_tokens / elapsed

    @property
    def average_queue_time_ms(self) -> float | None:
        count = self.queue_time_count - self.queue_time_baseline_count
        if count <= 0:
            return None
        total = self.queue_time_sum_s - self.queue_time_baseline_sum_s
        return 1000 * total / count

    @property
    def run_computed_prefill_tokens(self) -> int:
        return max(0, self.computed_prefill_tokens - self.computed_prefill_baseline)

    @property
    def ttft_p50_ms(self) -> float | None:
        if not self.recent_ttft_ms:
            return None
        return statistics.median(self.recent_ttft_ms)

    def apply(self, event: DemoEvent) -> None:
        if event.kind == "side_completed":
            self.ended_at = time.monotonic()
            self.run_completed = True
            return
        if event.kind == "reset_clock":
            self.started_at = time.monotonic()
            self.ended_at = None
            self.run_completed = False
            self.queue_time_baseline_sum_s = self.queue_time_sum_s
            self.queue_time_baseline_count = self.queue_time_count
            self.computed_prefill_baseline = self.computed_prefill_tokens
            return
        if event.kind == "metrics":
            self.hbm_used_mib = list(event.values.get("hbm_used_mib", []))
            self.hbm_total_mib = list(event.values.get("hbm_total_mib", []))
            self.rank_loads = list(event.values.get("rank_loads", []))
            self.kv_usage = list(event.values.get("kv_usage", []))
            self.waiting_sequences = int(event.values.get("waiting_sequences", 0))
            self.queue_time_sum_s = float(event.values.get("queue_time_sum_s", 0))
            self.queue_time_count = int(event.values.get("queue_time_count", 0))
            self.computed_prefill_tokens = int(
                event.values.get("computed_prefill_tokens", 0)
            )
            return
        if event.kind == "route" and event.request_id is not None:
            stream = self.streams.get(event.request_id)
            if stream is not None:
                stream.dp_rank = int(event.values["dp_rank"])
            return
        if event.request_id is None:
            return
        stream = self.streams.get(event.request_id)
        if event.kind == "queued":
            stream = AgentStream(
                request_id=event.request_id,
                case_id=event.case_id or "unknown",
                agent_id=event.agent_id or 0,
            )
            self.streams[event.request_id] = stream
            self.display_order.append(event.request_id)
            self.queued_requests += 1
        elif stream is None:
            return
        elif event.kind == "started":
            stream.status = "prefill"
            self.queued_requests = max(0, self.queued_requests - 1)
            self.active_requests += 1
        elif event.kind == "token":
            stream.status = "generating"
            stream.text += event.text
            stream.estimated_tokens = len(TOKENIZER.encode(stream.text))
            try:
                self.display_order.remove(event.request_id)
            except ValueError:
                pass
            self.display_order.append(event.request_id)
        elif event.kind == "completed":
            stream.status = "completed"
            stream.estimated_tokens = 0
            self.active_requests = max(0, self.active_requests - 1)
            self.completed_requests += 1
            self.completed_tokens += int(event.values.get("output_tokens", 0))
            self.prompt_tokens += int(event.values.get("prompt_tokens", 0))
            ttft = event.values.get("ttft_ms")
            if isinstance(ttft, (int, float)):
                self.recent_ttft_ms.append(float(ttft))
        elif event.kind == "failed":
            stream.status = "error"
            stream.text += f"\n{event.text}"
            self.active_requests = max(0, self.active_requests - 1)
            self.failed_requests += 1


def load_cases(path: Path, *, offset: int, limit: int) -> list[dict[str, Any]]:
    cases = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = cases[offset : offset + limit]
    if len(selected) != limit:
        raise ValueError(f"selected {len(selected)} cases, expected {limit}")
    return selected


def expected_request_count(
    cases: list[dict[str, Any]], *, rounds: int, mock: bool = False
) -> int:
    if mock:
        return sum(len(case["branches"]) for case in cases)
    return sum(
        min(len(branch.get("trajectory", [])) or 1, rounds)
        for case in cases
        for branch in case["branches"]
    )


async def reset_prefix_caches(
    configs: list[SideConfig] | tuple[SideConfig, ...],
) -> None:
    async def reset(config: SideConfig) -> None:
        endpoint = f"{config.base_url.removesuffix('/v1')}/reset_prefix_cache"

        def request() -> None:
            with urllib.request.urlopen(
                urllib.request.Request(endpoint, method="POST"), timeout=30
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"{config.title} cache reset returned HTTP {response.status}"
                    )

        await asyncio.to_thread(request)

    await asyncio.gather(*(reset(config) for config in configs))


def branch_messages(
    case: dict[str, Any],
    branch: dict[str, Any],
    *,
    rounds: int,
    prompt_compaction: bool,
) -> list[list[dict[str, str]]]:
    messages = list(case["shared_messages"])
    known_sections = [
        PromptSection(**section) for section in case.get("known_prompt_sections", [])
    ]
    snapshots = []
    trajectory = branch.get("trajectory") or [
        {"stage": "triage", "instruction": branch["private_instruction"]}
    ]
    for turn in trajectory[:rounds]:
        observation = turn.get("tool_observation")
        sections = [PromptSection(**item) for item in turn.get("tool_sections", [])]
        if sections:
            if prompt_compaction:
                section_text = compact_prompt_delta(
                    sections, known_sections=known_sections
                ).text
            else:
                section_text = "\n\n".join(section.render() for section in sections)
            observation = (
                f"{observation}\n\n{section_text}" if observation else section_text
            )
        instruction = turn["instruction"]
        if observation:
            content = (
                f"Tool `{turn['tool']}` returned:\n{observation}\n\n"
                f"Next instruction:\n{instruction}"
            )
        else:
            content = instruction
        messages.append({"role": "user", "content": content})
        snapshots.append(list(messages))
        messages.append(
            {
                "role": "assistant",
                "content": "Prior-round analysis is retained in the parent trace.",
            }
        )
    return snapshots


class WorkloadSide:
    def __init__(
        self,
        config: SideConfig,
        cases: list[dict[str, Any]],
        *,
        rounds: int,
        max_tokens: int,
        sink: EventSink,
    ) -> None:
        self.config = config
        self.cases = cases
        self.rounds = rounds
        self.max_tokens = max_tokens
        self.sink = sink
        self.client = AsyncOpenAI(
            api_key="local", base_url=config.base_url, timeout=900
        )

    async def close(self) -> None:
        await self.client.close()

    async def run(
        self,
        start: asyncio.Event,
        ready: asyncio.Event | None = None,
        fanout_start: asyncio.Event | None = None,
    ) -> None:
        await start.wait()
        # Match the benchmark methodology: establish one natural owner for each
        # case cohort before releasing the fan-out wave.
        await asyncio.gather(*(self._bootstrap(case) for case in self.cases))
        if ready is not None:
            ready.set()
        if fanout_start is not None:
            await fanout_start.wait()
        for round_index in range(self.rounds):
            tasks = []
            for case in self.cases:
                for branch in case["branches"]:
                    snapshots = branch_messages(
                        case,
                        branch,
                        rounds=self.rounds,
                        prompt_compaction=self.config.prompt_compaction,
                    )
                    if round_index >= len(snapshots):
                        continue
                    tasks.append(
                        self._complete(
                            case_id=str(case["case_id"]),
                            agent_id=int(branch["branch_id"]),
                            round_index=round_index + 1,
                            messages=snapshots[round_index],
                        )
                    )
            await asyncio.gather(*tasks)
        await self.sink(DemoEvent(side=self.config.side, kind="side_completed"))

    async def _bootstrap(self, case: dict[str, Any]) -> None:
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=case["shared_messages"],
            temperature=0,
            max_tokens=1,
            extra_headers={"X-Request-Id": f"bootstrap:{case['case_id']}"},
        )
        if not response.choices:
            raise RuntimeError(f"bootstrap failed for {case['case_id']}")

    async def _complete(
        self,
        *,
        case_id: str,
        agent_id: int,
        round_index: int,
        messages: list[dict[str, str]],
    ) -> None:
        request_id = f"{case_id}:{agent_id}:r{round_index}"
        common = {
            "side": self.config.side,
            "request_id": request_id,
            "case_id": case_id,
            "agent_id": agent_id,
        }
        await self.sink(DemoEvent(kind="queued", **common))
        await self.sink(DemoEvent(kind="started", **common))
        started = time.perf_counter()
        first_token_at: float | None = None
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0,
                max_tokens=self.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
                extra_headers={"X-Request-Id": request_id},
            )
            usage = None
            async for chunk in response:
                if chunk.usage is not None:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                text = chunk.choices[0].delta.content or ""
                if not text:
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                await self.sink(DemoEvent(kind="token", text=text, **common))
            if usage is None:
                raise RuntimeError("streaming response omitted token usage")
            ttft_ms = (
                (first_token_at - started) * 1000
                if first_token_at is not None
                else None
            )
            await self.sink(
                DemoEvent(
                    kind="completed",
                    values={
                        "prompt_tokens": usage.prompt_tokens,
                        "output_tokens": usage.completion_tokens,
                        "ttft_ms": ttft_ms,
                    },
                    **common,
                )
            )
        except Exception as error:
            await self.sink(DemoEvent(kind="failed", text=str(error), **common))


PROMETHEUS_LINE = re.compile(
    r"^(?P<name>[\w:]+)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+)$"
)


def parse_engine_metrics(text: str) -> dict[str, list[tuple[int, float]]]:
    wanted = {
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:gpu_cache_usage_perc",
        "vllm:request_queue_time_seconds_sum",
        "vllm:request_queue_time_seconds_count",
        "vllm:prompt_tokens_by_source",
        "vllm:prompt_tokens_by_source_total",
    }
    result: dict[str, list[tuple[int, float]]] = {}
    for line in text.splitlines():
        match = PROMETHEUS_LINE.match(line.strip())
        if not match or match.group("name") not in wanted:
            continue
        if match.group("name").startswith("vllm:prompt_tokens_by_source"):
            source_match = re.search(r'source="([^"]+)"', match.group("labels") or "")
            if source_match is None or source_match.group(1) != "local_compute":
                continue
        engine_match = re.search(r'engine="(\d+)"', match.group("labels") or "")
        engine = (
            int(engine_match.group(1))
            if engine_match
            else len(result.get(match.group("name"), []))
        )
        result.setdefault(match.group("name"), []).append(
            (engine, float(match.group("value")))
        )
    return result


def collect_metrics(config: SideConfig) -> dict[str, Any]:
    if config.gpu_metrics_url:
        with urllib.request.urlopen(config.gpu_metrics_url, timeout=2) as response:
            gpu_payload = json.loads(response.read().decode("utf-8"))
        gpu_rows = {
            int(row["index"]): (
                float(row["memory_used_mib"]),
                float(row["memory_total_mib"]),
            )
            for row in gpu_payload["gpus"]
        }
    else:
        command = (
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        )
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=5
        )
        gpu_rows = {}
        for line in completed.stdout.splitlines():
            index, used, total = (value.strip() for value in line.split(","))
            gpu_rows[int(index)] = (float(used), float(total))
    with urllib.request.urlopen(
        config.base_url.removesuffix("/v1") + "/metrics", timeout=2
    ) as response:
        metrics = parse_engine_metrics(response.read().decode("utf-8", "replace"))
    running = dict(metrics.get("vllm:num_requests_running", []))
    waiting = dict(metrics.get("vllm:num_requests_waiting", []))
    kv = metrics.get("vllm:kv_cache_usage_perc") or metrics.get(
        "vllm:gpu_cache_usage_perc", []
    )
    queue_sum = sum(
        value for _, value in metrics.get("vllm:request_queue_time_seconds_sum", [])
    )
    queue_count = sum(
        value for _, value in metrics.get("vllm:request_queue_time_seconds_count", [])
    )
    computed_prefill = sum(
        value
        for _, value in (
            metrics.get("vllm:prompt_tokens_by_source_total")
            or metrics.get("vllm:prompt_tokens_by_source", [])
        )
    )
    return {
        "hbm_used_mib": [gpu_rows[index][0] for index in config.gpu_ids],
        "hbm_total_mib": [gpu_rows[index][1] for index in config.gpu_ids],
        "rank_loads": [
            (int(running.get(index, 0)), int(waiting.get(index, 0)))
            for index in range(max((*running, *waiting), default=-1) + 1)
        ],
        "kv_usage": [value for _, value in sorted(kv)],
        "waiting_sequences": sum(int(value) for value in waiting.values()),
        "queue_time_sum_s": queue_sum,
        "queue_time_count": int(queue_count),
        "computed_prefill_tokens": int(computed_prefill),
    }


async def monitor_side(
    config: SideConfig, sink: EventSink, stop: asyncio.Event, interval: float = 0.5
) -> None:
    while not stop.is_set():
        try:
            values = await asyncio.to_thread(collect_metrics, config)
            await sink(DemoEvent(side=config.side, kind="metrics", values=values))
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            pass


MOCK_TEXT = 3 * (
    "Inspecting the repository snapshot and tracing the relevant call path. "
    "The implementation delegates state ownership to the wrapper before the "
    "test exercises the public behavior. Evidence points to the focused symbol "
    "and its regression coverage. "
)


async def run_mock_side(
    config: SideConfig,
    cases: list[dict[str, Any]],
    *,
    sink: EventSink,
    start: asyncio.Event,
    delay: float,
) -> None:
    await start.wait()
    tasks = []
    for case in cases:
        for branch in case["branches"]:
            tasks.append(
                _run_mock_request(
                    config,
                    str(case["case_id"]),
                    int(branch["branch_id"]),
                    sink,
                    delay,
                )
            )
    await asyncio.gather(*tasks)
    await sink(DemoEvent(side=config.side, kind="side_completed"))


async def _run_mock_request(
    config: SideConfig,
    case_id: str,
    agent_id: int,
    sink: EventSink,
    delay: float,
) -> None:
    request_id = f"{case_id}:{agent_id}:r1"
    common = {
        "side": config.side,
        "request_id": request_id,
        "case_id": case_id,
        "agent_id": agent_id,
    }
    await sink(DemoEvent(kind="queued", **common))
    await asyncio.sleep(delay * (agent_id % 3))
    await sink(DemoEvent(kind="started", **common))
    started = time.perf_counter()
    for word in MOCK_TEXT.split(" "):
        await asyncio.sleep(delay)
        await sink(DemoEvent(kind="token", text=word + " ", **common))
    await sink(
        DemoEvent(
            kind="completed",
            values={
                "prompt_tokens": 30_000,
                "output_tokens": len(TOKENIZER.encode(MOCK_TEXT)),
                "ttft_ms": delay * 1000,
                "latency_ms": (time.perf_counter() - started) * 1000,
            },
            **common,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Agentrix split-screen demo")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--case-count", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--left-base-url", default="http://127.0.0.1:9000/v1")
    parser.add_argument("--right-base-url", default="http://127.0.0.1:9001/v1")
    parser.add_argument("--left-model", default="agentrix-demo-model")
    parser.add_argument("--right-model", default="baseline-demo-model")
    parser.add_argument("--left-gpus", default="0,1,2,3")
    parser.add_argument("--right-gpus", default="4,5,6,7")
    parser.add_argument("--left-gpu-metrics-url")
    parser.add_argument("--right-gpu-metrics-url")
    parser.add_argument("--mock", action="store_true")
    return parser


def parse_gpu_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(item) for item in value.split(",") if item.strip())
    if not ids:
        raise ValueError("at least one GPU ID is required")
    return ids


def side_state_payload(state: SideState, *, stream_limit: int = 4) -> dict[str, Any]:
    selected: list[AgentStream] = []
    selected_ids: set[str] = set()
    for request_id in reversed(state.display_order):
        if request_id in selected_ids:
            continue
        stream = state.streams.get(request_id)
        if stream is not None and stream.status in {"generating", "prefill"}:
            selected.append(stream)
            selected_ids.add(request_id)
        if len(selected) == stream_limit:
            break
    if len(selected) < stream_limit:
        for request_id in reversed(state.display_order):
            if request_id in selected_ids:
                continue
            stream = state.streams.get(request_id)
            if stream is not None:
                selected.append(stream)
                selected_ids.add(request_id)
            if len(selected) == stream_limit:
                break
    return {
        "side": state.config.side,
        "title": state.config.title,
        "color": state.config.color,
        "routing_label": state.config.routing_label,
        "prompt_compaction": state.config.prompt_compaction,
        "tokens": state.displayed_tokens,
        "tokens_estimated": state.active_requests > 0,
        "completed_requests": state.completed_requests,
        "expected_requests": state.expected_requests,
        "failed_requests": state.failed_requests,
        "run_completed": state.run_completed,
        "active_requests": state.active_requests,
        "prefill_requests": state.prefill_requests,
        "generating_requests": state.generating_requests,
        "queued_requests": state.queued_requests,
        "prompt_tokens": state.prompt_tokens,
        "output_tokens_per_s": state.output_tokens_per_s,
        "ttft_p50_ms": state.ttft_p50_ms,
        "hbm_used_mib": state.hbm_used_mib,
        "hbm_total_mib": state.hbm_total_mib,
        "kv_usage": state.kv_usage,
        "rank_loads": state.rank_loads,
        "waiting_sequences": state.waiting_sequences,
        "average_queue_time_ms": state.average_queue_time_ms,
        "computed_prefill_tokens": state.run_computed_prefill_tokens,
        "streams": [
            {
                "request_id": stream.request_id,
                "case_id": stream.case_id,
                "agent_id": stream.agent_id,
                "status": stream.status,
                "text": stream.text[-4000:],
                "estimated_tokens": stream.estimated_tokens,
            }
            for stream in reversed(selected)
        ],
        "routes": [
            {
                "request_id": stream.request_id,
                "prefix_group": stream.case_id,
                "agent_id": stream.agent_id,
                "dp_rank": stream.dp_rank,
                "status": stream.status,
            }
            for stream in state.streams.values()
            if stream.dp_rank is not None
        ],
    }


async def run_live_pair(
    configs: tuple[SideConfig, SideConfig],
    cases: list[dict[str, Any]],
    *,
    rounds: int,
    max_tokens: int,
    sink: EventSink,
    stop: asyncio.Event,
) -> None:
    sides = [
        WorkloadSide(
            config,
            cases,
            rounds=rounds,
            max_tokens=max_tokens,
            sink=sink,
        )
        for config in configs
    ]
    start = asyncio.Event()
    ready = [asyncio.Event(), asyncio.Event()]
    fanout_start = asyncio.Event()
    runners = [
        asyncio.create_task(side.run(start, side_ready, fanout_start))
        for side, side_ready in zip(sides, ready, strict=True)
    ]
    monitors = [
        asyncio.create_task(monitor_side(config, sink, stop)) for config in configs
    ]
    try:
        start.set()
        ready_wait = asyncio.ensure_future(
            asyncio.gather(*(event.wait() for event in ready))
        )
        done, _ = await asyncio.wait(
            [ready_wait, *runners], return_when=asyncio.FIRST_COMPLETED
        )
        if ready_wait not in done:
            await next(task for task in runners if task in done)
            raise RuntimeError("a demo side exited before synchronized fanout")
        await ready_wait
        baseline_metrics = await asyncio.gather(
            *(asyncio.to_thread(collect_metrics, config) for config in configs)
        )
        for config, values in zip(configs, baseline_metrics, strict=True):
            await sink(DemoEvent(side=config.side, kind="metrics", values=values))
        for config in configs:
            await sink(DemoEvent(side=config.side, kind="reset_clock"))
        fanout_start.set()
        await asyncio.gather(*runners)
    finally:
        for task in runners + monitors:
            task.cancel()
        await asyncio.gather(*runners, *monitors, return_exceptions=True)
        await asyncio.gather(*(side.close() for side in sides))
