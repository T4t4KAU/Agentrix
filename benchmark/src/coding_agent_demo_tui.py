from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Static

from coding_agent_demo import (
    DemoEvent,
    SideConfig,
    SideState,
    WorkloadSide,
    build_parser,
    collect_metrics,
    expected_request_count,
    load_cases,
    monitor_side,
    parse_gpu_ids,
    run_mock_side,
    reset_prefix_caches,
)


class DemoApp(App[None]):
    CSS = """
    Screen {
        background: #080d14;
        color: #e5e7eb;
    }
    #columns {
        height: 1fr;
    }
    .side {
        width: 1fr;
        height: 1fr;
    }
    #left-side {
        border-right: solid #334155;
    }
    .title {
        height: 3;
        content-align: center middle;
        text-style: bold;
        background: #111827;
    }
    .metrics {
        height: 5;
        padding: 0 2;
        background: #0f172a;
    }
    .agents {
        height: 1fr;
        padding: 1 2;
        background: #080d14;
        content-align: left bottom;
    }
    Footer {
        height: 1;
        background: #111827;
        color: #94a3b8;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("space", "toggle_pause", "Pause view"),
    ]

    def __init__(self, args: Any) -> None:
        super().__init__()
        self.args = args
        self.events: asyncio.Queue[DemoEvent] = asyncio.Queue()
        self.stop = asyncio.Event()
        self.paused = False
        self.tasks: list[asyncio.Task[Any]] = []
        self.configs = {
            "left": SideConfig(
                side="left",
                title=f"AGENTRIX · GPU {args.left_gpus}",
                base_url=args.left_base_url,
                model=args.left_model,
                gpu_ids=parse_gpu_ids(args.left_gpus),
                prompt_compaction=True,
                routing_label="PREFIX-AWARE DP",
                color="#2dd4bf",
                gpu_metrics_url=args.left_gpu_metrics_url,
            ),
            "right": SideConfig(
                side="right",
                title=f"vLLM BASELINE · GPU {args.right_gpus}",
                base_url=args.right_base_url,
                model=args.right_model,
                gpu_ids=parse_gpu_ids(args.right_gpus),
                prompt_compaction=False,
                routing_label="ORDINARY DP",
                color="#94a3b8",
                gpu_metrics_url=args.right_gpu_metrics_url,
            ),
        }
        self.states = {side: SideState(config) for side, config in self.configs.items()}

    def compose(self) -> ComposeResult:
        with Horizontal(id="columns"):
            for side in ("left", "right"):
                config = self.configs[side]
                with Vertical(classes="side", id=f"{side}-side"):
                    yield Static(
                        Text(config.title, style=f"bold {config.color}"),
                        classes="title",
                        id=f"{side}-title",
                    )
                    yield Static(classes="metrics", id=f"{side}-metrics")
                    yield Static(classes="agents", id=f"{side}-agents")
        yield Footer()

    async def on_mount(self) -> None:
        self.set_interval(0.05, self._drain_events)
        self.set_interval(0.20, self._render)
        self.tasks.append(asyncio.create_task(self._run_demo()))

    async def on_unmount(self) -> None:
        self.stop.set()
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            with suppress(asyncio.CancelledError):
                await task

    async def emit(self, event: DemoEvent) -> None:
        await self.events.put(event)

    async def _run_demo(self) -> None:
        cases = load_cases(
            self.args.cases,
            offset=self.args.case_offset,
            limit=self.args.case_count,
        )
        expected = expected_request_count(
            cases, rounds=self.args.rounds, mock=self.args.mock
        )
        for state in self.states.values():
            state.expected_requests = expected
        if not self.args.mock:
            await reset_prefix_caches(list(self.configs.values()))
        start = asyncio.Event()
        if self.args.mock:
            runners = [
                asyncio.create_task(
                    run_mock_side(
                        self.configs["left"],
                        cases,
                        sink=self.emit,
                        start=start,
                        delay=0.025,
                    )
                ),
                asyncio.create_task(
                    run_mock_side(
                        self.configs["right"],
                        cases,
                        sink=self.emit,
                        start=start,
                        delay=0.065,
                    )
                ),
            ]
        else:
            sides = [
                WorkloadSide(
                    config,
                    cases,
                    rounds=self.args.rounds,
                    max_tokens=self.args.max_tokens,
                    sink=self.emit,
                )
                for config in self.configs.values()
            ]
            ready = [asyncio.Event(), asyncio.Event()]
            fanout_start = asyncio.Event()
            runners = [
                asyncio.create_task(side.run(start, side_ready, fanout_start))
                for side, side_ready in zip(sides, ready, strict=True)
            ]
            self.tasks.extend(
                asyncio.create_task(monitor_side(config, self.emit, self.stop))
                for config in self.configs.values()
            )
        await asyncio.sleep(0.5)
        start.set()
        if not self.args.mock:
            ready_wait = asyncio.ensure_future(
                asyncio.gather(*(event.wait() for event in ready))
            )
            done, _ = await asyncio.wait(
                [ready_wait, *runners], return_when=asyncio.FIRST_COMPLETED
            )
            if ready_wait not in done:
                await next(task for task in runners if task in done)
                raise RuntimeError("a demo side exited before the synchronized fanout")
            await ready_wait
            baseline_metrics = await asyncio.gather(
                *(
                    asyncio.to_thread(collect_metrics, config)
                    for config in self.configs.values()
                )
            )
            for config, values in zip(
                self.configs.values(), baseline_metrics, strict=True
            ):
                await self.emit(
                    DemoEvent(side=config.side, kind="metrics", values=values)
                )
            await self.emit(DemoEvent(side="left", kind="reset_clock"))
            await self.emit(DemoEvent(side="right", kind="reset_clock"))
            fanout_start.set()
        await asyncio.gather(*runners)
        if not self.args.mock:
            await asyncio.gather(*(side.close() for side in sides))

    def _drain_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except asyncio.QueueEmpty:
                break
            self.states[event.side].apply(event)

    def _render(self) -> None:
        if self.paused:
            return
        for side, state in self.states.items():
            self.query_one(f"#{side}-metrics", Static).update(metric_text(state))
            self.query_one(f"#{side}-agents", Static).update(agent_text(state))

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused


def metric_text(state: SideState) -> Text:
    color = state.config.color
    output = Text()
    output.append(f"{compact_number(state.displayed_tokens)} TOKENS", f"bold {color}")
    output.append(
        f"   ✓ {state.completed_requests}/{state.expected_requests} DONE",
        "bold #22d3ee",
    )
    output.append(
        f"   ◌ {state.prefill_requests} PREFILL/SCHED",
        "#facc15" if state.prefill_requests else "#64748b",
    )
    output.append(
        f"   ● {state.generating_requests} GENERATING",
        "#4ade80" if state.generating_requests else "#64748b",
    )
    output.append(
        f"   … {state.queued_requests} QUEUED\n",
        "#94a3b8" if state.queued_requests else "#64748b",
    )
    output.append(f"{state.output_tokens_per_s:,.1f} tok/s", color)
    queue_avg = (
        f"{state.average_queue_time_ms:,.0f} ms"
        if state.average_queue_time_ms is not None
        else "—"
    )
    output.append(
        f"   QUEUED SEQS {state.waiting_sequences}   AVG QUEUE {queue_avg}",
        "#facc15" if state.waiting_sequences else "#94a3b8",
    )
    output.append(
        f"   PREFILL COMPUTE {compact_number(state.run_computed_prefill_tokens)}",
        "#60a5fa",
    )
    output.append("\n")
    if state.hbm_used_mib:
        used = sum(state.hbm_used_mib) / len(state.hbm_used_mib) / 1024
        total = sum(state.hbm_total_mib) / len(state.hbm_total_mib) / 1024
        output.append(f"HBM avg/GPU {used:.1f}/{total:.1f} GiB", "#f8fafc")
        if state.kv_usage:
            output.append(
                f"   active KV {100 * sum(state.kv_usage) / len(state.kv_usage):.1f}%",
                "#c084fc",
            )
        output.append("\n")
    output.append(
        state.config.routing_label,
        "bold #c084fc" if state.config.prompt_compaction else "#94a3b8",
    )
    if state.rank_loads:
        output.append("   DP load (run/queue)", "#64748b")
        for rank, (running, waiting) in enumerate(state.rank_loads):
            output.append(f"   D{rank} {running}/{waiting}", "#cbd5e1")
    else:
        output.append("   waiting for server telemetry", "italic #64748b")
    return output


def agent_text(state: SideState) -> Text:
    output = Text()
    selected = []
    selected_ids = set()
    for request_id in reversed(state.display_order):
        if request_id in selected_ids:
            continue
        stream = state.streams.get(request_id)
        if stream is not None and stream.status in {"generating", "prefill"}:
            selected.append(stream)
            selected_ids.add(request_id)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        for request_id in reversed(state.display_order):
            stream = state.streams.get(request_id)
            if request_id not in selected_ids and stream is not None:
                selected.append(stream)
                selected_ids.add(request_id)
            if len(selected) == 3:
                break
    for stream in reversed(selected):
        symbol, status_color = {
            "queued": ("…", "#64748b"),
            "prefill": ("◌", "#facc15"),
            "generating": ("●", "#4ade80"),
            "completed": ("✓", "#22d3ee"),
            "error": ("✗", "#f87171"),
        }[stream.status]
        status = (
            "PREFILL / SCHEDULING"
            if stream.status == "prefill"
            else stream.status.upper()
        )
        output.append(
            f"{symbol} [{short_case(stream.case_id)} · agent-{stream.agent_id:02d}] "
            f"{status}\n",
            f"bold {status_color}",
        )
        body = stream.text[-1600:] if stream.text else "Waiting for the first token…"
        output.append(
            body.strip() + ("▌" if stream.status == "generating" else ""), "#e5e7eb"
        )
        output.append("\n\n")
    if not selected:
        output.append("Preparing identical coding-agent workloads…", "italic #64748b")
    return output


def compact_number(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def short_case(case_id: str) -> str:
    return case_id if len(case_id) <= 24 else case_id[:21] + "…"


def main() -> None:
    args = build_parser().parse_args()
    DemoApp(args).run()


if __name__ == "__main__":
    main()
