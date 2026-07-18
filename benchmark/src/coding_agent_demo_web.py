from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import re
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from coding_agent_demo import (
    DemoEvent,
    SideConfig,
    SideState,
    build_parser,
    expected_request_count,
    load_cases,
    parse_gpu_ids,
    run_live_pair,
    run_mock_side,
    reset_prefix_caches,
    side_state_payload,
)


ASSET_DIR = Path(__file__).resolve().parents[1] / "web" / "coding_agent_demo"


def normalize_route_request_id(value: str) -> str:
    request_id = value.removeprefix("chatcmpl-")
    return re.sub(r"-[0-9a-f]{8}$", "", request_id)


class WebDemo:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.configs = (
            SideConfig(
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
            SideConfig(
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
        )
        self.states = {config.side: SideState(config) for config in self.configs}
        self.phase = "idle"
        self.error: str | None = None
        self.workload: asyncio.Task[None] | None = None
        self.stop = asyncio.Event()
        self.sockets: set[web.WebSocketResponse] = set()
        self.reset_sides: set[str] = set()
        self.broadcast_task: asyncio.Task[None] | None = None
        self.route_task: asyncio.Task[None] | None = None
        self.route_cursors = {"left": 0, "right": 0}
        self.expected_requests = 0

    async def start_background(self, _: web.Application) -> None:
        self.broadcast_task = asyncio.create_task(self.broadcast_loop())
        self.route_task = asyncio.create_task(self.route_loop())

    async def stop_background(self, _: web.Application) -> None:
        self.stop.set()
        if self.workload is not None:
            self.workload.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.workload
        if self.broadcast_task is not None:
            self.broadcast_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.broadcast_task
        if self.route_task is not None:
            self.route_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.route_task

    async def index(self, _: web.Request) -> web.FileResponse:
        return web.FileResponse(ASSET_DIR / "index.html")

    async def websocket(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse(heartbeat=20)
        await socket.prepare(request)
        self.sockets.add(socket)
        await socket.send_json(self.snapshot())
        try:
            async for message in socket:
                if message.type == WSMsgType.TEXT and message.data == "start":
                    await self.start()
        finally:
            self.sockets.discard(socket)
        return socket

    async def start_api(self, _: web.Request) -> web.Response:
        started = await self.start()
        return web.json_response({"started": started, "phase": self.phase})

    async def start(self) -> bool:
        if self.workload is not None and not self.workload.done():
            return False
        self.states = {config.side: SideState(config) for config in self.configs}
        self.phase = "preparing"
        self.error = None
        self.reset_sides.clear()
        self.route_cursors = {"left": 0, "right": 0}
        self.stop = asyncio.Event()
        self.workload = asyncio.create_task(self.run_workload())
        return True

    async def run_workload(self) -> None:
        try:
            cases = load_cases(
                self.args.cases,
                offset=self.args.case_offset,
                limit=self.args.case_count,
            )
            self.expected_requests = expected_request_count(
                cases, rounds=self.args.rounds, mock=self.args.mock
            )
            for state in self.states.values():
                state.expected_requests = self.expected_requests
            if self.args.mock:
                start = asyncio.Event()
                tasks = [
                    run_mock_side(
                        self.configs[0],
                        cases,
                        sink=self.emit,
                        start=start,
                        delay=0.025,
                    ),
                    run_mock_side(
                        self.configs[1],
                        cases,
                        sink=self.emit,
                        start=start,
                        delay=0.065,
                    ),
                ]
                self.phase = "running"
                start.set()
                await asyncio.gather(*tasks)
            else:
                await reset_prefix_caches(self.configs)
                await run_live_pair(
                    self.configs,
                    cases,
                    rounds=self.args.rounds,
                    max_tokens=self.args.max_tokens,
                    sink=self.emit,
                    stop=self.stop,
                )
            self.phase = "completed"
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.error = str(error)
            self.phase = "error"

    async def emit(self, event: DemoEvent) -> None:
        state = self.states[event.side]
        state.apply(event)
        if (
            event.kind == "completed"
            and self.expected_requests
            and state.completed_requests == self.expected_requests
        ):
            state.ended_at = time.monotonic()
        if event.kind == "reset_clock":
            self.reset_sides.add(event.side)
            if len(self.reset_sides) == 2:
                self.phase = "running"

    def snapshot(self) -> dict[str, Any]:
        left = side_state_payload(self.states["left"])
        right = side_state_payload(self.states["right"])
        right_rate = float(right["output_tokens_per_s"])
        return {
            "phase": self.phase,
            "error": self.error,
            "left": left,
            "right": right,
            "speedup": (
                float(left["output_tokens_per_s"]) / right_rate
                if right_rate > 0
                else None
            ),
        }

    async def broadcast_loop(self) -> None:
        while True:
            if self.sockets:
                payload = json.dumps(self.snapshot(), ensure_ascii=False)
                dead = []
                for socket in tuple(self.sockets):
                    try:
                        await socket.send_str(payload)
                    except ConnectionError:
                        dead.append(socket)
                for socket in dead:
                    self.sockets.discard(socket)
            await asyncio.sleep(0.10)

    async def route_loop(self) -> None:
        while True:
            if self.phase in {"preparing", "running"} and not self.args.mock:
                await asyncio.gather(
                    *(self.poll_routes(config) for config in self.configs)
                )
            await asyncio.sleep(0.15)

    async def poll_routes(self, config: SideConfig) -> None:
        cursor = self.route_cursors[config.side]
        endpoint = (
            f"{config.base_url.removesuffix('/v1')}/agentrix/demo/routes?after={cursor}"
        )

        def fetch() -> dict[str, Any]:
            with urllib.request.urlopen(endpoint, timeout=2) as response:
                return json.load(response)

        try:
            payload = await asyncio.to_thread(fetch)
        except Exception:
            return
        self.route_cursors[config.side] = int(payload["latest_sequence"])
        for route in payload["events"]:
            request_id = normalize_route_request_id(str(route["request_id"]))
            if request_id.startswith("bootstrap:"):
                continue
            self.states[config.side].apply(
                DemoEvent(
                    side=config.side,
                    kind="route",
                    request_id=request_id,
                    values={"dp_rank": int(route["dp_rank"])},
                )
            )


def make_app(args: argparse.Namespace) -> web.Application:
    demo = WebDemo(args)
    app = web.Application()
    app.router.add_get("/", demo.index)
    app.router.add_get("/ws", demo.websocket)
    app.router.add_post("/api/start", demo.start_api)
    app.router.add_static("/assets", ASSET_DIR)
    app.on_startup.append(demo.start_background)
    app.on_cleanup.append(demo.stop_background)
    return app


def main() -> None:
    parser = build_parser()
    parser.description = "Serve the host-side Agentrix coding-agent dashboard"
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8088)
    parser.add_argument("--no-open-browser", action="store_true")
    args = parser.parse_args()
    if not args.no_open_browser:
        threading.Timer(
            0.6, webbrowser.open, args=(f"http://{args.web_host}:{args.web_port}",)
        ).start()
    web.run_app(make_app(args), host=args.web_host, port=args.web_port)


if __name__ == "__main__":
    main()
