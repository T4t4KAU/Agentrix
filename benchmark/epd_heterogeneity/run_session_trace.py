from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_trace import send_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:10001/v1/chat/completions"
    )
    parser.add_argument("--model", default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--timeout", type=float, default=300)
    return parser.parse_args()


async def run_workflow(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    results: asyncio.Queue[dict[str, Any]],
) -> None:
    initial_delay = int(records[0].get("arrival_offset_ms", 0)) / 1000
    await asyncio.sleep(initial_delay)
    for position, record in enumerate(records):
        request_id = f"epd-session-{uuid.uuid4()}"
        start_wall_ns = time.time_ns()
        start_monotonic_ns = time.monotonic_ns()
        response = await asyncio.to_thread(
            send_request,
            args.endpoint,
            args.model,
            record,
            request_id,
            args.timeout,
        )
        finish_monotonic_ns = time.monotonic_ns()
        await results.put(
            {
                **record,
                "request_id": request_id,
                "client_start_wall_ns": start_wall_ns,
                "client_start_monotonic_ns": start_monotonic_ns,
                "client_finish_monotonic_ns": finish_monotonic_ns,
                "client_latency_ms": (
                    finish_monotonic_ns - start_monotonic_ns
                )
                / 1e6,
                **response,
            }
        )
        if position + 1 < len(records):
            await asyncio.sleep(int(record.get("tool_gap_ms", 0)) / 1000)


async def run(args: argparse.Namespace) -> None:
    records = [
        json.loads(line)
        for line in args.trace.read_text(encoding="utf-8").splitlines()
        if line
    ]
    workflows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        workflows[int(record["workflow_id"])].append(record)
    for workflow in workflows.values():
        workflow.sort(key=lambda item: int(item["step_id"]))
    results: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    tasks = [
        asyncio.create_task(run_workflow(args, workflow, results))
        for workflow in workflows.values()
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for _ in range(len(records)):
            result = await results.get()
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()
    await asyncio.gather(*tasks)


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
