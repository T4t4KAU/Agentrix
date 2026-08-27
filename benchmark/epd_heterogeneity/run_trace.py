from __future__ import annotations

import argparse
import asyncio
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


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


def text_for_approximate_tokens(token_count: int) -> str:
    return "Observe the current state and choose the next action. " * max(
        1, token_count // 11
    )


def send_request(
    endpoint: str,
    model: str,
    record: dict[str, Any],
    request_id: str,
    timeout: float,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    image_path = record.get("image_path")
    if image_path:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": Path(str(image_path)).as_uri()},
            }
        )
    content.append(
        {
            "type": "text",
            "text": text_for_approximate_tokens(int(record["text_input_tokens"])),
        }
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": int(record["max_tokens"]),
        "temperature": 0,
        "stream": False,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
            return {
                "http_status": response.status,
                "usage": body.get("usage"),
                "response_id": body.get("id"),
                "error": None,
            }
    except urllib.error.HTTPError as error:
        return {
            "http_status": error.code,
            "usage": None,
            "response_id": None,
            "error": error.read().decode(errors="replace"),
        }
    except Exception as error:
        return {
            "http_status": None,
            "usage": None,
            "response_id": None,
            "error": repr(error),
        }


async def run_one(
    endpoint: str,
    model: str,
    trace_record: dict[str, Any],
    trace_start_monotonic: float,
    timeout: float,
) -> dict[str, Any]:
    target = trace_start_monotonic + int(trace_record["arrival_offset_ms"]) / 1000
    await asyncio.sleep(max(0, target - time.monotonic()))
    request_id = f"epd-{uuid.uuid4()}"
    start_wall_ns = time.time_ns()
    start_monotonic_ns = time.monotonic_ns()
    result = await asyncio.to_thread(
        send_request, endpoint, model, trace_record, request_id, timeout
    )
    finish_monotonic_ns = time.monotonic_ns()
    return {
        **trace_record,
        "request_id": request_id,
        "client_start_wall_ns": start_wall_ns,
        "client_start_monotonic_ns": start_monotonic_ns,
        "client_finish_monotonic_ns": finish_monotonic_ns,
        "client_latency_ms": (finish_monotonic_ns - start_monotonic_ns) / 1e6,
        **result,
    }


async def run(args: argparse.Namespace) -> None:
    records = [
        json.loads(line)
        for line in args.trace.read_text(encoding="utf-8").splitlines()
        if line
    ]
    trace_start = time.monotonic() + 1
    tasks = [
        asyncio.create_task(
            run_one(
                args.endpoint,
                args.model,
                record,
                trace_start,
                args.timeout,
            )
        )
        for record in records
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for task in asyncio.as_completed(tasks):
            result = await task
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
