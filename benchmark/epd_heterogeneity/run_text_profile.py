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
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:19535/v1/completions"
    )
    parser.add_argument("--model", default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--token-id", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=300)
    return parser.parse_args()


def send_request(
    args: argparse.Namespace, request_id: str
) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "prompt": [args.token_id] * args.input_tokens,
        "max_tokens": args.output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }
    request = urllib.request.Request(
        args.endpoint,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = json.loads(response.read())
            return {
                "http_status": response.status,
                "usage": body.get("usage"),
                "error": None,
            }
    except urllib.error.HTTPError as error:
        return {
            "http_status": error.code,
            "usage": None,
            "error": error.read().decode(errors="replace"),
        }
    except Exception as error:
        return {"http_status": None, "usage": None, "error": repr(error)}


async def run_one(args: argparse.Namespace, index: int) -> dict[str, Any]:
    request_id = f"epd-text-{uuid.uuid4()}"
    start_wall_ns = time.time_ns()
    start_monotonic_ns = time.monotonic_ns()
    result = await asyncio.to_thread(send_request, args, request_id)
    finish_monotonic_ns = time.monotonic_ns()
    return {
        "index": index,
        "request_id": request_id,
        "input_tokens": args.input_tokens,
        "requested_output_tokens": args.output_tokens,
        "client_start_wall_ns": start_wall_ns,
        "client_start_monotonic_ns": start_monotonic_ns,
        "client_finish_monotonic_ns": finish_monotonic_ns,
        "client_latency_ms": (finish_monotonic_ns - start_monotonic_ns) / 1e6,
        **result,
    }


async def run(args: argparse.Namespace) -> None:
    tasks = [asyncio.create_task(run_one(args, index)) for index in range(args.concurrency)]
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
