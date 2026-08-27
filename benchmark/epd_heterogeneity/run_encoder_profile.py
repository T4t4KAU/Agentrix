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
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--image-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:10001/v1/chat/completions"
    )
    parser.add_argument("--model", default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--timeout", type=float, default=300)
    return parser.parse_args()


def load_images(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in args.image_manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    bucket = [record for record in records if record["name"] == args.bucket]
    selected = bucket[args.image_offset : args.image_offset + args.requests]
    if len(selected) != args.requests:
        raise ValueError(
            f"bucket {args.bucket!r} has {len(bucket)} images, but offset "
            f"{args.image_offset} and {args.requests} requests were requested"
        )
    return selected


def send_request(
    args: argparse.Namespace,
    image: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": Path(image["path"]).as_uri()},
                    },
                    {"type": "text", "text": "Describe the image briefly."},
                ],
            }
        ],
        "max_tokens": 1,
        "temperature": 0,
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


async def worker(
    worker_id: int,
    args: argparse.Namespace,
    queue: asyncio.Queue[tuple[int, dict[str, Any]]],
    results: asyncio.Queue[dict[str, Any]],
) -> None:
    while True:
        try:
            index, image = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        request_id = f"epd-encoder-{uuid.uuid4()}"
        start_wall_ns = time.time_ns()
        start_monotonic_ns = time.monotonic_ns()
        response = await asyncio.to_thread(send_request, args, image, request_id)
        finish_monotonic_ns = time.monotonic_ns()
        await results.put(
            {
                "index": index,
                "worker_id": worker_id,
                "request_id": request_id,
                "bucket": args.bucket,
                "image_path": image["path"],
                "expected_visual_tokens": image["expected_visual_tokens"],
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
        queue.task_done()


async def run(args: argparse.Namespace) -> None:
    images = load_images(args)
    queue: asyncio.Queue[tuple[int, dict[str, Any]]] = asyncio.Queue()
    results: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for index, image in enumerate(images):
        queue.put_nowait((index, image))
    workers = [
        asyncio.create_task(worker(worker_id, args, queue, results))
        for worker_id in range(args.concurrency)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        completed = 0
        while completed < len(images):
            result = await results.get()
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()
            completed += 1
    await asyncio.gather(*workers)


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
