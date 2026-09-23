"""Collect a short cached-decode diagnostic trace, separate from official AgentX."""

import argparse
import json
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from transformers import AutoTokenizer


def post(url, path, body=None, rank=0):
    request = urllib.request.Request(
        url + path,
        data=json.dumps(body).encode() if body is not None else b"",
        headers={"Content-Type": "application/json", "X-data-parallel-rank": str(rank)},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        data = response.read()
    return json.loads(data) if data else None


def run_batch(args, tokens, salt, batch_size, output_tokens):
    barrier = Barrier(batch_size * 2)

    def generate(rank):
        barrier.wait(timeout=30)
        started = time.monotonic()
        result = post(
            args.url,
            "/v1/completions",
            {
                "model": args.model,
                "prompt": tokens,
                "max_tokens": output_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "cache_salt": salt,
            },
            rank,
        )
        assert result["usage"]["completion_tokens"] == output_tokens, result
        return {
            "rank": rank,
            "seconds": time.monotonic() - started,
            "usage": result["usage"],
        }

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=batch_size * 2) as executor:
        requests = list(executor.map(generate, [0] * batch_size + [1] * batch_size))
    return {
        "requests_per_rank": batch_size,
        "output_tokens_per_request": output_tokens,
        "seconds": time.monotonic() - started,
        "requests": requests,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--model-path", default="/data/models/Qwen3.5-9B")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timings-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokens = tokenizer.encode(
        "Reference text for a cached decoding diagnostic.\n" * 1500
    )[:8192]
    tokens += tokenizer.encode("\nContinue describing this reference text in detail.")
    salt = uuid.uuid4().hex
    run_batch(args, tokens, salt, 8, 4)
    timings = [
        run_batch(args, tokens, salt, size, 32)
        for _ in range(args.repeats)
        for size in (1, 3, 5, 8)
    ]
    profiled = None
    if not args.timings_only:
        post(args.url, "/start_profile")
        try:
            profiled = run_batch(args, tokens, salt, 8, 32)
        finally:
            post(args.url, "/stop_profile")
    args.output.write_text(
        json.dumps(
            {
                "kind": "cached_decode_diagnostic_not_official_benchmark",
                "unprofiled_batches": timings,
                "profiled_batch": profiled,
            },
            indent=2,
        )
        + "\n"
    )
    print("DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
