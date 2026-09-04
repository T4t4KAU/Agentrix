#!/usr/bin/env python3
"""Profile DP routing on a workload with strong prefix-cache affinity."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp


@dataclass(slots=True)
class RequestResult:
    document: int
    ttft_ms: float
    e2e_ms: float
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--documents", type=int, default=15)
    parser.add_argument("--prefix-tokens", type=int, default=3072)
    parser.add_argument("--suffix-tokens", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--settle-ms", type=float, default=150)
    parser.add_argument("--launch-gap-ms", type=float, default=1)
    parser.add_argument("--revisit-order", choices=("same", "shuffled"), default="same")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def make_prompts(
    documents: int,
    prefix_tokens: int,
    suffix_tokens: int,
    seed: int,
) -> tuple[list[list[int]], list[list[int]]]:
    warm_prompts: list[list[int]] = []
    revisit_prompts: list[list[int]] = []
    for document in range(documents):
        rng = random.Random(seed + document)
        prefix = [rng.randrange(1000, 30000) for _ in range(prefix_tokens)]
        warm_suffix = [100 + document] * suffix_tokens
        revisit_suffix = [500 + document] * suffix_tokens
        warm_prompts.append([*prefix, *warm_suffix])
        revisit_prompts.append([*prefix, *revisit_suffix])
    return warm_prompts, revisit_prompts


async def discover_model(session: aiohttp.ClientSession, base_url: str) -> str:
    async with session.get(f"{base_url}/v1/models") as response:
        response.raise_for_status()
        payload = await response.json()
    return payload["data"][0]["id"]


async def reset_cache(session: aiohttp.ClientSession, base_url: str) -> None:
    async with session.post(f"{base_url}/reset_prefix_cache") as response:
        response.raise_for_status()


async def run_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: list[int],
    document: int,
    output_tokens: int,
    launch_delay_s: float = 0.0,
) -> RequestResult:
    if launch_delay_s:
        await asyncio.sleep(launch_delay_s)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, Any] | None = None
    async with session.post(f"{base_url}/v1/completions", json=payload) as response:
        if response.status >= 400:
            body = await response.text()
            raise RuntimeError(f"request failed ({response.status}): {body}")
        async for raw_line in response.content:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line.removeprefix("data: "))
            if chunk.get("choices") and first_token_at is None:
                first_token_at = time.perf_counter()
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
    ended = time.perf_counter()
    if first_token_at is None or usage is None:
        raise RuntimeError("stream did not contain a token and final usage")
    details = usage.get("prompt_tokens_details") or {}
    return RequestResult(
        document=document,
        ttft_ms=(first_token_at - started) * 1000,
        e2e_ms=(ended - started) * 1000,
        prompt_tokens=int(usage["prompt_tokens"]),
        cached_tokens=int(details.get("cached_tokens") or 0),
        completion_tokens=int(usage["completion_tokens"]),
    )


def summarize(results: list[RequestResult], makespan_s: float) -> dict[str, Any]:
    ttft = [result.ttft_ms for result in results]
    e2e = [result.e2e_ms for result in results]
    cached = [result.cached_tokens for result in results]
    prompt = [result.prompt_tokens for result in results]
    return {
        "requests": len(results),
        "cached_requests": sum(value > 0 for value in cached),
        "cache_request_hit_rate": sum(value > 0 for value in cached) / len(cached),
        "cached_token_rate": sum(cached) / sum(prompt),
        "cached_tokens_total": sum(cached),
        "prompt_tokens_total": sum(prompt),
        "ttft_mean_ms": statistics.fmean(ttft),
        "ttft_p50_ms": percentile(ttft, 0.5),
        "ttft_p95_ms": percentile(ttft, 0.95),
        "e2e_mean_ms": statistics.fmean(e2e),
        "e2e_p50_ms": percentile(e2e, 0.5),
        "e2e_p95_ms": percentile(e2e, 0.95),
        "batch_makespan_ms": makespan_s * 1000,
        "requests_per_second": len(results) / makespan_s,
    }


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        model = args.model or await discover_model(session, args.base_url)
        warm_prompts, revisit_prompts = make_prompts(
            args.documents,
            args.prefix_tokens,
            args.suffix_tokens,
            args.seed,
        )
        trials = []
        for trial in range(args.trials):
            await reset_cache(session, args.base_url)
            await asyncio.sleep(1)

            warm_results = []
            for document, prompt in enumerate(warm_prompts):
                warm_results.append(
                    await run_request(
                        session,
                        args.base_url,
                        model,
                        prompt,
                        document,
                        args.output_tokens,
                    )
                )
                await asyncio.sleep(args.settle_ms / 1000)

            revisit_order = list(range(args.documents))
            if args.revisit_order == "shuffled":
                random.Random(args.seed + 1_000_000 + trial).shuffle(revisit_order)

            batch_started = time.perf_counter()
            revisit_results = await asyncio.gather(
                *[
                    run_request(
                        session,
                        args.base_url,
                        model,
                        revisit_prompts[document],
                        document,
                        args.output_tokens,
                        launch_position * args.launch_gap_ms / 1000,
                    )
                    for launch_position, document in enumerate(revisit_order)
                ]
            )
            makespan_s = time.perf_counter() - batch_started
            summary = summarize(revisit_results, makespan_s)
            trials.append(
                {
                    "trial": trial + 1,
                    "warm_cached_tokens": sum(r.cached_tokens for r in warm_results),
                    "summary": summary,
                    "warm_requests": [asdict(result) for result in warm_results],
                    "revisit_requests": [asdict(result) for result in revisit_results],
                }
            )
            print(
                f"trial={trial + 1} hits={summary['cached_requests']}/"
                f"{args.documents} cached={summary['cached_token_rate']:.1%} "
                f"TTFT p50/p95={summary['ttft_p50_ms']:.1f}/"
                f"{summary['ttft_p95_ms']:.1f} ms "
                f"makespan={summary['batch_makespan_ms']:.1f} ms",
                flush=True,
            )

    numeric_keys = [
        "cache_request_hit_rate",
        "cached_token_rate",
        "ttft_mean_ms",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "e2e_mean_ms",
        "e2e_p50_ms",
        "e2e_p95_ms",
        "batch_makespan_ms",
        "requests_per_second",
    ]
    aggregate = {
        key: {
            "median": statistics.median(trial["summary"][key] for trial in trials),
            "min": min(trial["summary"][key] for trial in trials),
            "max": max(trial["summary"][key] for trial in trials),
        }
        for key in numeric_keys
    }
    return {
        "configuration": {
            "base_url": args.base_url,
            "model": model,
            "documents": args.documents,
            "prefix_tokens": args.prefix_tokens,
            "suffix_tokens": args.suffix_tokens,
            "output_tokens": args.output_tokens,
            "trials": args.trials,
            "settle_ms": args.settle_ms,
            "launch_gap_ms": args.launch_gap_ms,
            "revisit_order": args.revisit_order,
            "seed": args.seed,
        },
        "aggregate": aggregate,
        "trials": trials,
    }


def main() -> None:
    args = parse_args()
    payload = asyncio.run(main_async(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
