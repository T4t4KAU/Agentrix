#!/usr/bin/env python3
"""Profile DP routing on a workload with strong prefix-cache affinity."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp
from prometheus_client.parser import text_string_to_metric_families


@dataclass(slots=True)
class RequestResult:
    document: int
    ttft_ms: float
    e2e_ms: float
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    output_token_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--documents", type=int, default=15)
    parser.add_argument(
        "--workload", choices=("revisit", "replicated", "cold"), default="revisit"
    )
    parser.add_argument("--warm-rank-counts", default="8,1")
    parser.add_argument("--policy-label", default="unknown")
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


async def reset_cache(session: aiohttp.ClientSession, base_url: str) -> int:
    deadline = time.monotonic() + 10
    attempts = 0
    while True:
        attempts += 1
        async with session.post(f"{base_url}/reset_prefix_cache") as response:
            response.raise_for_status()
            result = await response.json()
        if result.get("success") is True:
            return attempts
        if result.get("success") is not False or time.monotonic() >= deadline:
            raise RuntimeError(f"prefix cache reset did not succeed: {result}")
        await asyncio.sleep(0.05)


async def read_rank_metrics(
    session: aiohttp.ClientSession, base_url: str
) -> dict[str, dict[str, float]]:
    async with session.get(f"{base_url}/metrics") as response:
        response.raise_for_status()
        body = await response.text()
    names = {
        "vllm:request_success_total": "requests",
        "vllm:num_preemptions_total": "preemptions",
        "vllm:prefix_cache_hits_total": "cache_hits",
        "vllm:prefix_cache_queries_total": "cache_queries",
    }
    ranks: dict[str, dict[str, float]] = {}
    for family in text_string_to_metric_families(body):
        for sample in family.samples:
            if sample.name not in names or "engine" not in sample.labels:
                continue
            rank = ranks.setdefault(sample.labels["engine"], {})
            key = names[sample.name]
            rank[key] = rank.get(key, 0.0) + sample.value
    if len(ranks) < 2:
        raise RuntimeError("benchmark requires metrics from at least two DP ranks")
    return ranks


async def wait_for_completions(session, base_url, before, expected):
    deadline = time.monotonic() + 10
    while True:
        after = await read_rank_metrics(session, base_url)
        counts = {
            rank: {
                key: value - before.get(rank, {}).get(key, 0.0)
                for key, value in metrics.items()
            }
            for rank, metrics in after.items()
        }
        completed = sum(metrics.get("requests", 0) for metrics in counts.values())
        if completed == expected:
            return counts
        if completed > expected or time.monotonic() >= deadline:
            raise RuntimeError(
                f"expected {expected} engine completions, got {completed}"
            )
        await asyncio.sleep(0.05)


async def run_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: list[int],
    document: int,
    output_tokens: int,
    launch_delay_s: float = 0.0,
    rank: int | None = None,
) -> RequestResult:
    if launch_delay_s:
        await asyncio.sleep(launch_delay_s)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "return_token_ids": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, Any] | None = None
    generated_tokens: list[int] = []
    headers = {"X-data-parallel-rank": str(rank)} if rank is not None else {}
    async with session.post(
        f"{base_url}/v1/completions", json=payload, headers=headers
    ) as response:
        if response.status >= 400:
            body = await response.text()
            raise RuntimeError(f"request failed ({response.status}): {body}")
        async for raw_line in response.content:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line.removeprefix("data: "))
            for choice in chunk.get("choices", ()):
                token_ids = choice.get("token_ids") or []
                if token_ids:
                    generated_tokens.extend(token_ids)
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
    ended = time.perf_counter()
    if first_token_at is None or usage is None:
        raise RuntimeError("stream did not contain a token and final usage")
    details = usage.get("prompt_tokens_details") or {}
    if "cached_tokens" not in details:
        raise RuntimeError("enable prompt-token details on the server")
    if (
        usage["prompt_tokens"] != len(prompt)
        or usage["completion_tokens"] != output_tokens
        or len(generated_tokens) != output_tokens
    ):
        raise RuntimeError(f"unexpected token counts: {usage}")
    return RequestResult(
        document=document,
        ttft_ms=(first_token_at - started) * 1000,
        e2e_ms=(ended - started) * 1000,
        prompt_tokens=int(usage["prompt_tokens"]),
        cached_tokens=int(details.get("cached_tokens") or 0),
        completion_tokens=int(usage["completion_tokens"]),
        output_token_sha256=hashlib.sha256(
            json.dumps(generated_tokens).encode()
        ).hexdigest(),
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
        if args.workload == "replicated":
            prefix = warm_prompts[0][: args.prefix_tokens]
            revisit_prompts = [
                prefix + prompt[args.prefix_tokens :] for prompt in revisit_prompts
            ]
        corpus_sha256 = hashlib.sha256(
            json.dumps([warm_prompts, revisit_prompts]).encode()
        ).hexdigest()
        trials = []
        for trial in range(args.trials):
            reset_attempts = await reset_cache(session, args.base_url)
            await asyncio.sleep(1)

            warm_results = []
            if args.workload == "replicated":
                warm_plan = [
                    (0, warm_prompts[0], rank)
                    for rank, count in enumerate(args.warm_rank_counts)
                    for _ in range(count)
                ]
            elif args.workload == "revisit":
                warm_plan = [(i, prompt, None) for i, prompt in enumerate(warm_prompts)]
            else:
                warm_plan = []
            before_warm = await read_rank_metrics(session, args.base_url)
            for document, prompt, rank in warm_plan:
                warm_results.append(
                    await run_request(
                        session,
                        args.base_url,
                        model,
                        prompt,
                        document,
                        1,
                        rank=rank,
                    )
                )
                await asyncio.sleep(args.settle_ms / 1000)
            warm_rank_metrics = await wait_for_completions(
                session, args.base_url, before_warm, len(warm_results)
            )

            revisit_order = list(range(args.documents))
            if args.revisit_order == "shuffled":
                random.Random(args.seed + 1_000_000 + trial).shuffle(revisit_order)

            before_batch = await read_rank_metrics(session, args.base_url)
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
            rank_metrics = await wait_for_completions(
                session, args.base_url, before_batch, len(revisit_results)
            )
            summary = summarize(revisit_results, makespan_s)
            trials.append(
                {
                    "trial": trial + 1,
                    "cache_reset_attempts": reset_attempts,
                    "warm_cached_tokens": sum(r.cached_tokens for r in warm_results),
                    "summary": summary,
                    "warm_rank_metrics": warm_rank_metrics,
                    "rank_metrics": rank_metrics,
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
            "workload": args.workload,
            "warm_rank_counts": args.warm_rank_counts,
            "policy_label": args.policy_label,
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
        "corpus_sha256": corpus_sha256,
        "trials": trials,
    }


def main() -> None:
    args = parse_args()
    args.warm_rank_counts = [int(value) for value in args.warm_rank_counts.split(",")]
    if (
        min(
            args.documents,
            args.trials,
            args.prefix_tokens,
            args.suffix_tokens,
            args.output_tokens,
        )
        <= 0
    ):
        raise ValueError("request counts and token lengths must be positive")
    if len(args.warm_rank_counts) < 2 or min(args.warm_rank_counts) <= 0:
        raise ValueError("provide positive warm counts for at least two DP ranks")
    payload = asyncio.run(main_async(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
