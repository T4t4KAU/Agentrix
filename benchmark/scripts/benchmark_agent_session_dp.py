#!/usr/bin/env python3
"""Profile data-parallel routing on concurrent multi-turn agent sessions."""

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
    session: int
    phase: str
    ttft_ms: float
    e2e_ms: float
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--policy-label", default="unknown")
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--shared-prefix-tokens", type=int, default=2048)
    parser.add_argument("--session-tokens", type=int, default=1024)
    parser.add_argument("--followup-tokens", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--reset-settle-s", type=float, default=1.0)
    parser.add_argument("--phase-settle-ms", type=float, default=150)
    parser.add_argument("--launch-gap-ms", type=float, default=1)
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
    sessions: int,
    shared_prefix_tokens: int,
    session_tokens: int,
    followup_tokens: int,
    seed: int,
) -> tuple[list[int], list[list[int]], list[list[int]]]:
    shared_rng = random.Random(seed)
    shared = [shared_rng.randrange(1000, 30000) for _ in range(shared_prefix_tokens)]
    first_turns: list[list[int]] = []
    followups: list[list[int]] = []
    for session_id in range(sessions):
        rng = random.Random(seed + session_id + 1)
        private = [rng.randrange(1000, 30000) for _ in range(session_tokens)]
        followup = [rng.randrange(1000, 30000) for _ in range(followup_tokens)]
        first = [*shared, *private]
        first_turns.append(first)
        followups.append([*first, *followup])
    return shared, first_turns, followups


async def discover_model(session: aiohttp.ClientSession, base_url: str) -> str:
    async with session.get(f"{base_url}/v1/models") as response:
        response.raise_for_status()
        payload = await response.json()
    return payload["data"][0]["id"]


async def reset_cache(session: aiohttp.ClientSession, base_url: str) -> None:
    async with session.post(f"{base_url}/reset_prefix_cache") as response:
        response.raise_for_status()


async def read_prompt_sources(
    session: aiohttp.ClientSession, base_url: str
) -> dict[str, float]:
    async with session.get(f"{base_url}/metrics") as response:
        response.raise_for_status()
        body = await response.text()
    totals: dict[str, float] = {}
    metric = "vllm:prompt_tokens_by_source_total{"
    for line in body.splitlines():
        if not line.startswith(metric) or 'source="' not in line:
            continue
        source = line.split('source="', 1)[1].split('"', 1)[0]
        totals[source] = totals.get(source, 0.0) + float(line.rsplit(" ", 1)[1])
    return totals


async def run_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: list[int],
    session_id: int,
    phase: str,
    turn: int,
    history_tokens: int,
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
        "vllm_xargs": {
            "agentrix_session_id": f"session-{session_id}" if session_id >= 0 else "",
            "agentrix_turn": turn,
            "agentrix_history_tokens": history_tokens,
        },
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
        session=session_id,
        phase=phase,
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
        "api_reported_cached_requests": sum(value > 0 for value in cached),
        "api_reported_cache_request_hit_rate": (
            sum(value > 0 for value in cached) / len(cached)
        ),
        "api_reported_cached_token_rate": sum(cached) / sum(prompt),
        "api_reported_cached_tokens_total": sum(cached),
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


def add_server_metrics(
    summary: dict[str, Any],
    before: dict[str, float],
    after: dict[str, float],
) -> None:
    sources = {
        source: after.get(source, 0.0) - before.get(source, 0.0)
        for source in set(before) | set(after)
    }
    local_hit = sources.get("local_cache_hit", 0.0)
    local_compute = sources.get("local_compute", 0.0)
    external = sources.get("external_kv_transfer", 0.0)
    total = local_hit + local_compute + external
    summary.update(
        {
            "server_local_cache_hit_tokens": local_hit,
            "server_local_compute_tokens": local_compute,
            "server_external_kv_tokens": external,
            "server_local_cache_hit_rate": local_hit / total if total else 0.0,
            "server_reused_token_rate": (
                (local_hit + external) / total if total else 0.0
            ),
        }
    )


async def run_batch(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    model: str,
    prompts: list[list[int]],
    phase: str,
    turn: int,
    history_tokens: list[int],
) -> tuple[list[RequestResult], float]:
    started = time.perf_counter()
    results = await asyncio.gather(
        *[
            run_request(
                session,
                args.base_url,
                model,
                prompt,
                session_id,
                phase,
                turn,
                history_tokens[session_id],
                args.output_tokens,
                session_id * args.launch_gap_ms / 1000,
            )
            for session_id, prompt in enumerate(prompts)
        ]
    )
    return results, time.perf_counter() - started


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        model = args.model or await discover_model(session, args.base_url)
        shared, first_turns, followups = make_prompts(
            args.sessions,
            args.shared_prefix_tokens,
            args.session_tokens,
            args.followup_tokens,
            args.seed,
        )
        trials = []
        for trial_index in range(args.trials):
            await reset_cache(session, args.base_url)
            await asyncio.sleep(args.reset_settle_s)
            metrics_before_prime = await read_prompt_sources(session, args.base_url)
            prime = await run_request(
                session,
                args.base_url,
                model,
                shared,
                -1,
                "shared_prefix_prime",
                0,
                0,
                args.output_tokens,
            )
            await asyncio.sleep(args.phase_settle_ms / 1000)
            metrics_after_prime = await read_prompt_sources(session, args.base_url)

            first_results, first_makespan = await run_batch(
                session,
                args,
                model,
                first_turns,
                "first_turn",
                0,
                [0] * args.sessions,
            )
            await asyncio.sleep(args.phase_settle_ms / 1000)
            metrics_after_first = await read_prompt_sources(session, args.base_url)
            followup_results, followup_makespan = await run_batch(
                session,
                args,
                model,
                followups,
                "followup",
                1,
                [len(prompt) for prompt in first_turns],
            )
            metrics_after_followup = await read_prompt_sources(session, args.base_url)
            first_summary = summarize(first_results, first_makespan)
            followup_summary = summarize(followup_results, followup_makespan)
            prime_server_metrics: dict[str, Any] = {}
            add_server_metrics(
                prime_server_metrics, metrics_before_prime, metrics_after_prime
            )
            add_server_metrics(first_summary, metrics_after_prime, metrics_after_first)
            add_server_metrics(
                followup_summary, metrics_after_first, metrics_after_followup
            )
            trials.append(
                {
                    "trial": trial_index + 1,
                    "prime": asdict(prime),
                    "prime_server_metrics": prime_server_metrics,
                    "first_turn": first_summary,
                    "followup": followup_summary,
                    "first_turn_requests": [asdict(result) for result in first_results],
                    "followup_requests": [
                        asdict(result) for result in followup_results
                    ],
                }
            )
            print(
                f"trial={trial_index + 1} "
                f"first_p50={first_summary['ttft_p50_ms']:.1f}ms "
                f"first_rate={first_summary['requests_per_second']:.2f}req/s "
                f"followup_cached="
                f"{followup_summary['server_local_cache_hit_rate']:.1%} "
                f"followup_p50={followup_summary['ttft_p50_ms']:.1f}ms "
                f"followup_rate={followup_summary['requests_per_second']:.2f}req/s",
                flush=True,
            )

    numeric_keys = (
        "api_reported_cache_request_hit_rate",
        "api_reported_cached_token_rate",
        "server_local_cache_hit_tokens",
        "server_local_compute_tokens",
        "server_external_kv_tokens",
        "server_local_cache_hit_rate",
        "server_reused_token_rate",
        "ttft_mean_ms",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "e2e_mean_ms",
        "e2e_p50_ms",
        "e2e_p95_ms",
        "batch_makespan_ms",
        "requests_per_second",
    )
    aggregate = {
        phase: {
            key: {
                "median": statistics.median(trial[phase][key] for trial in trials),
                "min": min(trial[phase][key] for trial in trials),
                "max": max(trial[phase][key] for trial in trials),
            }
            for key in numeric_keys
        }
        for phase in ("first_turn", "followup")
    }
    return {
        "configuration": {
            "base_url": args.base_url,
            "model": model,
            "policy_label": args.policy_label,
            "sessions": args.sessions,
            "shared_prefix_tokens": args.shared_prefix_tokens,
            "session_tokens": args.session_tokens,
            "followup_tokens": args.followup_tokens,
            "output_tokens": args.output_tokens,
            "trials": args.trials,
            "reset_settle_s": args.reset_settle_s,
            "phase_settle_ms": args.phase_settle_ms,
            "launch_gap_ms": args.launch_gap_ms,
            "seed": args.seed,
        },
        "aggregate": aggregate,
        "trials": trials,
    }


def main() -> None:
    args = parse_args()
    if args.sessions <= 0 or args.trials <= 0:
        raise ValueError("sessions and trials must be positive")
    payload = asyncio.run(main_async(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
