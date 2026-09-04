#!/usr/bin/env python3
"""Measure frontend CPU overhead of Agentrix data-parallel routing policies."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

from vllm.v1.engine.prefix_router import PrefixAwareDPRouter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=int, default=5000)
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--prefix-tokens", type=int, default=3072)
    parser.add_argument("--followup-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def request(
    request_id: str,
    token_ids: list[int],
    session_id: str,
    turn: int,
    history_tokens: int | None = None,
):
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=token_ids,
        prompt_embeds=None,
        prompt_is_token_ids=None,
        lora_request=None,
        mm_features=None,
        cache_salt=None,
        sampling_params=SimpleNamespace(
            max_tokens=1,
            extra_args={
                "agentrix_session_id": session_id,
                "agentrix_turn": turn,
                "agentrix_history_tokens": (
                    len(token_ids) if history_tokens is None else history_tokens
                ),
            },
        ),
        pooling_params=None,
        data_parallel_rank=None,
        resumable=False,
    )


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def make_prompts(args: argparse.Namespace) -> list[list[int]]:
    prompts = []
    for session_id in range(args.sessions):
        rng = random.Random(args.seed + session_id)
        prompt = [rng.randrange(1000, 30000) for _ in range(args.prefix_tokens)]
        prompt.extend(rng.randrange(1000, 30000) for _ in range(args.followup_tokens))
        prompts.append(prompt)
    return prompts


def profile_policy(
    policy: str,
    prompts: list[list[int]],
    decisions: int,
    followup_tokens: int,
) -> dict[str, float | int | str]:
    router = PrefixAwareDPRouter(
        num_ranks=2,
        block_size=16,
        load_slack=32,
        warm_ttl_s=30,
        min_prefix_blocks=4,
        routing_policy=policy,
    )
    for session_id, prompt in enumerate(prompts):
        initial = request(
            f"initial-{session_id}",
            prompt[: len(prompt) - followup_tokens],
            f"session-{session_id}",
            0,
        )
        router.add_request(initial, rank=session_id % 2)
        router.observe_outputs(
            [
                SimpleNamespace(
                    request_id=initial.request_id, new_token_ids=[], events=None
                )
            ],
            {initial.request_id},
        )

    samples_us: list[float] = []
    for decision in range(decisions):
        session_id = decision % len(prompts)
        followup = request(
            f"decision-{decision}",
            prompts[session_id],
            f"session-{session_id}",
            1,
            len(prompts[session_id]) - followup_tokens,
        )
        started_ns = time.perf_counter_ns()
        router.choose_rank(
            followup,
            [[decision % 2, 0], [(decision + 1) % 2, 0]],
            baseline_rank=(decision + 1) % 2,
            start_index=decision % 2,
        )
        samples_us.append((time.perf_counter_ns() - started_ns) / 1000)
        router._pending_prefixes.pop(followup.request_id, None)
        router._pending_work.pop(followup.request_id, None)

    return {
        "policy": policy,
        "decisions": decisions,
        "mean_us": statistics.fmean(samples_us),
        "p50_us": percentile(samples_us, 0.5),
        "p95_us": percentile(samples_us, 0.95),
        "p99_us": percentile(samples_us, 0.99),
        "router_internal_mean_us": router.average_route_us,
    }


def main() -> None:
    args = parse_args()
    if args.decisions <= 0 or args.sessions <= 0:
        raise ValueError("decisions and sessions must be positive")
    if args.prefix_tokens <= 0 or args.followup_tokens < 0:
        raise ValueError("prefix tokens must be positive and follow-up non-negative")
    prompts = make_prompts(args)
    result = {
        "configuration": {
            "decisions": args.decisions,
            "sessions": args.sessions,
            "prefix_tokens": args.prefix_tokens,
            "followup_tokens": args.followup_tokens,
            "seed": args.seed,
        },
        "policies": [
            profile_policy(policy, prompts, args.decisions, args.followup_tokens)
            for policy in ("prefix_aware", "session_aware")
        ],
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
