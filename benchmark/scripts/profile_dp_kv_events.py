#!/usr/bin/env python3
"""Compare DP decision cost with logical and GPU-event prefix hints."""

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import msgspec
from vllm.distributed.kv_events import BlockStored
from vllm.v1.engine.prefix_router import PrefixAwareDPRouter


def request(index: int, request_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=[1000 + index] * 1536,
        prompt_embeds=None,
        prompt_is_token_ids=None,
        lora_request=None,
        mm_features=None,
        cache_salt=None,
        sampling_params=SimpleNamespace(
            max_tokens=1,
            extra_args={
                "agentrix_turn": 1,
                "agentrix_session_id": f"session-{index}",
            },
        ),
    )


def run(use_events: bool, rounds: int) -> dict:
    router = PrefixAwareDPRouter(
        2, 16, 4, 300, 1, routing_policy="session_aware", use_kv_events=use_events
    )
    for index in range(12):
        req = request(index, f"warm-{index}")
        router.add_request(req, index % 2)
        router.observe_outputs(
            [SimpleNamespace(request_id=req.request_id, new_token_ids=[], events=None)],
            {req.request_id},
        )
        event = BlockStored(
            block_hashes=list(range(index * 1000, index * 1000 + 96)),
            parent_block_hash=None,
            token_ids=req.prompt_token_ids,
            block_size=16,
            lora_id=None,
            lora_name=None,
            medium="GPU",
        )
        router.observe_cache_events(index % 2, msgspec.msgpack.encode([event]))
    samples = []
    for index in range(rounds):
        req = request(index % 12, f"probe-{index}")
        started = time.perf_counter_ns()
        rank = router.choose_rank(req, [[0, 0], [0, 0]], baseline_rank=1)
        samples.append((time.perf_counter_ns() - started) / 1000)
        router.add_request(req, rank)
        router.observe_outputs(
            [SimpleNamespace(request_id=req.request_id, new_token_ids=[], events=None)],
            {req.request_id},
        )
    samples.sort()
    return {
        "p50_us": statistics.median(samples),
        "p99_us": samples[min(len(samples) - 1, int(len(samples) * 0.99))],
        "mean_us": statistics.mean(samples),
        "event_routed": router.cache_event_route_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error("rounds must be positive")
    result = {
        "rounds": args.rounds,
        "logical": run(False, args.rounds),
        "physical": run(True, args.rounds),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
