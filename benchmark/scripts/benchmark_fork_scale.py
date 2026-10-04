"""Fixed-length fanout through official session routing on two DP ranks.

Synthetic token inputs isolate prefix/branch scaling; this is not QA or AgentX.
Use a dedicated backend: each shape resets device prefix caches before priming.
"""

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import aiohttp
from benchmark_agent_kv_tiering import delta, metrics, save_result
from benchmark_prefix_aware_dp import (
    read_rank_metrics,
    reset_cache,
    run_request,
    wait_for_completions,
)


def make_prompts(seed, prefix_tokens, branches):
    groups = []
    for group in range(2):
        rng = random.Random(seed + group)
        prefix = [rng.randrange(1000, 30000) for _ in range(prefix_tokens)]
        groups.append(
            {
                "prime": prefix + [100 + group] * 128,
                "branches": [
                    prefix + [500 + group * 16 + branch] * 128
                    for branch in range(branches)
                ],
            }
        )
    return groups


async def snapshot(session, url):
    async with session.get(url + "/metrics") as response:
        response.raise_for_status()
        body = await response.text()
    return {rank: metrics(body, engine=rank) for rank in ("0", "1")}


async def select_sessions(session, args):
    """Observe official hash placement; never inject a DP-rank header."""
    owners = {}
    for document in range(32):
        before = await read_rank_metrics(session, args.control_url)
        await run_request(
            session,
            args.base_url,
            args.model,
            [1000, 1001, 1002],
            document,
            1,
            allow_missing_prompt_details=True,
        )
        changes = await wait_for_completions(session, args.control_url, before, 1)
        touched = [rank for rank, values in changes.items() if values["requests"]]
        if len(touched) != 1:
            raise RuntimeError("a placement probe must reach exactly one rank")
        owners.setdefault(touched[0], document)
        if set(owners) == {"0", "1"}:
            return owners
    raise RuntimeError("official session hashes did not cover both ranks")


async def run(args, *, request_fn=run_request, batch_request_fn=None):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "valid": False,
        "scope": getattr(
            args,
            "scope",
            "Warm shared prefixes; synthetic fixed-length decode; no QA claim",
        ),
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "shapes": [],
    }
    save_result(args.output, result)
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=900),
            connector=aiohttp.TCPConnector(limit=0),
        ) as session:
            owners = await select_sessions(session, args)
            result["session_owners"] = owners
            shapes = [(p, b) for p in args.prefix_tokens for b in args.branches]
            random.Random(args.seed).shuffle(shapes)
            for prefix, branches in shapes:
                groups = make_prompts(args.seed, prefix, branches)
                shape = {
                    "prefix_tokens": prefix,
                    "branches_per_rank": branches,
                    "input_sha256": hashlib.sha256(
                        json.dumps(groups).encode()
                    ).hexdigest(),
                    "trials": [],
                }
                result["shapes"].append(shape)
                print(
                    json.dumps(
                        {"phase": "priming", "prefix": prefix, "branches": branches}
                    ),
                    flush=True,
                )
                await reset_cache(session, args.control_url)

                async def fanout(groups, output_tokens, prime=False):
                    calls = []
                    identities = []
                    for group, rank in enumerate(("0", "1")):
                        prompts = (
                            [groups[group]["prime"]]
                            if prime
                            else groups[group]["branches"]
                        )
                        if batch_request_fn is not None:
                            identities.extend(
                                (rank, branch) for branch in range(len(prompts))
                            )
                            calls.append(
                                batch_request_fn(
                                    session,
                                    args.base_url,
                                    args.model,
                                    prompts,
                                    owners[rank],
                                    output_tokens,
                                )
                            )
                            continue
                        for branch, prompt in enumerate(prompts):
                            identities.append((rank, branch))
                            calls.append(
                                request_fn(
                                    session,
                                    args.base_url,
                                    args.model,
                                    prompt,
                                    owners[rank],
                                    output_tokens,
                                    allow_missing_prompt_details=True,
                                )
                            )
                    started = time.perf_counter()
                    rows = await asyncio.gather(*calls)
                    elapsed = time.perf_counter() - started
                    if batch_request_fn is not None:
                        rows = [row for batch in rows for row in batch]
                    return [
                        {**asdict(row), "expected_rank": rank, "branch": branch}
                        for row, (rank, branch) in zip(rows, identities)
                    ], elapsed

                # Populate common pages first; then warm private suffixes/operators.
                before = await read_rank_metrics(session, args.control_url)
                await fanout(groups, 1, prime=True)
                await fanout(groups, args.output_tokens)
                await wait_for_completions(
                    session, args.control_url, before, 2 + 2 * branches
                )
                await asyncio.sleep(0.5)
                for trial in range(args.trials):
                    counts_before = await read_rank_metrics(session, args.control_url)
                    before = await snapshot(session, args.control_url)
                    started = time.time()
                    rows, elapsed = await fanout(groups, args.output_tokens)
                    ended = time.time()
                    counts = await wait_for_completions(
                        session, args.control_url, counts_before, 2 * branches
                    )
                    await asyncio.sleep(0.5)
                    after = await snapshot(session, args.control_url)
                    changes = {
                        rank: delta(before[rank], after[rank]) for rank in owners
                    }
                    for rank in owners:
                        if counts[rank]["requests"] != branches or changes[rank].get(
                            "preemptions", 0
                        ):
                            raise RuntimeError(
                                "rank imbalance or preemption in measured fanout"
                            )
                        c = changes[rank]
                        if sum(
                            c.get("tokens_" + source, 0)
                            for source in [
                                "local_compute",
                                "local_cache_hit",
                                "external_kv_transfer",
                            ]
                        ) != branches * (prefix + 128):
                            raise RuntimeError("incomplete prompt-source accounting")
                        if c.get("tokens_local_cache_hit", 0) < branches * prefix:
                            raise RuntimeError(
                                "shared prefix was not retained for every branch"
                            )
                    shape["trials"].append(
                        {
                            "trial": trial,
                            "started": started,
                            "ended": ended,
                            "wall_seconds": elapsed,
                            "rows": rows,
                            "rank_counters": changes,
                            "mean_ttft_ms": statistics.fmean(
                                row["ttft_ms"] for row in rows
                            ),
                            "mean_decode_response_ms": statistics.fmean(
                                row["e2e_ms"] - row["ttft_ms"] for row in rows
                            ),
                            "output_tokens": 2 * branches * args.output_tokens,
                        }
                    )
                    save_result(args.output, result)
                    print(
                        json.dumps(
                            {
                                "phase": "measured",
                                "prefix": prefix,
                                "branches": branches,
                                "trial": trial,
                                "wall_seconds": elapsed,
                            }
                        ),
                        flush=True,
                    )
            result["valid"] = True
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        save_result(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-url", "control-url", "model"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--prefix-tokens", type=int, nargs="+", default=[16384, 32768, 65536]
    )
    parser.add_argument("--branches", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials < 1 or args.output_tokens < 2:
        parser.error("positive trials and at least two output tokens required")
    if any(n < 1024 or n % 1024 for n in args.prefix_tokens):
        parser.error("prefix lengths must be positive multiples of 1024")
    if any(n < 2 or n > 8 for n in args.branches):
        parser.error("FIA serving supports 2 to 8 branches per rank")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
