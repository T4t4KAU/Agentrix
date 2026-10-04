"""Growing, concurrent and branching KV lifecycle validation on a dedicated server.

Histories are fixed token plans, not model-generated conversations or AgentX.
Resets occur only before a complete trajectory, never between its rounds.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from benchmark_agent_kv_tiering import TOKEN_COUNTERS, delta, make_prompt, save_result
from benchmark_dp_kv_lifecycle import RoutedBackend, percentile


def make_plan(
    tokenizer,
    *,
    seed,
    sessions,
    branches,
    rounds,
    initial_tokens,
    growth_tokens,
    pressure_requests,
):
    if sessions % branches:
        raise ValueError("sessions must be divisible by branches")
    longest = initial_tokens + growth_tokens * (rounds - 1)
    if branches > 1 and initial_tokens < 4608:
        raise ValueError(
            "branch cases require room beyond their 4096-token shared prefix"
        )
    histories = []
    for index in range(sessions):
        if branches == 1:
            prompt = make_prompt(tokenizer, seed, index, longest)
        else:
            prompt = make_prompt(tokenizer, seed, index // branches, 4096)
            prompt += make_prompt(tokenizer, seed, 10000 + index, longest - 4096)
        histories.append(prompt)
    return [
        {
            "round": turn,
            "active": [p[: initial_tokens + turn * growth_tokens] for p in histories],
            "pressure": [
                make_prompt(
                    tokenizer, seed, 100000 + turn * pressure_requests + i, 8193
                )
                for i in range(pressure_requests)
            ],
        }
        for turn in range(rounds)
    ]


def execute_phase(args, prompts, identities, *, terminal):
    queued = time.perf_counter()

    def infer(item):
        index, prompt = item
        backend = RoutedBackend(
            args.base_url, args.control_url, args.model, args.output_tokens
        )
        backend.session_id = identities[index]
        started = time.perf_counter()
        try:
            row = backend.infer(
                prompt,
                max_offload_tokens=0 if terminal and args.mode == "selective" else None,
            )
            return {
                "index": index,
                "session_id": identities[index],
                "input_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                "prompt_tokens": len(prompt),
                "queue_ms": (started - queued) * 1000,
                "end_to_end_ttft_ms": (started - queued) * 1000 + row["ttft_ms"],
                **row,
            }
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            return {
                "index": index,
                "session_id": identities[index],
                "error": repr(error),
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(infer, enumerate(prompts)))
    return {"rows": rows, "wall_seconds": time.perf_counter() - queued}


def evaluate(result):
    failures = []
    resumes = []
    for turn in result["rounds"]:
        for phase in ("grow", "pressure", "resume"):
            rows = turn[phase]["rows"]
            if any("error" in row for row in rows):
                failures.append(f"round {turn['round']} {phase}: request failure")
        for cold, resumed in zip(
            turn["grow"]["rows"], turn["resume"]["rows"], strict=True
        ):
            if "error" not in cold and "error" not in resumed:
                if (
                    cold["input_sha256"] != resumed["input_sha256"]
                    or cold["token_ids"] != resumed["token_ids"]
                ):
                    failures.append(
                        f"round {turn['round']} session {cold['index']}: restored output differs"
                    )
                resumes.append(resumed)
    result["correctness_failures"] = failures
    result["correctness_passed"] = not failures
    counters = [
        turn[phase]["counters"]
        for turn in result["rounds"]
        for phase in ("grow", "pressure", "resume")
    ]
    result["summary"] = {
        "preemptions": sum(c.get("preemptions", 0) for c in counters),
        "resume_mean_ttft_ms": statistics.fmean(r["ttft_ms"] for r in resumes)
        if resumes
        else None,
        "resume_p95_ttft_ms": percentile([r["ttft_ms"] for r in resumes], 0.95)
        if resumes
        else None,
        "resume_mean_end_to_end_ttft_ms": statistics.fmean(
            r["end_to_end_ttft_ms"] for r in resumes
        )
        if resumes
        else None,
        "pressure_store_bytes": sum(
            t["pressure"]["counters"].get("store_bytes", 0) for t in result["rounds"]
        ),
        "resume_load_bytes": sum(
            t["resume"]["counters"].get("load_bytes", 0) for t in result["rounds"]
        ),
        "resume_compute_tokens": sum(
            t["resume"]["counters"].get("tokens_local_compute", 0)
            for t in result["rounds"]
        ),
        "resume_cache_tokens": sum(
            t["resume"]["counters"].get("tokens_local_cache_hit", 0)
            for t in result["rounds"]
        ),
        "resume_external_tokens": sum(
            t["resume"]["counters"].get("tokens_external_kv_transfer", 0)
            for t in result["rounds"]
        ),
    }
    if (
        result["configuration"]["mode"] == "selective"
        and result["summary"]["pressure_store_bytes"]
    ):
        result["correctness_failures"].append("Terminal requests wrote CPU KV")
        result["correctness_passed"] = False


def run(args):
    from transformers import AutoTokenizer

    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    plan = make_plan(
        tokenizer,
        **{
            key: getattr(args, key)
            for key in (
                "seed",
                "sessions",
                "branches",
                "rounds",
                "initial_tokens",
                "growth_tokens",
                "pressure_requests",
            )
        },
    )
    backend = RoutedBackend(
        args.base_url, args.control_url, args.model, args.output_tokens
    )
    result = {
        "complete": False,
        "configuration": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "plan_sha256": hashlib.sha256(json.dumps(plan).encode()).hexdigest(),
        "rounds": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    backend.reset(external=args.mode != "apc")
    previous = backend.metrics()
    result["started"] = time.time()
    try:
        for entry in plan:
            turn = {"round": entry["round"]}
            result["rounds"].append(turn)
            for phase in ("grow", "pressure", "resume"):
                pressure = phase == "pressure"
                prompts = entry["pressure"] if pressure else entry["active"]
                identities = [
                    f"kv-growth-{args.seed}-terminal-{entry['round']}-{i}"
                    if pressure
                    else f"kv-growth-{args.seed}-parent-{i // args.branches}"
                    for i in range(len(prompts))
                ]
                if phase == "resume":
                    time.sleep(args.tool_gap_seconds)
                turn[phase] = execute_phase(
                    args, prompts, identities, terminal=pressure
                )
                save_result(args.output, result)
                if any("error" in row for row in turn[phase]["rows"]):
                    raise RuntimeError(
                        "Request failure; preserve this trajectory as failed"
                    )
                after = backend.settled_metrics(
                    previous, sum(map(len, prompts)), args.settle_seconds
                )
                changes = delta(previous, after)
                assert sum(changes[key] for key in TOKEN_COUNTERS) == sum(
                    map(len, prompts)
                )
                turn[phase]["counters"] = changes
                previous = after
                save_result(args.output, result)
        evaluate(result)
        result["complete"] = True
    except Exception as error:
        result["error"] = repr(error)
        raise
    finally:
        result["ended"] = time.time()
        result["wall_seconds"] = result["ended"] - result["started"]
        save_result(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-url", "control-url", "model", "tokenizer"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("apc", "offload", "selective"), required=True
    )
    for name, default in (
        ("seed", 20261005),
        ("sessions", 8),
        ("branches", 1),
        ("rounds", 3),
        ("initial-tokens", 4097),
        ("growth-tokens", 2048),
        ("pressure-requests", 16),
        ("output-tokens", 8),
        ("concurrency", 8),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--tool-gap-seconds", type=float, default=0)
    parser.add_argument("--settle-seconds", type=float, default=1)
    args = parser.parse_args()
    if (
        min(
            args.sessions,
            args.branches,
            args.rounds,
            args.initial_tokens,
            args.growth_tokens,
            args.output_tokens,
            args.concurrency,
        )
        < 1
        or args.pressure_requests < 0
    ):
        parser.error("counts must be positive")
    if args.sessions % args.branches or any(
        not math.isfinite(v) or v < 0
        for v in (args.tool_gap_seconds, args.settle_seconds)
    ):
        parser.error("invalid branching or delay")
    result = run(args)
    print(
        json.dumps(
            {"correctness_passed": result["correctness_passed"], **result["summary"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
