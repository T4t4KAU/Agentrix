#!/usr/bin/env python3
"""Check mixed KV partial-tail caps against a dedicated offload backend.

Use the server's actual physical block size and finer prefix-match unit.
This resets both caches and writes diagnostic output on the experiment server.
It is a correctness probe, not an AgentX or performance benchmark.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmark_agent_kv_tiering import Backend, delta, make_prompt, save_result


def validate_case(case: dict, cold: dict, target_tokens: int) -> None:
    counters = case["resume_counters"]
    expected = case["expected_external_tokens"]
    if case["resume"]["token_ids"] != cold["token_ids"]:
        raise RuntimeError("partial-tail restore changed generated tokens")
    if counters["tokens_local_cache_hit"] != 0:
        raise RuntimeError("device reset left a local cache hit")
    if (
        counters["tokens_external_kv_transfer"] != expected
        or counters["tokens_local_compute"] != target_tokens - expected
    ):
        raise RuntimeError("unexpected partial-tail restore boundary")
    if counters.get("store_bytes", 0) != 0:
        raise RuntimeError("zero-cap resume created a new backup")
    stored = case["store_counters"].get("store_bytes", 0)
    loaded = counters.get("load_bytes", 0)
    if expected == 0:
        if stored != 0 or loaded != 0:
            raise RuntimeError("excluded boundary was stored or loaded")
    elif stored <= 0 or loaded <= 0:
        raise RuntimeError("restore did not exercise both transfer directions")


def run(args) -> dict:
    from transformers import AutoTokenizer

    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    parent = make_prompt(tokenizer, args.seed, 0, args.prompt_tokens)
    suffix = tokenizer.encode("\n", add_special_tokens=False)[:1]
    if not suffix:
        raise ValueError("tokenizer did not encode the continuation")
    target = parent + suffix
    complete_boundary = args.prompt_tokens // args.block_tokens * args.block_tokens
    caps = [
        None,
        0,
        args.prefix_match_unit,
        complete_boundary,
        args.prompt_tokens - 1,
        args.prompt_tokens,
    ]
    backend = Backend(args.base_url, args.model, args.output_tokens)
    result = {
        "scope": "Mixed KV boundary correctness; no performance claim",
        "configuration": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "parent_token_ids": parent,
        "target_token_ids": target,
        "valid": False,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        save_result(args.output, result)

    try:
        backend.reset(external=True)
        time.sleep(2)
        before = backend.metrics()
        cold = backend.infer(target, max_offload_tokens=0)
        after = backend.settled_metrics(before, len(target), 2)
        result["cold"] = cold
        result["cold_counters"] = delta(before, after)
        cold_counters = result["cold_counters"]
        if (
            cold_counters.get("store_bytes", 0) != 0
            or cold_counters.get("load_bytes", 0) != 0
            or cold_counters["tokens_local_compute"] != len(target)
            or cold_counters["tokens_local_cache_hit"] != 0
            or cold_counters["tokens_external_kv_transfer"] != 0
        ):
            raise RuntimeError("zero-cap cold reference used or created a cache")
        save()
        for cap in caps:
            expected = (
                args.prompt_tokens
                if cap is None or cap >= args.prompt_tokens
                else cap // args.block_tokens * args.block_tokens
            )
            backend.reset(external=True)
            time.sleep(2)
            before = backend.metrics()
            case = {"max_offload_tokens": cap, "expected_external_tokens": expected}
            result["cases"].append(case)
            case["parent"] = backend.infer(parent, max_offload_tokens=cap)
            save()
            backend.reset(external=False)
            stored = backend.settled_metrics(before, len(parent), 2)
            case["store_counters"] = delta(before, stored)
            case["resume"] = backend.infer(target, max_offload_tokens=0)
            loaded = backend.settled_metrics(stored, len(target), 2)
            case["resume_counters"] = delta(stored, loaded)
            validate_case(case, cold, len(target))
            case["valid"] = True
            save()
        result["valid"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-tokens", type=int, required=True)
    parser.add_argument("--prefix-match-unit", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    if min(args.block_tokens, args.prefix_match_unit, args.prompt_tokens) < 1:
        parser.error("block, prefix unit and prompt lengths must be positive")
    if (
        args.prefix_match_unit >= args.block_tokens
        or args.block_tokens % args.prefix_match_unit
        or args.prompt_tokens <= args.block_tokens
        or args.prompt_tokens % args.prefix_match_unit
        or args.prompt_tokens % args.block_tokens == 0
        or args.output_tokens < 2
    ):
        parser.error(
            "require a finer prefix unit dividing the physical block, a prompt "
            "beyond one block ending on a partial hash boundary, and >=2 outputs"
        )
    result = run(args)
    print(json.dumps({"valid": result["valid"], "cases": len(result["cases"])}))


if __name__ == "__main__":
    main()
