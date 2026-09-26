"""Compare model decode with shared, unrelated and padded request batches.

Run against native and ForkAttention services in separate lifetimes. This is
a correctness/short latency diagnostic, not an official AgentX score. Store
the JSON and logs on the experiment server.
"""

import argparse
import json
import math
from pathlib import Path
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

from hybrid_cache_smoke import post


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--model-path", default="/data/models/Qwen3.5-9B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--prefix-tokens", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--logprob-atol", type=float, default=0.03)
    parser.add_argument(
        "--batch-request",
        action="store_true",
        help="Submit each case in one HTTP request with group-specific prefixes",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if not math.isfinite(args.logprob_atol) or args.logprob_atol <= 0:
        parser.error("logprob-atol must be finite and positive")
    if args.prefix_tokens <= 0 or args.max_tokens <= 0:
        parser.error("prefix-tokens and max-tokens must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference:
        assert reference["passed"] and reference["prefix_tokens"] == args.prefix_tokens
        assert reference["max_tokens"] == args.max_tokens
        assert reference.get("batch_request", False) == args.batch_request
    start = tokenizer.encode("<|im_start|>user\n")
    filler = tokenizer.encode(
        "These notes provide context for a simple counting exercise.\n"
    )
    prefix = (start + filler * (args.prefix_tokens // len(filler) + 1))[
        : args.prefix_tokens
    ]

    def prompt(index, group=0):
        context = prefix.copy()
        if args.batch_request:
            context[len(start)] = tokenizer.encode(str(group))[0]
        return context + tokenizer.encode(
            f"\nPrint the integers from {index + 1} through {index + 100}, separated by commas. "
            "Output only the sequence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
        )

    def generate(rank, index, salt, tokens=None, group=0):
        response = post(
            args.url,
            "/v1/completions",
            {
                "model": args.model,
                "prompt": prompt(index, group),
                "temperature": 0,
                "max_tokens": tokens or args.max_tokens,
                "ignore_eos": True,
                "seed": 42,
                "logprobs": 1,
                "cache_salt": salt,
            },
            rank,
        )
        choice = response["choices"][0]
        return {
            "tokens": choice["logprobs"]["tokens"],
            "logprobs": choice["logprobs"]["token_logprobs"],
            "finish_reason": choice["finish_reason"],
            "cached_tokens": (response["usage"].get("prompt_tokens_details") or {}).get(
                "cached_tokens", 0
            ),
        }

    result = {
        "kind": "fork_model_correctness",
        "prefix_tokens": args.prefix_tokens,
        "max_tokens": args.max_tokens,
        "ranks": [],
        "logprob_atol": args.logprob_atol,
        "batch_request": args.batch_request,
        "failures": [],
        "passed": False,
    }
    # The 3/7 query cases exercise padding into 4/8-query model graphs. Distinct
    # salts force separate physical cache pages despite identical prompt text.
    cases = {
        "shared-2": [0, 0],
        "shared-3": [0] * 3,
        "shared-8": [0] * 8,
        "mixed-7": [0, 1, 0, 1, 2, 0, 1],
        "unrelated-4": [0, 1, 2, 3],
        "single": [0],
    }
    for rank in (0, 1):
        rank_result = {"rank": rank, "cases": {}}
        result["ranks"].append(rank_result)
        for name, groups in cases.items():
            salts = {group: uuid.uuid4().hex for group in set(groups)}
            if args.batch_request:
                salt = uuid.uuid4().hex
                salts = dict.fromkeys(salts, salt)
            for group, salt in salts.items():
                generate(rank, 0, salt, tokens=1, group=group)
            started = time.monotonic()
            if args.batch_request:
                response = post(
                    args.url,
                    "/v1/completions",
                    {
                        "model": args.model,
                        "prompt": [prompt(i, group) for i, group in enumerate(groups)],
                        "temperature": 0,
                        "max_tokens": args.max_tokens,
                        "ignore_eos": True,
                        "seed": 42,
                        "logprobs": 1,
                        "cache_salt": salts[0],
                    },
                    rank,
                )
                cached = (response["usage"].get("prompt_tokens_details") or {}).get(
                    "cached_tokens", 0
                )
                outputs = [
                    {
                        "tokens": c["logprobs"]["tokens"],
                        "logprobs": c["logprobs"]["token_logprobs"],
                        "finish_reason": c["finish_reason"],
                        # vLLM 0.22.1 reports the final prompt's cache count in
                        # a batched completion; this is not a per-choice count.
                        "cached_tokens": None,
                        "batch_reported_cached_tokens": cached,
                    }
                    for c in sorted(
                        response["choices"], key=lambda choice: choice["index"]
                    )
                ]
            else:
                with ThreadPoolExecutor(max_workers=len(groups)) as executor:
                    outputs = list(
                        executor.map(
                            lambda pair: generate(rank, pair[0], salts[pair[1]]),
                            enumerate(groups),
                        )
                    )
            elapsed = time.monotonic() - started
            if args.batch_request:
                assert cached >= args.prefix_tokens, (rank, name, cached)
            else:
                assert all(
                    output["cached_tokens"] >= args.prefix_tokens for output in outputs
                ), (rank, name, outputs)
            assert all(len(output["tokens"]) == args.max_tokens for output in outputs)
            case = {"outputs": outputs, "elapsed_seconds": elapsed}
            rank_result["cases"][name] = case
            if reference:
                expected = reference["ranks"][rank]["cases"][name]["outputs"]
                comparisons = []
                for index, (actual, baseline) in enumerate(
                    zip(outputs, expected, strict=True)
                ):
                    differences = [
                        abs(a - b)
                        for a, b in zip(
                            actual["logprobs"], baseline["logprobs"], strict=True
                        )
                    ]
                    comparison = {
                        "tokens_match": actual["tokens"] == baseline["tokens"],
                        "finish_reason_match": actual["finish_reason"]
                        == baseline["finish_reason"],
                        "max_logprob_difference": max(differences),
                        "logprobs_match": all(
                            math.isfinite(d) and d <= result["logprob_atol"]
                            for d in differences
                        ),
                    }
                    comparisons.append(comparison)
                    if not all(
                        comparison[key]
                        for key in (
                            "tokens_match",
                            "finish_reason_match",
                            "logprobs_match",
                        )
                    ):
                        result["failures"].append(
                            {"rank": rank, "case": name, "request": index, **comparison}
                        )
                case["comparisons"] = comparisons
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                f"rank={rank} case={name} requests={len(groups)} elapsed={elapsed:.3f}s",
                flush=True,
            )
    result["passed"] = not result["failures"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if result["failures"]:
        raise AssertionError(
            f"{len(result['failures'])} mismatches; details in {args.output}"
        )
    print("FORK_MODEL_CHECK_PASSED", flush=True)


if __name__ == "__main__":
    main()
