"""Compare cold and restored hybrid states on both ranks; not an AgentX score."""

import argparse
import json
import math
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer


def post(url, path, body, rank=0):
    request = urllib.request.Request(
        url + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-data-parallel-rank": str(rank)},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        data = response.read()
    return json.loads(data) if data else None


def generate(args, tokens, rank, salt):
    started = time.monotonic()
    response = post(
        args.url,
        "/v1/completions",
        {
            "model": args.model,
            "prompt": tokens,
            "temperature": 0,
            "max_tokens": 16,
            "seed": 42,
            "logprobs": 5,
            "cache_salt": salt,
        },
        rank,
    )
    choice = response["choices"][0]
    details = response["usage"].get("prompt_tokens_details") or {}
    return {
        "text": choice["text"],
        "finish_reason": choice["finish_reason"],
        "logprobs": choice["logprobs"]["token_logprobs"],
        "tokens": choice["logprobs"]["tokens"],
        "cached_tokens": details.get("cached_tokens", 0),
        "prompt_tokens": response["usage"]["prompt_tokens"],
        "elapsed_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--model-path", default="/data/models/Qwen3.5-9B")
    parser.add_argument("--expect-sparse", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--reference-file",
        help="Compare against an earlier execution mode's cold results",
    )
    parser.add_argument(
        "--concurrent-siblings", type=int, choices=range(1, 9), default=2
    )
    args = parser.parse_args()
    reference_results = None
    if args.reference_file:
        with open(args.reference_file) as source:
            reference_results = json.load(source)
        assert reference_results["passed"]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    prefix = tokenizer.encode(
        "<|im_start|>user\n"
        + "This is reference text for a cache consistency check.\n" * 2000
    )

    def suffix(number):
        return tokenizer.encode(
            f"\nReply with only the number {number}.\n<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n"
        )

    parent = prefix[: 12288 - len(suffix(2))] + suffix(2)
    fork = parent[:6144] + suffix(3)
    sibling = parent[:6144] + suffix(4)
    continuation = parent + tokenizer.encode(
        "2<|im_end|>\n<|im_start|>user\n"
        "Reply with only the number 5.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n"
    )
    prompts = {
        "parent": parent,
        "fork": fork,
        "sibling": sibling,
        "continuation": continuation,
    }
    results = {"kind": "hybrid_state_correctness", "ranks": [], "logprob_atol": 0.02}
    for rank in (0, 1):
        cold = {
            name: generate(args, tokens, rank, uuid.uuid4().hex)
            for name, tokens in prompts.items()
        }
        if reference_results is not None:
            reference_rank = next(
                r for r in reference_results["ranks"] if r["rank"] == rank
            )
            for name, actual in cold.items():
                expected = reference_rank["cold"][name]
                assert actual["tokens"] == expected["tokens"], (
                    rank,
                    name,
                    actual,
                    expected,
                )
                assert actual["finish_reason"] == expected["finish_reason"]
                assert all(
                    math.isclose(a, b, rel_tol=0, abs_tol=results["logprob_atol"])
                    for a, b in zip(
                        actual["logprobs"], expected["logprobs"], strict=True
                    )
                ), (rank, name, actual, expected)
        salt = uuid.uuid4().hex
        warm = {
            name: generate(args, tokens, rank, salt) for name, tokens in prompts.items()
        }
        warm["replay"] = generate(args, parent, rank, salt)
        # Concurrent siblings must each get a private running state, preserving
        # the shared checkpoint even when one starts decoding first.
        with ThreadPoolExecutor(max_workers=args.concurrent_siblings) as executor:
            concurrent = list(
                executor.map(
                    lambda _: generate(args, sibling, rank, salt),
                    range(args.concurrent_siblings),
                )
            )
        warm.update({f"concurrent_{i}": result for i, result in enumerate(concurrent)})
        for name, restored in warm.items():
            source = (
                "parent"
                if name == "replay"
                else "sibling"
                if name.startswith("concurrent_")
                else name
            )
            reference = cold[source]
            expected_text = {
                "parent": "2",
                "fork": "3",
                "sibling": "4",
                "continuation": "5",
            }[source]
            assert reference["text"].strip() == expected_text, (rank, source, reference)
            assert restored["tokens"] == reference["tokens"], (
                rank,
                name,
                restored,
                reference,
            )
            assert restored["finish_reason"] == reference["finish_reason"]
            assert all(
                math.isclose(
                    actual, expected, rel_tol=0, abs_tol=results["logprob_atol"]
                )
                for actual, expected in zip(
                    restored["logprobs"], reference["logprobs"], strict=True
                )
            ), (rank, name, restored["logprobs"], reference["logprobs"])
        assert all(result["cached_tokens"] == 0 for result in cold.values())
        assert warm["sibling"]["cached_tokens"] >= 6144
        assert warm["continuation"]["cached_tokens"] >= 12288
        assert warm["replay"]["cached_tokens"] == 11264
        if args.expect_sparse:
            assert warm["fork"]["cached_tokens"] < 6144
        results["ranks"].append({"rank": rank, "cold": cold, "restored": warm})
        print(
            f"rank {rank}: cold/restored tokens and logprobs agree; fork checkpoint reused",
            flush=True,
        )
    results["passed"] = True
    with open(args.output, "w") as output:
        json.dump(results, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
