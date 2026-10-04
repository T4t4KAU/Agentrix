"""Capture full generated tokens and top logprobs on the ForkAttention fanout.

Concurrent HTTP and one batched HTTP request per rank isolate arrival effects.
Instrumentation changes cost: this diagnostic is not a performance benchmark.
All output belongs on the experiment server.
"""

import argparse
import asyncio
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from benchmark_fork_scale import run
from benchmark_prefix_aware_dp import RequestResult


@dataclass(slots=True)
class NumericResult(RequestResult):
    token_ids: list[int]
    token_logprobs: list[float]
    logprob_tokens: list[str]
    top_logprobs: list[dict[str, float]]


def validate_logprobs(ids, tokens, logprobs, tops):
    if not (len(ids) == len(tokens) == len(logprobs) == len(tops)):
        raise RuntimeError("Token/logprob stream lengths differ")
    for token, value, top in zip(tokens, logprobs, tops):
        if not isinstance(token, str) or not top or token not in top:
            raise RuntimeError("Missing chosen token in top logprobs")
        if any(not math.isfinite(v) for v in [value, *top.values()]):
            raise RuntimeError("Nonfinite logprob")
        if abs(top[token] - value) > 1e-5 or value > 1e-5:
            raise RuntimeError("Inconsistent chosen logprob")


async def capture_requests(
    session, base_url, model, prompts, document, output_tokens, *, batched
):
    payload = {
        "model": model,
        "prompt": prompts if batched else prompts[0],
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
        "logprobs": 5,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    outputs = [
        {
            "ids": [],
            "tokens": [],
            "logprobs": [],
            "tops": [],
            "first": None,
            "ended": None,
        }
        for _ in prompts
    ]
    usage = None
    started = time.perf_counter()
    async with session.post(
        base_url + "/v1/completions",
        json=payload,
        headers={"X-Session-ID": f"document-{document}"},
    ) as response:
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}: {await response.text()}")
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            for choice in event.get("choices", []):
                index = choice["index"]
                if type(index) is not int or not 0 <= index < len(outputs):
                    raise RuntimeError("Unexpected batched choice index")
                out = outputs[index]
                ids = choice.get("token_ids") or []
                if ids:
                    out["first"] = out["first"] or time.perf_counter()
                    lp = choice.get("logprobs")
                    if not lp:
                        raise RuntimeError("Generated token missing logprobs")
                    out["ids"].extend(ids)
                    out["tokens"].extend(lp["tokens"])
                    out["logprobs"].extend(lp["token_logprobs"])
                    out["tops"].extend(lp["top_logprobs"])
                if choice.get("finish_reason"):
                    out["ended"] = time.perf_counter()
            if event.get("usage"):
                usage = event["usage"]
    ended = time.perf_counter()
    if (
        not usage
        or usage["prompt_tokens"] != sum(map(len, prompts))
        or usage["completion_tokens"] != len(prompts) * output_tokens
    ):
        raise RuntimeError("Unexpected aggregate token usage")
    rows = []
    for prompt, out in zip(prompts, outputs):
        if (
            len(out["ids"]) != output_tokens
            or out["first"] is None
            or out["ended"] is None
        ):
            raise RuntimeError("Incomplete generated sequence")
        validate_logprobs(out["ids"], out["tokens"], out["logprobs"], out["tops"])
        rows.append(
            NumericResult(
                document=document,
                ttft_ms=(out["first"] - started) * 1000,
                e2e_ms=(ended - started) * 1000,
                prompt_tokens=len(prompt),
                # Legacy batched usage exposes only the last prompt's cache count.
                # Per-rank prompt-source counters in the shared runner are authoritative.
                cached_tokens=0,
                completion_tokens=output_tokens,
                output_token_sha256=hashlib.sha256(
                    json.dumps(out["ids"]).encode()
                ).hexdigest(),
                token_ids=out["ids"],
                token_logprobs=out["logprobs"],
                logprob_tokens=out["tokens"],
                top_logprobs=out["tops"],
            )
        )
    return rows


async def capture_single(
    session, base_url, model, prompt, document, output_tokens, **_
):
    return (
        await capture_requests(
            session, base_url, model, [prompt], document, output_tokens, batched=False
        )
    )[0]


async def capture_batch(session, base_url, model, prompts, document, output_tokens):
    return await capture_requests(
        session, base_url, model, prompts, document, output_tokens, batched=True
    )


def compare_outputs(a, b):
    """Compare probabilities only while both runs have the same token history."""
    for row in (a, b):
        validate_logprobs(
            row["token_ids"],
            row["logprob_tokens"],
            row["token_logprobs"],
            row["top_logprobs"],
        )
    if len(a["token_ids"]) != len(b["token_ids"]):
        raise ValueError("Output lengths differ")
    first = next(
        (i for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"])) if x != y),
        None,
    )
    stop = len(a["token_ids"]) if first is None else first + 1
    differences = []
    for i in range(stop):
        x, y = a["top_logprobs"][i], b["top_logprobs"][i]
        differences.extend(abs(x[token] - y[token]) for token in x.keys() & y.keys())
    result = {
        "tokens_equal": first is None,
        "first_difference_index": first,
        "same_history_steps": stop,
        "common_topk_values": len(differences),
        "max_common_logprob_difference": max(differences, default=None),
        "mean_common_logprob_difference": sum(differences) / len(differences)
        if differences
        else None,
    }
    if first is not None:
        result["first_difference"] = {}
        for name, row in (("baseline", a), ("candidate", b)):
            top = row["top_logprobs"][first]
            values = sorted(top.values(), reverse=True)
            result["first_difference"][name] = {
                "token_id": row["token_ids"][first],
                "chosen_logprob": row["token_logprobs"][first],
                "top1_top2_gap": values[0] - values[1] if len(values) > 1 else None,
                "top_logprobs": top,
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-url", "control-url", "model"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument(
        "--prefix-tokens", type=int, nargs="+", default=[16384, 32768, 65536]
    )
    parser.add_argument("--branches", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument(
        "--request-mode", choices=("concurrent", "batched"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.trials < 1
        or args.output_tokens < 2
        or any(p < 1024 or p % 1024 for p in args.prefix_tokens)
        or any(b < 2 or b > 8 for b in args.branches)
    ):
        parser.error(
            "require positive trials, >=2 outputs, aligned prefixes and 2..8 branches"
        )
    args.scope = "Full token/logprob diagnostic; instrumented latency is not a performance result"
    asyncio.run(
        run(
            args,
            request_fn=capture_single,
            batch_request_fn=capture_batch if args.request_mode == "batched" else None,
        )
    )


if __name__ == "__main__":
    main()
