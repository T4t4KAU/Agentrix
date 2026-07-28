"""Run a deterministic LongBench QA workload against an OpenAI-compatible API."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from longbench_qa import load_cases, score_answer


async def ask(
    client: AsyncOpenAI,
    model: str,
    case: dict[str, Any],
    question: dict[str, Any],
    semaphore: asyncio.Semaphore,
    max_tokens: int,
) -> dict[str, Any]:
    prompt = (
        "Answer the question using only the document below. Give only the "
        "shortest answer that fully answers the question; do not explain."
        "\n\nDOCUMENT:\n"
        + case["context"]
    )
    started = time.perf_counter()
    first_token = None
    chunks = []
    usage = None
    async with semaphore:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question["question"]},
            ],
            temperature=0,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        async for event in stream:
            content = (
                event.choices[0].delta.content if event.choices else None
            )
            if content:
                first_token = first_token or time.perf_counter()
                chunks.append(content)
            if event.usage is not None:
                usage = event.usage.model_dump()

    ended = time.perf_counter()
    prediction = "".join(chunks).strip()
    return {
        "case_id": case["case_id"],
        "dataset": case["dataset"],
        "context_sha256": case["context_sha256"],
        "context_tokens": case["context_tokens"],
        "source_id": question["source_id"],
        "question": question["question"],
        "answers": question["answers"],
        "prediction": prediction,
        **score_answer(prediction, question["answers"]),
        "ttft_seconds": (first_token or ended) - started,
        "latency_seconds": ended - started,
        "usage": usage,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(args.cases)
    client = AsyncOpenAI(
        base_url=args.base_url, api_key="agentrix", timeout=args.timeout
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    results = await asyncio.gather(
        *(
            ask(
                client,
                args.model,
                case,
                question,
                semaphore,
                args.max_tokens,
            )
            for case in cases
            for question in case["questions"]
        )
    )
    wall = time.perf_counter() - started
    manifest_bytes = args.cases.read_bytes()
    return {
        "schema_version": 1,
        "model": args.model,
        "case_manifest": args.cases.name,
        "case_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "case_count": len(cases),
        "question_count": len(results),
        "wall_seconds": wall,
        "questions_per_second": len(results) / wall,
        "mean_exact_match": statistics.fmean(
            result["exact_match"] for result in results
        ),
        "mean_f1": statistics.fmean(result["f1"] for result in results),
        "mean_ttft_seconds": statistics.fmean(
            result["ttft_seconds"] for result in results
        ),
        "p95_ttft_seconds": sorted(
            result["ttft_seconds"] for result in results
        )[max(0, int(len(results) * 0.95) - 1)],
        "mean_latency_seconds": statistics.fmean(
            result["latency_seconds"] for result in results
        ),
        "prompt_tokens": sum(
            (result["usage"] or {}).get("prompt_tokens", 0)
            for result in results
        ),
        "completion_tokens": sum(
            (result["usage"] or {}).get("completion_tokens", 0)
            for result in results
        ),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:9000/v1"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "results"},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
