from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

from longbench_qa import load_cases, score_answer
from openai import AsyncOpenAI


async def ask(
    client: AsyncOpenAI,
    model: str,
    case: dict[str, Any],
    question: dict[str, Any],
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    document_routing: bool = False,
    prefill_gate=None,
) -> dict[str, Any]:
    prompt = (
        "Answer the question using only the document below. Give only the "
        "shortest answer that fully answers the question; do not explain."
        "\n\nDOCUMENT:\n" + case["context"]
    )
    queued, first_token, chunks, usage = time.perf_counter(), None, [], None
    finish_reason = None
    gate_key = hashlib.sha256((model + "\0" + prompt).encode()).hexdigest()
    lease = await prefill_gate.enter(gate_key) if prefill_gate is not None else None
    try:
        async with semaphore:
            started = time.perf_counter()
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
                extra_headers={"X-Session-ID": "longbench-" + case["context_sha256"]}
                if document_routing
                else {},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            async for event in stream:
                content = event.choices[0].delta.content if event.choices else None
                if event.choices and event.choices[0].finish_reason:
                    finish_reason = event.choices[0].finish_reason
                if content:
                    first_token = first_token or time.perf_counter()
                    chunks.append(content)
                    if prefill_gate is not None:
                        prefill_gate.release(gate_key, lease)
                if event.usage is not None:
                    usage = event.usage.model_dump()
    finally:
        if prefill_gate is not None:
            prefill_gate.release(gate_key, lease)
    ended, prediction = time.perf_counter(), "".join(chunks).strip()
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
        "client_queue_seconds": started - queued,
        "end_to_end_ttft_seconds": (first_token or ended) - queued,
        "end_to_end_latency_seconds": ended - queued,
        "finish_reason": finish_reason,
        "usage": usage,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(args.cases)
    client = AsyncOpenAI(
        base_url=args.base_url, api_key="agentrix", timeout=args.timeout, max_retries=0
    )
    prefill_gate = None
    if args.coalesce_prefill:
        from agentrix_application.prefix_prefill_gate import PrefixPrefillGate

        prefill_gate = PrefixPrefillGate()
    if not args.question_waves:
        random.Random(args.seed).shuffle(cases)
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    phase_seconds = {}
    results = []
    try:
        if args.prime_first_question:
            for phase, first in (("first", True), ("followup", False)):
                phase_started = time.perf_counter()
                batch = await asyncio.gather(
                    *(
                        ask(
                            client,
                            args.model,
                            case,
                            question,
                            semaphore,
                            args.max_tokens,
                            args.document_routing,
                            prefill_gate,
                        )
                        for case in cases
                        for question in (
                            case["questions"][:1] if first else case["questions"][1:]
                        )
                    )
                )
                phase_seconds[phase] = time.perf_counter() - phase_started
                for row in batch:
                    row["phase"] = phase
                results.extend(batch)
        elif args.question_waves:
            rng = random.Random(args.seed)
            for wave in range(max(len(c["questions"]) for c in cases)):
                order = [c for c in cases if len(c["questions"]) > wave]
                rng.shuffle(order)
                batch = await asyncio.gather(
                    *(
                        ask(
                            client,
                            args.model,
                            case,
                            case["questions"][wave],
                            semaphore,
                            args.max_tokens,
                            args.document_routing,
                            prefill_gate,
                        )
                        for case in order
                    )
                )
                for row in batch:
                    row["wave"] = wave
                results.extend(batch)
        else:
            results = await asyncio.gather(
                *(
                    ask(
                        client,
                        args.model,
                        case,
                        question,
                        semaphore,
                        args.max_tokens,
                        args.document_routing,
                        prefill_gate,
                    )
                    for case in cases
                    for question in case["questions"]
                )
            )
    finally:
        await client.close()
    wall = time.perf_counter() - started
    return {
        "schema_version": 2,
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "seed": args.seed,
        "question_waves": args.question_waves,
        "prime_first_question": args.prime_first_question,
        "phase_seconds": phase_seconds,
        "coalesce_prefill": args.coalesce_prefill,
        "mean_end_to_end_ttft_seconds": statistics.fmean(
            r["end_to_end_ttft_seconds"] for r in results
        ),
        "document_routing": args.document_routing,
        "model": args.model,
        "case_count": len(cases),
        "question_count": len(results),
        "wall_seconds": wall,
        "questions_per_second": len(results) / wall,
        "mean_exact_match": statistics.fmean(r["exact_match"] for r in results),
        "mean_f1": statistics.fmean(r["f1"] for r in results),
        "mean_ttft_seconds": statistics.fmean(r["ttft_seconds"] for r in results),
        "p95_ttft_seconds": sorted(r["ttft_seconds"] for r in results)[
            max(0, int(len(results) * 0.95) - 1)
        ],
        "mean_latency_seconds": statistics.fmean(r["latency_seconds"] for r in results),
        "prompt_tokens": sum(
            (r["usage"] or {}).get("prompt_tokens", 0) for r in results
        ),
        "completion_tokens": sum(
            (r["usage"] or {}).get("completion_tokens", 0) for r in results
        ),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:9000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--document-routing", action="store_true")
    parser.add_argument("--question-waves", action="store_true")
    parser.add_argument(
        "--prime-first-question",
        action="store_true",
        help="Evaluate each document's first question, then fan out its remaining original questions",
    )
    parser.add_argument("--coalesce-prefill", action="store_true")
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_tokens < 1:
        parser.error("concurrency and max-tokens must be positive")
    if args.prime_first_question and (args.question_waves or args.coalesce_prefill):
        parser.error(
            "prime-first-question is a separate arrival pattern; omit waves and prefill coalescing"
        )
    if args.output.exists():
        parser.error("output already exists")
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
