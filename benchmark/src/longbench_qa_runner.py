from __future__ import annotations
import argparse, asyncio, json, statistics, time
from pathlib import Path
from typing import Any
from openai import AsyncOpenAI
from longbench_qa import load_cases, score_answer

async def ask(client: AsyncOpenAI, model: str, case: dict[str, Any],
              question: dict[str, Any], semaphore: asyncio.Semaphore,
              max_tokens: int) -> dict[str, Any]:
    prompt = ("Answer the question using only the document below. Give only the "
              "shortest answer that fully answers the question; do not explain."
              "\n\nDOCUMENT:\n" + case["context"])
    started, first_token, chunks, usage = time.perf_counter(), None, [], None
    async with semaphore:
        stream = await client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": prompt},
                {"role": "user", "content": question["question"]}],
            temperature=0, max_tokens=max_tokens, stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        async for event in stream:
            content = event.choices[0].delta.content if event.choices else None
            if content:
                first_token = first_token or time.perf_counter()
                chunks.append(content)
            if event.usage is not None:
                usage = event.usage.model_dump()
    ended, prediction = time.perf_counter(), "".join(chunks).strip()
    return {
        "case_id": case["case_id"], "dataset": case["dataset"],
        "context_sha256": case["context_sha256"], "context_tokens": case["context_tokens"],
        "source_id": question["source_id"], "question": question["question"],
        "answers": question["answers"], "prediction": prediction,
        **score_answer(prediction, question["answers"]),
        "ttft_seconds": (first_token or ended) - started,
        "latency_seconds": ended - started, "usage": usage,
    }

async def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(args.cases)
    client = AsyncOpenAI(base_url=args.base_url, api_key="agentrix", timeout=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    results = await asyncio.gather(*(ask(client, args.model, case, question,
        semaphore, args.max_tokens) for case in cases for question in case["questions"]))
    wall = time.perf_counter() - started
    return {
        "schema_version": 1, "model": args.model, "case_count": len(cases),
        "question_count": len(results), "wall_seconds": wall,
        "questions_per_second": len(results) / wall,
        "mean_exact_match": statistics.fmean(r["exact_match"] for r in results),
        "mean_f1": statistics.fmean(r["f1"] for r in results),
        "mean_ttft_seconds": statistics.fmean(r["ttft_seconds"] for r in results),
        "p95_ttft_seconds": sorted(r["ttft_seconds"] for r in results)[max(0, int(len(results)*.95)-1)],
        "mean_latency_seconds": statistics.fmean(r["latency_seconds"] for r in results),
        "prompt_tokens": sum((r["usage"] or {}).get("prompt_tokens", 0) for r in results),
        "completion_tokens": sum((r["usage"] or {}).get("completion_tokens", 0) for r in results),
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
    args = parser.parse_args()
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
