from __future__ import annotations

import asyncio
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from openai import AsyncOpenAI

from models import BenchmarkTrace, BranchTrace
from synthetic import sample_suffixes
from tokens import count_tokens, fit_text_to_tokens


STRATEGIES = [
    "Provide the minimal fix.",
    "Analyze from the caller's perspective.",
    "Analyze from the data-structure perspective.",
    "Focus on error handling.",
    "Focus on compatibility.",
    "Propose a controlled refactor.",
    "Prioritize performance.",
    "Prioritize maintainability.",
]


@dataclass
class APIBranchResult:
    branch_id: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    text: str
    strategy: str


async def run_api_case(
    common_context: str,
    model: str,
    branch_count: int,
    output_tokens: int,
    suffix_distribution: str,
    suffix_mean: int,
    seed: int,
    target_prefix_tokens: int | None = None,
    concurrency: int = 8,
    arrival_interval_ms: int = 0,
    common_analysis_tokens: int = 256,
    api_mode: str = "responses",
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    reasoning_effort: str | None = None,
) -> tuple[BenchmarkTrace, dict[str, Any]]:
    import random

    if branch_count <= 0 or output_tokens <= 0 or common_analysis_tokens <= 0:
        raise ValueError("branch count and output token limits must be positive")
    if concurrency <= 0 or arrival_interval_ms < 0:
        raise ValueError("concurrency must be positive and arrival interval non-negative")
    if api_mode not in {"responses", "chat"}:
        raise ValueError(f"unsupported API mode: {api_mode}")

    case_started = time.perf_counter()
    client = AsyncOpenAI(api_key=os.getenv(api_key_env), base_url=base_url)
    if target_prefix_tokens:
        common_context = fit_text_to_tokens(common_context, target_prefix_tokens, model)

    common_started = time.perf_counter()

    async def request(
        messages: list[dict[str, str]], max_tokens: int
    ) -> tuple[str, int, int]:
        if api_mode == "responses":
            kwargs: dict[str, Any] = {
                "model": model,
                "input": messages,
                "max_output_tokens": max_tokens,
            }
            if reasoning_effort:
                kwargs["reasoning"] = {"effort": reasoning_effort}
            response = await client.responses.create(**kwargs)
            if response.usage is None:
                raise RuntimeError("API response did not include token usage")
            return (
                response.output_text,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
        if api_mode == "chat":
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            response = await client.chat.completions.create(**kwargs)
            if response.usage is None:
                raise RuntimeError("API response did not include token usage")
            message = response.choices[0].message
            text = message.content or ""
            if not text:
                # DeepSeek reasoning models expose the generated reasoning through
                # an OpenAI-compatible extension when the visible answer is empty.
                text = str((message.model_extra or {}).get("reasoning_content") or "")
            return text, response.usage.prompt_tokens, response.usage.completion_tokens
        raise AssertionError("unreachable")

    common_analysis, common_input_tokens, common_output_tokens = await request(
        [{"role": "user", "content": common_context}], common_analysis_tokens
    )
    common_latency_ms = (time.perf_counter() - common_started) * 1000
    shared_context = (
        common_context
        + "\n\n--- Shared Analysis ---\n\n"
        + common_analysis
        + "\n\nContinue the analysis based on the shared context above."
    )
    local_prefix_tokens = count_tokens(shared_context, model)
    desired_suffixes = sample_suffixes(
        branch_count, suffix_distribution, suffix_mean, random.Random(seed)
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def run_branch(index: int) -> APIBranchResult:
        if arrival_interval_ms:
            await asyncio.sleep(index * arrival_interval_ms / 1000)
        strategy = STRATEGIES[index % len(STRATEGIES)]
        private_seed = (
            f"{strategy}\n\nBranch-private analysis material:\n{common_context}\n"
            f"\nBranch ID: {index}"
        )
        private_context = fit_text_to_tokens(
            private_seed, max(desired_suffixes[index], 1), model
        )
        started = time.perf_counter()
        async with semaphore:
            text, input_tokens, actual_output_tokens = await request(
                [
                    {"role": "user", "content": shared_context},
                    {"role": "user", "content": private_context},
                ],
                output_tokens,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        return APIBranchResult(
            branch_id=index,
            input_tokens=input_tokens,
            output_tokens=actual_output_tokens,
            latency_ms=latency_ms,
            text=text,
            strategy=strategy,
        )

    branch_phase_started = time.perf_counter()
    branches = await asyncio.gather(*(run_branch(i) for i in range(branch_count)))
    branch_phase_latency_ms = (time.perf_counter() - branch_phase_started) * 1000
    # API input usage includes message framing. The tokenizer-derived shared text
    # length makes that framing part of each observed private suffix.
    trace = BenchmarkTrace(
        case_id=(
            f"api_p{local_prefix_tokens}_b{branch_count}_{suffix_distribution}"
        ),
        prefix_tokens=local_prefix_tokens,
        branches=[
            BranchTrace(
                branch_id=result.branch_id,
                suffix_tokens=max(0, result.input_tokens - local_prefix_tokens),
                decode_tokens=result.output_tokens,
                input_tokens=result.input_tokens,
                latency_ms=result.latency_ms,
                strategy=result.strategy,
            )
            for result in branches
        ],
        suffix_distribution=suffix_distribution,
        output_tokens=output_tokens,
        arrival_mode="fixed_interval" if arrival_interval_ms else "simultaneous",
        metadata={
            "model": model,
            "seed": seed,
            "api_mode": api_mode,
            "base_url": base_url,
            "api_key_env": api_key_env,
            "reasoning_effort": reasoning_effort,
        },
    )
    raw = {
        "model": model,
        "api_mode": api_mode,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "reasoning_effort": reasoning_effort,
        "total_latency_ms": (time.perf_counter() - case_started) * 1000,
        "branch_phase_latency_ms": branch_phase_latency_ms,
        "common": {
            "input_tokens": common_input_tokens,
            "output_tokens": common_output_tokens,
            "latency_ms": common_latency_ms,
            "text": common_analysis,
        },
        "branches": [asdict(result) for result in branches],
    }
    return trace, raw
