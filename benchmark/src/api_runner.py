from __future__ import annotations

import asyncio
import os
import statistics
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
    case_index: int
    group_id: int
    input_tokens: int
    output_tokens: int
    latency_ms: float
    text: str
    strategy: str


async def run_api_case(
    common_context: str | list[str],
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
    case_count: int = 1,
    branch_group_size: int = 1,
    branch_order: str = "case_major",
) -> tuple[BenchmarkTrace, dict[str, Any]]:
    import random

    if branch_count <= 0 or output_tokens <= 0 or common_analysis_tokens <= 0:
        raise ValueError("branch count and output token limits must be positive")
    if case_count <= 0 or branch_group_size <= 0:
        raise ValueError("case count and branch group size must be positive")
    if concurrency <= 0 or arrival_interval_ms < 0:
        raise ValueError(
            "concurrency must be positive and arrival interval non-negative"
        )
    if api_mode not in {"responses", "chat"}:
        raise ValueError(f"unsupported API mode: {api_mode}")
    if branch_order not in {"case_major", "round_robin", "shuffle"}:
        raise ValueError(f"unsupported branch order: {branch_order}")

    case_started = time.perf_counter()
    client = AsyncOpenAI(api_key=os.getenv(api_key_env), base_url=base_url)
    contexts = (
        [common_context]
        if isinstance(common_context, str)
        else list(common_context[:case_count])
    )
    if len(contexts) != case_count:
        raise ValueError("common_context list length must match case_count")
    if target_prefix_tokens:
        contexts = [
            fit_text_to_tokens(context, target_prefix_tokens, model)
            for context in contexts
        ]

    common_started = time.perf_counter()

    async def request(
        messages: list[dict[str, str]], max_tokens: int
    ) -> tuple[str, int, int]:
        if api_mode == "responses":
            kwargs: dict[str, Any] = {
                "model": model,
                "input": messages,
                "max_output_tokens": max_tokens,
                "temperature": 0,
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
                "temperature": 0,
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

    common_results = await asyncio.gather(
        *(
            request([{"role": "user", "content": context}], common_analysis_tokens)
            for context in contexts
        )
    )
    common_latency_ms = (time.perf_counter() - common_started) * 1000
    shared_contexts = []
    common_cases = []
    for case_index, (context, result) in enumerate(zip(contexts, common_results)):
        common_analysis, common_input_tokens, common_output_tokens = result
        shared_context = (
            context
            + "\n\n--- Shared Analysis ---\n\n"
            + common_analysis
            + "\n\nContinue the analysis based on the shared context above."
        )
        shared_contexts.append(shared_context)
        common_cases.append(
            {
                "case_index": case_index,
                "input_tokens": common_input_tokens,
                "output_tokens": common_output_tokens,
                "prefix_tokens": count_tokens(shared_context, model),
                "text": common_analysis,
            }
        )
    local_prefix_tokens_by_case = [
        int(case["prefix_tokens"]) for case in common_cases
    ]
    local_prefix_tokens = int(statistics.fmean(local_prefix_tokens_by_case))
    total_branches = case_count * branch_count
    desired_suffixes = sample_suffixes(
        total_branches,
        suffix_distribution,
        suffix_mean,
        random.Random(seed),
    )
    semaphore = asyncio.Semaphore(concurrency)
    branch_specs = [
        (case_index, branch_index)
        for case_index in range(case_count)
        for branch_index in range(branch_count)
    ]
    if branch_order == "round_robin":
        branch_specs = [
            (case_index, branch_index)
            for branch_index in range(branch_count)
            for case_index in range(case_count)
        ]
    elif branch_order == "shuffle":
        random.Random(seed ^ 0x5F3759DF).shuffle(branch_specs)

    async def run_branch(
        case_index: int,
        branch_index: int,
        arrival_rank: int,
    ) -> APIBranchResult:
        index = case_index * branch_count + branch_index
        if arrival_interval_ms:
            await asyncio.sleep(arrival_rank * arrival_interval_ms / 1000)
        strategy = STRATEGIES[index % len(STRATEGIES)]
        group_id = branch_index // branch_group_size
        suffix_budget = max(desired_suffixes[index], 2)
        group_budget = (
            0 if branch_group_size == 1 else max(1, suffix_budget // 2)
        )
        leaf_budget = max(1, suffix_budget - group_budget)
        group_context = ""
        if group_budget:
            group_seed = (
                f"Case {case_index} group {group_id} shared branch notes.\n"
                f"All branches in this group inspect the same subsystem."
            )
            group_context = fit_text_to_tokens(group_seed, group_budget, model)
        private_seed = (
            f"{strategy}\n\nBranch-private analysis material for case "
            f"{case_index}.\nBranch ID: {branch_index}"
        )
        private_context = fit_text_to_tokens(
            private_seed,
            leaf_budget,
            model,
        )
        messages = [{"role": "user", "content": shared_contexts[case_index]}]
        if group_context:
            messages.append({"role": "user", "content": group_context})
        messages.append({"role": "user", "content": private_context})
        started = time.perf_counter()
        async with semaphore:
            text, input_tokens, actual_output_tokens = await request(
                messages,
                output_tokens,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        return APIBranchResult(
            branch_id=index,
            case_index=case_index,
            group_id=group_id,
            input_tokens=input_tokens,
            output_tokens=actual_output_tokens,
            latency_ms=latency_ms,
            text=text,
            strategy=strategy,
        )

    branch_phase_started = time.perf_counter()
    branches = await asyncio.gather(
        *(
            run_branch(case_index, branch_index, arrival_rank)
            for arrival_rank, (case_index, branch_index) in enumerate(branch_specs)
        )
    )
    branch_phase_latency_ms = (time.perf_counter() - branch_phase_started) * 1000
    # API input usage includes message framing. The tokenizer-derived shared text
    # length makes that framing part of each observed private suffix.
    trace = BenchmarkTrace(
        case_id=(
            f"api_forest_c{case_count}_p{local_prefix_tokens}_"
            f"b{branch_count}_g{branch_group_size}_{suffix_distribution}"
        ),
        prefix_tokens=local_prefix_tokens,
        branches=[
            BranchTrace(
                branch_id=result.branch_id,
                suffix_tokens=max(
                    0,
                    result.input_tokens
                    - local_prefix_tokens_by_case[result.case_index],
                ),
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
            "case_count": case_count,
            "branches_per_case": branch_count,
            "branch_group_size": branch_group_size,
            "branch_order": branch_order,
            "prefix_tokens_by_case": local_prefix_tokens_by_case,
        },
    )
    raw = {
        "model": model,
        "api_mode": api_mode,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "reasoning_effort": reasoning_effort,
        "case_count": case_count,
        "branches_per_case": branch_count,
        "branch_group_size": branch_group_size,
        "branch_order": branch_order,
        "total_latency_ms": (time.perf_counter() - case_started) * 1000,
        "branch_phase_latency_ms": branch_phase_latency_ms,
        "common": {
            "input_tokens": sum(case["input_tokens"] for case in common_cases),
            "output_tokens": sum(case["output_tokens"] for case in common_cases),
            "latency_ms": common_latency_ms,
            "cases": common_cases,
        },
        "branches": [asdict(result) for result in branches],
    }
    await client.close()
    return trace, raw
