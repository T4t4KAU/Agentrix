from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import tiktoken


def analyze_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    per_case = []
    segment_occurrences: Counter[str] = Counter()
    segment_tokens: dict[str, int] = {}
    total_materialized = 0
    total_unique_by_case = 0
    for case in cases:
        encoding = tiktoken.get_encoding(case["tokenizer"])
        shared = int(case["shared_parent_tokens"])
        branch_turn_tokens = []
        for branch in case["branches"]:
            trajectory = branch.get("trajectory")
            if not trajectory:
                trajectory = [
                    {"instruction": branch["private_instruction"]}
                ]
            tokens = []
            for turn in trajectory:
                content = turn["instruction"]
                if turn.get("tool_observation"):
                    content = (
                        f"Tool `{turn['tool']}` returned:\n"
                        f"{turn['tool_observation']}\n\n"
                        f"Next instruction:\n{turn['instruction']}"
                    )
                tokens.append(len(encoding.encode(content)))
            branch_turn_tokens.append(tokens)
        request_count = sum(len(tokens) for tokens in branch_turn_tokens)
        declared_user_tokens = sum(
            sum(tokens) for tokens in branch_turn_tokens
        )
        cumulative_user_tokens = sum(
            sum(tokens[: round_index + 1])
            for tokens in branch_turn_tokens
            for round_index in range(len(tokens))
        )
        materialized = request_count * shared + cumulative_user_tokens
        unique = shared + declared_user_tokens
        redundant = materialized - unique
        total_materialized += materialized
        total_unique_by_case += unique
        for segment in case.get("source_segments", []):
            digest = str(segment["sha256"])
            segment_occurrences[digest] += 1
            segment_tokens.setdefault(digest, int(segment["rendered_tokens"]))
        per_case.append(
            {
                "case_id": case["case_id"],
                "branches": len(branch_turn_tokens),
                "declared_model_requests": request_count,
                "shared_parent_tokens": shared,
                "declared_user_tokens": declared_user_tokens,
                "cumulative_user_tokens_materialized": cumulative_user_tokens,
                "materialized_prompt_tokens": materialized,
                "unique_case_tokens": unique,
                "logical_redundant_tokens": redundant,
                "logical_redundancy_ratio": redundant / materialized,
                "ideal_shared_prefix_reduction_ratio": (
                    redundant / materialized
                ),
            }
        )
    cross_case_redundant = sum(
        segment_tokens[digest] * (count - 1)
        for digest, count in segment_occurrences.items()
        if count > 1
    )
    return {
        "schema_version": 2,
        "definition": (
            "Logical prompt redundancy counts repeated frozen-parent and declared "
            "user/tool tokens across model requests. Runtime-generated assistant "
            "tokens are excluded. It is an upper bound on removable representation, "
            "not measured HBM traffic."
        ),
        "cases": per_case,
        "summary": {
            "case_count": len(cases),
            "branch_count": sum(len(case["branches"]) for case in cases),
            "declared_model_requests": sum(
                item["declared_model_requests"] for item in per_case
            ),
            "materialized_prompt_tokens": total_materialized,
            "unique_tokens_with_parent_once_per_case": total_unique_by_case,
            "logical_redundant_tokens": total_materialized - total_unique_by_case,
            "logical_redundancy_ratio": (
                (total_materialized - total_unique_by_case) / total_materialized
                if total_materialized
                else 0.0
            ),
            "source_segment_occurrences": sum(segment_occurrences.values()),
            "unique_source_segments": len(segment_occurrences),
            "cross_case_duplicate_segment_tokens": cross_case_redundant,
        },
        "runtime_metrics_required": [
            "token-exact shared-prefix depth per request",
            "physical local KV blocks reused per rank",
            "same case resident on multiple ranks",
            "tool-result segment occurrences and exact duplicate bytes",
            "context tokens retained after a branch terminates",
            "KV occupancy over time and eviction/recompute events",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Agent context redundancy")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = analyze_cases(cases)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
