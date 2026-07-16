#!/usr/bin/env python3
"""Freeze long-prefix HotpotQA cases for the Agentrix fanout benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR / "src"))

from hotpot import HotpotExample, load_hotpot  # noqa: E402


def render_shared_context(examples: list[HotpotExample]) -> str:
    sections = []
    seen_titles: set[str] = set()
    for example in examples:
        for paragraph in example.context:
            if paragraph.title in seen_titles:
                continue
            seen_titles.add(paragraph.title)
            sentences = "\n".join(
                f"[{sentence_id}] {sentence}"
                for sentence_id, sentence in enumerate(paragraph.sentences)
            )
            sections.append(f"[Local source: {paragraph.title}]\n{sentences}")
    return "\n\n".join(sections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hotpot-path", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--paragraphs", type=int, default=10)
    parser.add_argument(
        "--context-groups",
        type=int,
        default=4,
        help="Target case plus this many minus one deterministic donor cases.",
    )
    parser.add_argument("--branches", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.cases <= 0
        or args.paragraphs <= 0
        or args.branches <= 0
        or args.context_groups <= 0
    ):
        raise SystemExit(
            "cases, paragraphs, branches, and context-groups must be positive"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    examples = [
        example
        for example in load_hotpot(args.hotpot_path)
        if len(example.context) == args.paragraphs
    ]
    ranked = []
    for example in examples:
        shared_context = render_shared_context([example])
        context_tokens = len(tokenizer.encode(shared_context, add_special_tokens=False))
        ranked.append((context_tokens, example.example_id, example))
    selected = sorted(ranked, key=lambda item: (-item[0], item[1]))[: args.cases]
    if len(selected) != args.cases:
        raise SystemExit(
            f"requested {args.cases} cases but only {len(selected)} matched"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    selected_lengths = []
    donor_pool = sorted(ranked, key=lambda item: (-item[0], item[1]))
    for rank, (_, _, example) in enumerate(selected, 1):
        target_index = next(
            index
            for index, item in enumerate(donor_pool)
            if item[1] == example.example_id
        )
        donors = []
        offset = 1
        while len(donors) < args.context_groups - 1:
            donor = donor_pool[(target_index + offset) % len(donor_pool)][2]
            offset += 1
            if donor.example_id != example.example_id:
                donors.append(donor)
        combined_context = render_shared_context([example, *donors])
        context_tokens = len(
            tokenizer.encode(combined_context, add_special_tokens=False)
        )
        paragraph_count = sum(
            len(item.context) for item in [example, *donors]
        )
        selected_lengths.append(context_tokens)
        lines.append(
            json.dumps(
                {
                    "id": example.example_id,
                    "rank": rank,
                    "question": example.question,
                    "type": example.question_type,
                    "level": example.level,
                    "paragraphs": paragraph_count,
                    "shared_context_tokens": context_tokens,
                    "branches": args.branches,
                    "distractor_ids": [donor.example_id for donor in donors],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    lengths = selected_lengths
    print(
        json.dumps(
            {
                "cases": len(selected),
                "min_shared_context_tokens": min(lengths),
                "max_shared_context_tokens": max(lengths),
                "mean_shared_context_tokens": sum(lengths) / len(lengths),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
