from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from longbench_qa import build_shared_document_cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["multifieldqa_en", "qasper"])
    parser.add_argument("--model")
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--min-questions", type=int, default=2)
    parser.add_argument("--max-questions", type=int, default=4)
    parser.add_argument("--unique-questions", action="store_true")
    parser.add_argument("--min-context-tokens", type=int, default=0)
    parser.add_argument("--max-context-tokens", type=int, default=38_000)
    args = parser.parse_args()
    if (
        not 2 <= args.min_questions <= args.max_questions
        or not 0 <= args.min_context_tokens <= args.max_context_tokens
        or args.cases < 1
    ):
        parser.error(
            "require 2 <= min-questions <= max-questions, a valid context range and positive cases"
        )
    counter = None
    if args.model:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        def counter(text):
            return len(tokenizer.encode(text, add_special_tokens=False))

    sources = [args.data_dir / f"{name}.jsonl" for name in args.datasets]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        parser.error("missing dataset files: " + ", ".join(missing))
    cases = build_shared_document_cases(
        sources,
        maximum_cases=args.cases,
        minimum_questions=args.min_questions,
        maximum_questions=args.max_questions,
        token_counter=counter,
        minimum_context_tokens=args.min_context_tokens,
        maximum_context_tokens=args.max_context_tokens,
        unique_questions=args.unique_questions,
    )
    if len(cases) < args.cases:
        parser.error(f"only {len(cases)} qualifying shared-document cases found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cases": len(cases),
                "questions": sum(len(c["questions"]) for c in cases),
                "context_tokens": sum(c["context_tokens"] for c in cases),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
