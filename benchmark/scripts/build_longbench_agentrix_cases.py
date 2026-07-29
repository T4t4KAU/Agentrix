#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
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
    parser.add_argument("--max-context-tokens", type=int, default=38_000)
    args = parser.parse_args()
    counter = None
    if args.model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        counter = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    sources = [args.data_dir / f"{name}.jsonl" for name in args.datasets]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        parser.error("missing dataset files: " + ", ".join(missing))
    cases = build_shared_document_cases(sources, maximum_cases=args.cases,
        token_counter=counter, maximum_context_tokens=args.max_context_tokens)
    if len(cases) < args.cases:
        parser.error(f"only {len(cases)} qualifying shared-document cases found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "questions": sum(len(c["questions"]) for c in cases),
        "context_tokens": sum(c["context_tokens"] for c in cases), "output": str(args.output)}, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
