from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import yaml

from api_runner import run_api_case
from data import inspect_datasets, load_records, record_to_prompt
from metrics import compare_trace
from reporting import write_results
from synthetic import traces_from_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentrix-bench",
        description="Benchmark logical shared-prefix KV/Tile/Launch savings.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-data")
    inspect_parser.add_argument("--json", action="store_true")

    simulate = subparsers.add_parser("simulate")
    simulate.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    simulate.add_argument("--output-dir", type=Path, default=Path("results/simulated"))

    api = subparsers.add_parser("run-api")
    api.add_argument("--dataset", choices=["swebench", "agencybench"], required=True)
    api.add_argument("--data-path", type=Path)
    api.add_argument("--sample-index", type=int, default=0)
    api.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.5"))
    api.add_argument(
        "--api-mode",
        choices=["responses", "chat"],
        default=os.getenv("OPENAI_API_MODE", "responses"),
    )
    api.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    api.add_argument("--api-key-env", default=os.getenv("OPENAI_API_KEY_ENV"))
    api.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default=os.getenv("OPENAI_REASONING_EFFORT"),
    )
    api.add_argument("--branches", type=int, default=8)
    api.add_argument("--prefix-tokens", type=int, default=8192)
    api.add_argument("--suffix-mean", type=int, default=768)
    api.add_argument(
        "--suffix-distribution",
        choices=["equal", "uniform", "lognormal", "long_tail"],
        default="lognormal",
    )
    api.add_argument("--output-tokens", type=int, default=256)
    api.add_argument("--common-analysis-tokens", type=int, default=256)
    api.add_argument("--concurrency", type=int, default=8)
    api.add_argument("--arrival-interval-ms", type=int, default=0)
    api.add_argument("--seed", type=int, default=2026)
    api.add_argument("--output-dir", type=Path, default=Path("results/api"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect-data":
        report = inspect_datasets()
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            for item in report:
                print(
                    f"{item['dataset']}: {item['records']} records "
                    f"({item['path']})"
                )
        return 0

    if args.command == "simulate":
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        traces = traces_from_config(config)
        results = [compare_trace(trace, int(config.get("tile_k", 128))) for trace in traces]
        write_results(args.output_dir, traces, results)
        print(f"Wrote {len(results)} cases to {args.output_dir}")
        return 0

    api_key_env = args.api_key_env or (
        "DEEPSEEK_API_KEY"
        if args.api_mode == "chat" and args.base_url == "https://api.deepseek.com"
        else "OPENAI_API_KEY"
    )
    if not os.getenv(api_key_env):
        raise SystemExit(f"{api_key_env} is required for run-api")
    records = load_records(args.dataset, args.data_path)
    if not 0 <= args.sample_index < len(records):
        raise SystemExit(
            f"sample index {args.sample_index} is outside 0..{len(records) - 1}"
        )
    prompt = record_to_prompt(args.dataset, records[args.sample_index])
    trace, raw = asyncio.run(
        run_api_case(
            prompt,
            model=args.model,
            branch_count=args.branches,
            output_tokens=args.output_tokens,
            suffix_distribution=args.suffix_distribution,
            suffix_mean=args.suffix_mean,
            seed=args.seed,
            target_prefix_tokens=args.prefix_tokens,
            concurrency=args.concurrency,
            arrival_interval_ms=args.arrival_interval_ms,
            common_analysis_tokens=args.common_analysis_tokens,
            api_mode=args.api_mode,
            base_url=args.base_url,
            api_key_env=api_key_env,
            reasoning_effort=args.reasoning_effort,
        )
    )
    result = compare_trace(trace)
    write_results(args.output_dir, [trace], [result], raw)
    print(f"Wrote API benchmark to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
