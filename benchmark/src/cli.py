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
from models import BenchmarkTrace
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
    api.add_argument("--sample-count", type=int, default=1)
    api.add_argument("--full-dataset", action="store_true")
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
    api.add_argument("--case-count", type=int, default=1)
    api.add_argument("--branch-group-size", type=int, default=1)
    api.add_argument(
        "--branch-order",
        choices=["case_major", "round_robin", "shuffle"],
        default="case_major",
    )
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
    if args.sample_count <= 0:
        raise SystemExit("sample count must be positive")

    remaining_records = len(records) - args.sample_index
    sample_count = remaining_records if args.full_dataset else args.sample_count
    sample_count = min(sample_count, remaining_records)

    traces: list[BenchmarkTrace] = []
    raw_results = []
    end_index = args.sample_index + sample_count
    for batch_index, sample_start in enumerate(
        range(args.sample_index, end_index, args.case_count)
    ):
        current_case_count = min(args.case_count, end_index - sample_start)
        prompts = [
            record_to_prompt(args.dataset, records[sample_start + offset])
            for offset in range(current_case_count)
        ]
        common_context: str | list[str]
        common_context = prompts[0] if current_case_count == 1 else prompts
        trace, raw = asyncio.run(
            run_api_case(
                common_context,
                model=args.model,
                branch_count=args.branches,
                output_tokens=args.output_tokens,
                suffix_distribution=args.suffix_distribution,
                suffix_mean=args.suffix_mean,
                seed=args.seed + sample_start,
                target_prefix_tokens=args.prefix_tokens,
                concurrency=args.concurrency,
                arrival_interval_ms=args.arrival_interval_ms,
                common_analysis_tokens=args.common_analysis_tokens,
                api_mode=args.api_mode,
                base_url=args.base_url,
                api_key_env=api_key_env,
                reasoning_effort=args.reasoning_effort,
                case_count=current_case_count,
                branch_group_size=args.branch_group_size,
                branch_order=args.branch_order,
            )
        )
        trace = BenchmarkTrace(
            case_id=f"{trace.case_id}_s{sample_start}",
            prefix_tokens=trace.prefix_tokens,
            branches=trace.branches,
            suffix_distribution=trace.suffix_distribution,
            output_tokens=trace.output_tokens,
            arrival_mode=trace.arrival_mode,
            metadata={
                **trace.metadata,
                "batch_index": batch_index,
                "sample_start": sample_start,
                "sample_count": current_case_count,
            },
        )
        raw["batch_index"] = batch_index
        raw["sample_start"] = sample_start
        traces.append(trace)
        raw_results.append(raw)

    results = [compare_trace(trace) for trace in traces]
    write_results(args.output_dir, traces, results, raw_results)
    print(f"Wrote API benchmark to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
