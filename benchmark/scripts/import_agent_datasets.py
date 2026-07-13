#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))

from src.data import load_records  # noqa: E402


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import AgentBoard and AppWorld prompt records into benchmark data."
    )
    parser.add_argument("--agentboard-source", type=Path, required=True)
    parser.add_argument("--appworld-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=BENCHMARK_DIR / "data")
    args = parser.parse_args()

    datasets = {
        "agentboard": args.agentboard_source,
        "appworld": args.appworld_source,
    }
    for name, source in datasets.items():
        records = load_records(name, source)
        output = args.output_dir / f"{name}.jsonl"
        _write_jsonl(output, records)
        print(f"{name}: {len(records)} records -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
