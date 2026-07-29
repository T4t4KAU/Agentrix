from __future__ import annotations
import argparse, csv, json, statistics
from pathlib import Path
from typing import Any

def peak_memory(path: Path) -> int | None:
    if not path.is_file():
        return None
    values = []
    with path.open() as handle:
        for row in csv.reader(handle):
            # nvidia-smi --format=csv,noheader,nounits emits:
            # timestamp,gpu_uuid,pid,used_memory
            if len(row) < 4:
                continue
            try:
                values.append(int(float(row[3].strip())))
            except ValueError:
                continue
    return max(values) if values else None

def compare(baseline: dict[str, Any], optimized: dict[str, Any]) -> dict[str, Any]:
    left = {r["source_id"]: r for r in baseline["results"]}
    right = {r["source_id"]: r for r in optimized["results"]}
    keys = sorted(left.keys() & right.keys())
    deltas = [right[k]["f1"] - left[k]["f1"] for k in keys]
    agreement = [float(left[k]["prediction"].strip().casefold() ==
                       right[k]["prediction"].strip().casefold()) for k in keys]
    return {"paired_questions": len(keys), "baseline_f1": baseline["mean_f1"],
        "optimized_f1": optimized["mean_f1"],
        "paired_mean_f1_delta": statistics.fmean(deltas) if deltas else 0,
        "prediction_exact_agreement": statistics.fmean(agreement) if agreement else 0,
        "wall_speedup": baseline["wall_seconds"] / optimized["wall_seconds"],
        "throughput_speedup": optimized["questions_per_second"] / baseline["questions_per_second"],
        "ttft_speedup": baseline["mean_ttft_seconds"] / optimized["mean_ttft_seconds"]}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads((args.baseline / "run.json").read_text())
    optimized = json.loads((args.optimized / "run.json").read_text())
    report = compare(baseline, optimized)
    report["baseline_peak_process_memory_mib"] = peak_memory(args.baseline / "gpu_process_memory.csv")
    report["optimized_peak_process_memory_mib"] = peak_memory(args.optimized / "gpu_process_memory.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
