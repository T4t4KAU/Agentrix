"""Validate and aggregate completed lifecycle cells on the experiment server."""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from benchmark_agent_kv_tiering import save_result
from benchmark_dp_kv_lifecycle import percentile


def sampled_hbm(directory, labels):
    """Observed board-wide HBM during benchmark phases, not allocator peaks."""
    paths = sorted(directory.glob("npu-samples*.jsonl"))
    if not paths:
        return {}
    transitions = []
    for line in (directory / "controller.log").read_text().splitlines():
        try:
            state = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "phase" in state and "time" in state:
            transitions.append(state)
    peaks = defaultdict(list)
    samples = sorted(
        (json.loads(line) for path in paths for line in path.read_text().splitlines()),
        key=lambda sample: sample["time"],
    )
    recently_owned = {}
    for sample in samples:
        # Device snapshots and process-tree snapshots are not atomic. Workers
        # may be reparented during cleanup between the two reads. Only allow
        # PIDs whose ownership was independently observed in the last 10 s.
        recently_owned = {
            pid: seen
            for pid, seen in recently_owned.items()
            if 0 <= sample["time"] - seen <= 10
        }
        recently_owned.update(
            {pid: sample["time"] for pid in sample["controller_descendants"]}
        )
        prior = [s for s in transitions if s["time"] <= sample["time"]]
        if not prior or prior[-1]["phase"] != "benchmark":
            continue
        label = next((v for v in labels if prior[-1]["cell"].startswith(v + "-")), None)
        if label is None or sample["returncode"]:
            continue
        # Select memory rows with a PCI bus address, excluding process rows.
        usages = []
        for row in sample["stdout"].splitlines():
            process = re.match(
                r"\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|[^|]+\|\s*\d+\s*\|\s*(\d+)\s*\|",
                row,
            )
            if process and int(process.group(1)) not in recently_owned:
                raise ValueError(
                    "NPU sample contains a process outside this experiment"
                )
            if re.search(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]", row):
                numbers = re.findall(r"(\d+)\s*/\s*(\d+)", row)
                if numbers:
                    usages.append(int(numbers[-1][0]))
        if len(usages) == 2:
            peaks[label].append(sum(usages))
    return {
        label: {"samples": len(values), "observed_total_hbm_peak_mib": max(values)}
        for label, values in peaks.items()
    }


def summarize(directory):
    config = json.loads((directory / "launch-private.json").read_text())
    progress = json.loads((directory / "progress.json").read_text())
    if progress["phase"] != "complete":
        raise ValueError("Matrix is incomplete")
    cells = {}
    for case in config["cases"]:
        for mode in case["modes"]:
            for seed in config.get("seeds", [20261001, 20261020]):
                name = f"{case['label']}-{mode}-{seed}"
                cells[name] = (case, mode, seed)
    if set(progress["completed"]) != set(cells):
        raise ValueError("Completed cells differ from planned matrix")
    plans, reference, groups = {}, {}, defaultdict(list)
    comparisons = 0
    for name, (case, mode, seed) in cells.items():
        result = json.loads((directory / f"{name}.json").read_text())
        if not result.get("valid") or len(result["trials"]) != config.get("trials", 2):
            raise ValueError(f"Invalid or incomplete cell: {name}")
        cfg = result["configuration"]
        settings = {
            k: cfg[k]
            for k in (
                "sessions",
                "pressure_requests",
                "prompt_tokens",
                "output_tokens",
                "tool_gap_seconds",
                "settle_seconds",
            )
        }
        plan = (result["prompt_plan_sha256"], settings)
        if plans.setdefault(seed, plan) != plan:
            raise ValueError("Input plans or workload settings differ")
        for trial in result["trials"]:
            for phase in ("prime", "pressure", "resume"):
                expected = (
                    cfg["pressure_requests"] if phase == "pressure" else cfg["sessions"]
                )
                if len(trial[phase]) != expected:
                    raise ValueError("Incomplete request set")
                for row in trial[phase]:
                    key = (seed, phase, row["session"])
                    if len(row["token_ids"]) != cfg["output_tokens"]:
                        raise ValueError("Incomplete output tokens")
                    if key in reference:
                        comparisons += 1
                        if reference[key] != row["token_ids"]:
                            raise ValueError(f"Output mismatch: {name}/{key}")
                    else:
                        reference[key] = row["token_ids"]
            groups[(case["label"], mode)].append(trial)
    rows = []
    for (label, mode), trials in groups.items():
        case = next(c for c in config["cases"] if c["label"] == label)
        argv = case["argv"]
        latencies = [r["ttft_ms"] for t in trials for r in t["resume"]]
        reads = sum(t["resume_counters"].get("load_bytes", 0) for t in trials)
        if mode == "selective" and reads == 0:
            raise ValueError("Selective arm did not demonstrate external restoration")
        rows.append(
            {
                "case": label,
                "mode": mode,
                "trials": len(trials),
                "kv_bytes_per_rank": int(
                    argv[argv.index("--kv-cache-memory-bytes") + 1]
                ),
                "cpu_bytes_per_rank": 0
                if mode == "apc"
                else json.loads(argv[argv.index("--kv-transfer-config") + 1])[
                    "kv_connector_extra_config"
                ]["cpu_bytes_to_use"],
                "mean_resume_ttft_ms": statistics.fmean(latencies),
                "p95_resume_ttft_ms": percentile(latencies, 0.95),
                "pressure_store_bytes": sum(
                    t["pressure_counters"].get("store_bytes", 0) for t in trials
                ),
                "resume_load_bytes": reads,
                "resume_recomputed_tokens": sum(
                    t["resume_counters"]["tokens_local_compute"] for t in trials
                ),
                "mean_trial_elapsed_seconds": statistics.fmean(
                    t["elapsed_seconds"] for t in trials
                ),
            }
        )
    return {
        "valid": True,
        "cross_run_output_comparisons": comparisons,
        "board_samples": sampled_hbm(directory, [c["label"] for c in config["cases"]]),
        "scope": "Sequential pressure, not concurrent capacity or AgentX; KV budgets and sampled board HBM are reported separately",
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    save_result(args.directory / "validated-summary.json", result)
    print(json.dumps(result, indent=2))
