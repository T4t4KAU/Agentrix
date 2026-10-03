"""Analyze server-side AgentX resource samples; counters include warmup/drain."""

import argparse
import json
import re
from pathlib import Path


def counters(text):
    result = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric, value = line.rsplit(" ", 1)
        if metric.startswith("vllm:prompt_tokens_by_source_total{"):
            key = re.search(r'source="([^"]+)"', metric).group(1)
        elif metric.startswith("vllm:kv_offload_total_bytes_total{"):
            key = re.search(r'transfer_type="([^"]+)"', metric).group(1)
        elif metric.startswith("vllm:num_preemptions_total{"):
            key = "preemptions"
        else:
            continue
        result[key] = result.get(key, 0) + float(value)
    return result


def summarize(runs, samples):
    progress = json.loads((runs / "progress.json").read_text())
    config = json.loads((runs / "launch-private.json").read_text())
    labels = [c["label"] for c in config["cells"]]
    if progress["phase"] != "complete" or set(progress["completed"]) != set(labels):
        raise ValueError("Incomplete experiment")
    rows = {}
    for cell in config["cells"]:
        label = cell["label"]
        directory = runs / label
        report = json.loads((directory / "result-summary.json").read_text())
        if (
            not report["submission_valid"]
            or report["request_errors"]
            or report["cancelled_at_drain_timeout"]
        ):
            raise ValueError(f"Invalid report: {label}")
        with (directory / "metrics.jsonl").open() as stream:
            first = json.loads(next(stream))
            last = first
            for line in stream:
                last = json.loads(line)
        before, after = counters(first["metrics"]), counters(last["metrics"])
        delta = {key: value - before.get(key, 0) for key, value in after.items()}
        if any(v < 0 for v in delta.values()):
            raise ValueError("Counter reset during benchmark")
        argv = cell["server"]
        cpu = 0
        if "--kv-transfer-config" in argv:
            cpu = json.loads(argv[argv.index("--kv-transfer-config") + 1])[
                "kv_connector_extra_config"
            ]["cpu_bytes_to_use"]
        rows[label] = {
            "report": report,
            "sample_window": [first["time"], last["time"]],
            "device_kv_budget_bytes_per_rank": int(
                argv[argv.index("--kv-cache-memory-bytes") + 1]
            ),
            "cpu_budget_bytes_per_rank": cpu,
            "counters_including_warmup_drain": delta,
            "board_hbm_peak_mib": None,
            "board_samples": 0,
        }
    recently_owned = {}
    with samples.open() as stream:
        for line in stream:
            sample = json.loads(line)
            now = sample["time"]
            recently_owned = {
                pid: t for pid, t in recently_owned.items() if 0 <= now - t <= 10
            }
            recently_owned.update(
                {pid: now for pid in sample["controller_descendants"]}
            )
            for row in rows.values():
                start, end = row["sample_window"]
                if not start <= now <= end or sample["returncode"]:
                    continue
                usages = []
                for text in sample["stdout"].splitlines():
                    process = re.match(
                        r"\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|[^|]+\|\s*\d+\s*\|\s*(\d+)\s*\|",
                        text,
                    )
                    if process and int(process.group(1)) not in recently_owned:
                        raise ValueError("Unowned NPU process in sample")
                    if re.search(
                        r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9]", text
                    ):
                        numbers = re.findall(r"(\d+)\s*/\s*(\d+)", text)
                        if numbers:
                            usages.append(int(numbers[-1][0]))
                if len(usages) == 2:
                    row["board_samples"] += 1
                    row["board_hbm_peak_mib"] = max(
                        row["board_hbm_peak_mib"] or 0, sum(usages)
                    )
    if any(not row["board_samples"] for row in rows.values()):
        raise ValueError("Missing board memory samples")
    return {
        "scope": "Counters include warmup/drain; board-wide observed peaks, not allocator peaks; CPU budgets are not resident usage; different completed request sets are not matched work.",
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.runs, args.samples)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    for name, row in result["rows"].items():
        print(name, json.dumps({k: v for k, v in row.items() if k != "report"}))
