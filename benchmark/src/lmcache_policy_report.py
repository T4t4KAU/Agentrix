from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PERFORMANCE_METRICS = (
    ("case_wall_time_ms", "E2E wall ms", False),
    ("branch_phase_wall_ms", "Branch wall ms", False),
    ("branch_output_tokens_per_s", "Branch tok/s", True),
    ("end_to_end_output_tokens_per_s", "E2E tok/s", True),
)


def read_results(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"expected at least one result row in {path}")

    def total(key: str) -> float:
        return sum(float(row.get(key) or 0) for row in rows)

    total_output_tokens = total("branch_total_output_tokens")
    total_case_ms = total("case_wall_time_ms")
    total_branch_ms = total("branch_phase_wall_ms")
    baseline_kv = total("baseline_unique_kv")
    actual_kv = total("monowire_unique_kv")
    kv_tokens_saved = baseline_kv - actual_kv
    return {
        "case_wall_time_ms": total_case_ms,
        "branch_phase_wall_ms": total_branch_ms,
        "branch_output_tokens_per_s": _throughput(total_output_tokens, total_branch_ms),
        "end_to_end_output_tokens_per_s": _throughput(
            total_output_tokens, total_case_ms
        ),
        "baseline_unique_kv": baseline_kv,
        "monowire_unique_kv": actual_kv,
        "kv_tokens_saved": kv_tokens_saved,
        "kv_bytes_per_token": float(rows[0].get("kv_bytes_per_token") or 0),
        "kv_gib_saved": total("kv_bytes_saved") / 1024**3,
        "kv_reduction_percent": (
            100 * kv_tokens_saved / baseline_kv if baseline_kv else 0
        ),
    }


def parse_lmcache_log(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="ignore")

    def token_sum(pattern: str) -> int:
        return sum(int(value) for value in re.findall(pattern, text))

    reload_demand_by_request: dict[str, int] = {}
    for request_id, value in re.findall(
        r"Reqid:\s*([^,\s]+).*?need to load:\s*(\d+)", text
    ):
        reload_demand_by_request[request_id] = max(
            reload_demand_by_request.get(request_id, 0), int(value)
        )

    return {
        "reload_demand_tokens": sum(reload_demand_by_request.values()),
        "retrieved_tokens": token_sum(r"Retrieved\s+(\d+)\s+out of"),
        "stored_tokens": token_sum(r"Stored\s+(\d+)\s+out of"),
        "disk_load_allocation_failures": len(
            re.findall(r"Memory allocation failed during (?:async )?disk load", text)
        ),
    }


def read_smoke_summary(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def render_report(
    root: Path,
    baseline_name: str,
    optimized_name: str,
) -> str:
    baseline = read_results(root / "baseline" / "fork_attn" / "benchmark_results.csv")
    optimized = read_results(root / "optimized" / "fork_attn" / "benchmark_results.csv")
    baseline_cache = parse_lmcache_log(
        root / "baseline" / "fork_attn" / "vllm_server.log"
    )
    optimized_cache = parse_lmcache_log(
        root / "optimized" / "fork_attn" / "vllm_server.log"
    )
    baseline_summary = read_smoke_summary(root / "baseline" / "smoke_summary.txt")
    optimized_summary = read_smoke_summary(root / "optimized" / "smoke_summary.txt")
    kv_bytes_per_token = int(
        optimized["kv_bytes_per_token"] or baseline["kv_bytes_per_token"]
    )
    baseline_reload = baseline_cache["reload_demand_tokens"]
    optimized_reload = optimized_cache["reload_demand_tokens"]
    reload_saved = baseline_reload - optimized_reload
    reload_reduction = _reduction(baseline_reload, optimized_reload)
    reload_gib_saved = reload_saved * kv_bytes_per_token / 1024**3
    baseline_disk_bytes = int(baseline_summary.get("disk_bytes", 0))
    optimized_disk_bytes = int(optimized_summary.get("disk_bytes", 0))
    disk_gib_saved = (baseline_disk_bytes - optimized_disk_bytes) / 1024**3

    lines = [
        "# LMCache Policy Comparison",
        "",
        f"Baseline: LMCache default `{baseline_name}` policy.",
        "",
        "## End-to-End Performance",
        "",
        "| Metric | Baseline | Optimized | Improvement |",
        "|---|---:|---:|---:|",
    ]
    for key, label, higher_is_better in PERFORMANCE_METRICS:
        base = baseline[key]
        opt = optimized[key]
        raw = (opt / base - 1) * 100 if base else 0
        improvement = raw if higher_is_better else -raw
        lines.append(f"| {label} | {base:.3f} | {opt:.3f} | {improvement:+.2f}% |")

    lines.extend(
        [
            "",
            "## LMCache KV Reload Reduction",
            "",
            f"**Total KV reload demand reduction vs default LMCache:** "
            f"{reload_saved} tokens, {reload_gib_saved:.3f} GiB, "
            f"{reload_reduction:.2f}%.",
            "",
            f"**Disk-tier footprint reduction:** {disk_gib_saved:.3f} GiB, "
            f"{_reduction(baseline_disk_bytes, optimized_disk_bytes):.2f}%.",
            "",
            f"| Metric | {baseline_name} | {optimized_name} | Reduction |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("reload_demand_tokens", "KV reload demand tokens"),
        ("retrieved_tokens", "Actually retrieved tokens"),
        ("stored_tokens", "Stored tokens"),
        ("disk_load_allocation_failures", "Disk load allocation failures"),
    ):
        base = baseline_cache[key]
        opt = optimized_cache[key]
        lines.append(f"| {label} | {base} | {opt} | {_format_reduction(base, opt)} |")
    lines.append(
        f"| Disk resident GiB | {baseline_disk_bytes / 1024**3:.3f} "
        f"| {optimized_disk_bytes / 1024**3:.3f} "
        f"| {_format_reduction(baseline_disk_bytes, optimized_disk_bytes)} |"
    )

    lines.extend(
        [
            "",
            "> Reload demand is the KV token volume requested from LMCache after "
            "vLLM GPU-cache hits. Actual retrieval is reported separately so cache "
            "misses or allocation failures cannot look like a traffic improvement.",
            "",
            "## Logical ForkAttention KV Footprint",
            "",
            "| Baseline branch-local KV | Shared KV | Saved tokens | Saved GiB "
            "| Reduction |",
            "|---:|---:|---:|---:|---:|",
            f"| {int(optimized['baseline_unique_kv'])} "
            f"| {int(optimized['monowire_unique_kv'])} "
            f"| {int(optimized['kv_tokens_saved'])} "
            f"| {optimized['kv_gib_saved']:.3f} "
            f"| {optimized['kv_reduction_percent']:.2f}% |",
            "",
            "> This footprint estimate is workload-derived and independent of the "
            "LMCache eviction policy; use the reload table for the policy A/B result.",
            "",
        ]
    )
    return "\n".join(lines)


def _throughput(tokens: float, milliseconds: float) -> float:
    return tokens / (milliseconds / 1000) if milliseconds else 0


def _reduction(baseline: int | float, optimized: int | float) -> float:
    return 100 * (baseline - optimized) / baseline if baseline else 0


def _format_reduction(baseline: int | float, optimized: int | float) -> str:
    if baseline == 0:
        return "+0.00%" if optimized == 0 else "n/a"
    return f"{_reduction(baseline, optimized):+.2f}%"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("baseline_name")
    parser.add_argument("optimized_name")
    args = parser.parse_args(argv)
    report = render_report(args.root, args.baseline_name, args.optimized_name)
    output = args.root / "policy_comparison.md"
    output.write_text(report, encoding="utf-8")
    print(f"Wrote policy comparison to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
