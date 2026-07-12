from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from lmcache_policy_report import parse_lmcache_log, read_results, read_smoke_summary


VARIANTS = (
    ("no_offload", "No offload"),
    ("native_cpu", "Native CPU"),
    ("lmcache_default_cpu", "LMCache default CPU (LRU)"),
    ("lmcache_cpu", "LMCache fork-aware CPU"),
    ("lmcache_tiered", "LMCache CPU+disk"),
    ("flash_no_offload", "FlashAttention no offload"),
    ("flash_native_cpu", "FlashAttention + native CPU"),
)


def collect_variant(root: Path, name: str) -> dict[str, float]:
    backend = "flash_attn" if name.startswith("flash_") else "fork_attn"
    backend_root = root / name / backend
    result = read_results(backend_root / "benchmark_results.csv")
    profile_path = backend_root / "server_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    storage = read_smoke_summary(root / name / "storage_summary.txt")
    log_path = backend_root / "vllm_server.log"
    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    kv_bytes_per_token = result["kv_bytes_per_token"]

    if name.startswith("lmcache_"):
        cache = parse_lmcache_log(log_path)
        load_bytes = cache["retrieved_tokens"] * kv_bytes_per_token
        store_bytes = cache["stored_tokens"] * kv_bytes_per_token
        disk_load_failures = cache["disk_load_allocation_failures"]
    else:
        load_bytes = float(profile.get("kv_offload_load_bytes", 0))
        store_bytes = float(profile.get("kv_offload_store_bytes", 0))
        disk_load_failures = 0
    load_failures = len(re.findall(r"failed to load \d+ tokens", log_text))

    return {
        **result,
        "load_gib": load_bytes / 1024**3,
        "store_gib": store_bytes / 1024**3,
        "disk_gib": int(storage.get("disk_bytes", 0)) / 1024**3,
        "load_failures": float(load_failures),
        "disk_load_failures": float(disk_load_failures),
    }


def render_report(root: Path) -> str:
    rows = [(name, label, collect_variant(root, name)) for name, label in VARIANTS]
    by_name = {name: row for name, _, row in rows}
    baseline = rows[0][2]
    lines = [
        "# ForkAttention Offload Backend Comparison",
        "",
        "Baseline: ForkAttention without KV offloading.",
        "",
        "## End-to-End Performance",
        "",
        "| Variant | E2E wall ms | E2E tok/s | vs baseline | Branch tok/s "
        "| vs baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, label, row in rows:
        e2e_delta = _improvement(
            baseline["end_to_end_output_tokens_per_s"],
            row["end_to_end_output_tokens_per_s"],
        )
        branch_delta = _improvement(
            baseline["branch_output_tokens_per_s"],
            row["branch_output_tokens_per_s"],
        )
        lines.append(
            f"| {label} | {row['case_wall_time_ms']:.3f} "
            f"| {row['end_to_end_output_tokens_per_s']:.3f} | {e2e_delta:+.2f}% "
            f"| {row['branch_output_tokens_per_s']:.3f} "
            f"| {branch_delta:+.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Pairwise Offload Impact",
            "",
            "| Comparison | E2E tok/s | Branch tok/s |",
            "|---|---:|---:|",
        ]
    )
    for label, reference_name, candidate_name in (
        ("ForkAttention native CPU vs no offload", "no_offload", "native_cpu"),
        (
            "LMCache default CPU vs ForkAttention no offload",
            "no_offload",
            "lmcache_default_cpu",
        ),
        (
            "LMCache fork-aware CPU vs default CPU",
            "lmcache_default_cpu",
            "lmcache_cpu",
        ),
        ("LMCache CPU+disk vs fork-aware CPU", "lmcache_cpu", "lmcache_tiered"),
        (
            "FlashAttention native CPU vs no offload",
            "flash_no_offload",
            "flash_native_cpu",
        ),
    ):
        reference = by_name[reference_name]
        candidate = by_name[candidate_name]
        lines.append(
            f"| {label} "
            f"| {_improvement(reference['end_to_end_output_tokens_per_s'], candidate['end_to_end_output_tokens_per_s']):+.2f}% "
            f"| {_improvement(reference['branch_output_tokens_per_s'], candidate['branch_output_tokens_per_s']):+.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Logical KV Cache Footprint",
            "",
            "| FlashAttention branch-local KV | ForkAttention shared KV | "
            "Saved tokens | Saved GiB | Reduction |",
            "|---:|---:|---:|---:|---:|",
            f"| {int(baseline['baseline_unique_kv'])} "
            f"| {int(baseline['monowire_unique_kv'])} "
            f"| {int(baseline['kv_tokens_saved'])} "
            f"| {baseline['kv_gib_saved']:.3f} "
            f"| {baseline['kv_reduction_percent']:.2f}% |",
            "",
            "> This is the logical KV footprint required by the common-prefix "
            "workload. It is independent of the selected offload backend.",
            "",
            "## KV Movement and Storage",
            "",
            "| Variant | KV load GiB | KV store GiB | Disk GiB | Load failures "
            "| Disk allocation failures |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, label, row in rows:
        lines.append(
            f"| {label} | {row['load_gib']:.3f} | {row['store_gib']:.3f} "
            f"| {row['disk_gib']:.3f} | {int(row['load_failures'])} "
            f"| {int(row['disk_load_failures'])} |"
        )
    lines.extend(
        [
            "",
            "> Native traffic uses vLLM Prometheus byte counters. LMCache traffic "
            "uses retrieved/stored token counts multiplied by model KV bytes per "
            "token. All variants use the same model, trace, concurrency, GPU "
            "allocation, and CPU capacity.",
            "",
        ]
    )
    return "\n".join(lines)


def _improvement(baseline: float, candidate: float) -> float:
    return 100 * (candidate / baseline - 1) if baseline else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    report = render_report(args.root)
    output = args.root / "offload_comparison.md"
    output.write_text(report, encoding="utf-8")
    print(f"Wrote offload comparison to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
