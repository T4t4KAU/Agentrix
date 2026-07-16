from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from cacheblend_rag_report import summarize_run
from langgraph_e2e_report import memory_summary


def _prom_value(path: Path, name: str) -> float:
    if not path.is_file():
        return 0.0
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = rf"^(?:{re.escape(name)})(?:_total)?(?:\{{[^}}]*\}})?\s+(\S+)$"
    return sum(float(value) for value in re.findall(pattern, text, re.MULTILINE))


def _prom_delta(directory: Path, name: str) -> float:
    return max(
        0.0,
        _prom_value(directory / "metrics.prom", name)
        - _prom_value(directory / "metrics_before.prom", name),
    )


def _warning_count(directory: Path, pattern: str) -> int:
    path = directory / "measured_server.log"
    if not path.is_file():
        path = directory / "vllm_server.log"
    if not path.is_file():
        return 0
    return path.read_text(encoding="utf-8", errors="replace").count(pattern)


def _sample_summary(directory: Path) -> dict[str, float]:
    path = directory / "memory_samples.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    def values(name: str) -> list[float]:
        output = []
        for row in rows:
            try:
                output.append(float(row.get(name, "")))
            except (TypeError, ValueError):
                pass
        return output

    def peak(name: str) -> float:
        return max(values(name), default=0.0)

    def growth(name: str) -> float:
        items = values(name)
        return max(items, default=0.0) - min(items, default=0.0)

    def integrate_kib_per_second(name: str) -> float:
        total_kib = 0.0
        previous_time = None
        for row in rows:
            try:
                timestamp = float(row["unix_s"])
                rate = float(row[name])
            except (KeyError, TypeError, ValueError):
                continue
            if previous_time is not None:
                total_kib += rate * max(0.0, timestamp - previous_time)
            previous_time = timestamp
        return total_kib * 1024.0

    return {
        "cpu_cache_usage_peak_percent": 100.0 * peak(
            "vllm:kv_offload_cpu_cache_usage_perc"
        ),
        "cpu_cache_occupancy_peak_percent": 100.0 * peak(
            "vllm:kv_offload_cpu_cache_occupancy_perc"
        ),
        "process_rchar_growth_bytes": growth("process_rchar_bytes"),
        "process_wchar_growth_bytes": growth("process_wchar_bytes"),
        "process_read_growth_bytes": growth("process_read_bytes"),
        "process_write_growth_bytes": growth("process_write_bytes"),
        "fs_cache_files_peak": peak("fs_cache_files"),
        "fs_cache_bytes_peak": peak("fs_cache_bytes"),
        "pcie_rx_peak_mib_s": peak("pcie_rx_kib_s") / 1024.0,
        "pcie_tx_peak_mib_s": peak("pcie_tx_kib_s") / 1024.0,
        "pcie_rx_integral_bytes": integrate_kib_per_second("pcie_rx_kib_s"),
        "pcie_tx_integral_bytes": integrate_kib_per_second("pcie_tx_kib_s"),
    }


def summarize_variant(directory: Path) -> dict[str, Any]:
    payload = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    run = summarize_run(payload)
    memory = memory_summary(directory)
    samples = _sample_summary(directory)
    load_bytes = _prom_delta(directory, "vllm:kv_offload_load_bytes")
    store_bytes = _prom_delta(directory, "vllm:kv_offload_store_bytes")
    load_time = _prom_delta(directory, "vllm:kv_offload_load_time")
    store_time = _prom_delta(directory, "vllm:kv_offload_store_time")
    load_ops = _prom_delta(directory, "vllm:kv_offload_load_size_count")
    store_ops = _prom_delta(directory, "vllm:kv_offload_store_size_count")
    drain_path = directory / "measured_drain.json"
    drain = (
        json.loads(drain_path.read_text(encoding="utf-8"))
        if drain_path.is_file()
        else {}
    )
    drain_elapsed_s = float(drain.get("elapsed_s", 0.0))
    drain_confirmation_s = float(drain.get("confirmation_window_s", 0.0))
    drain_active_upper_s = max(0.0, drain_elapsed_s - drain_confirmation_s)
    admission_failures = _warning_count(directory, "cannot store blocks")
    run_complete = all(
        (directory / name).is_file()
        for name in ("metrics_before.prom", "metrics.prom", "measured_drain.json")
    )
    offload_variant = directory.name.endswith(("_cpu", "_tiered"))
    if not run_complete:
        offload_status = "incomplete"
    elif not offload_variant:
        offload_status = "not-configured"
    elif store_bytes <= 0:
        offload_status = "inactive"
    elif admission_failures:
        offload_status = "partial"
    else:
        offload_status = "active"
    return {
        **run,
        **memory,
        **samples,
        "offload_load_bytes": load_bytes,
        "offload_store_bytes": store_bytes,
        "offload_load_time_s": load_time,
        "offload_store_time_s": store_time,
        "offload_load_operations": load_ops,
        "offload_store_operations": store_ops,
        "offload_admission_failures": admission_failures,
        "offload_status": offload_status,
        "run_complete": run_complete,
        "offload_drain_elapsed_s": drain_elapsed_s,
        "offload_drain_active_upper_s": drain_active_upper_s,
        "wall_with_drain_ms": run["wall_ms"] + drain_active_upper_s * 1000.0,
        "offload_load_average_mib": (
            load_bytes / load_ops / 2**20 if load_ops else 0.0
        ),
        "offload_store_average_mib": (
            store_bytes / store_ops / 2**20 if store_ops else 0.0
        ),
    }


def build_report(root: Path) -> dict[str, Any]:
    variants = {
        directory.name: summarize_variant(directory)
        for directory in sorted(root.iterdir())
        if directory.is_dir() and (directory / "run.json").is_file()
    }
    return {"schema_version": 1, "variants": variants}


def _gib(value: float) -> str:
    return f"{value / 2**30:.3f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# HotpotQA Tiered-Offload Comparison",
        "",
        "| Variant | Status | Admission failures | Request wall s | Wall + drain s | GPU peak MiB | KV live tokens | CPU cache peak | CPU→GPU GiB | GPU→CPU GiB | Load time s | Store time s | Disk GiB | Disk read GiB | Disk write GiB | RSS MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in report["variants"].items():
        lines.append(
            f"| {name} | {item['offload_status']} | "
            f"{item['offload_admission_failures']} | "
            f"{item['wall_ms'] / 1000:.3f} | "
            f"{item['wall_with_drain_ms'] / 1000:.3f} | "
            f"{item['gpu_peak_used_mib']:.0f} | "
            f"{item['kv_cache_peak_live_tokens']:.0f} | "
            f"{item.get('cpu_cache_occupancy_peak_percent', 0):.1f}% | "
            f"{_gib(item['offload_load_bytes'])} | "
            f"{_gib(item['offload_store_bytes'])} | "
            f"{item['offload_load_time_s']:.3f} | "
            f"{item['offload_store_time_s']:.3f} | "
            f"{_gib(item.get('fs_cache_bytes_peak', 0))} | "
            f"{_gib(item.get('process_read_growth_bytes', 0))} | "
            f"{_gib(item.get('process_write_growth_bytes', 0))} | "
            f"{item['server_tree_rss_peak_mib']:.0f} |"
        )
    paired = (
        ("No offload", "baseline", "forkattention"),
        ("Two level", "flash_original_cpu", "fork_cpu"),
        ("Three level", "flash_original_tiered", "fork_tiered"),
    )
    if all(
        flash_name in report["variants"] and fork_name in report["variants"]
        for _, flash_name, fork_name in paired
    ):
        lines.extend(
            [
                "",
                "## FlashAttention vs ForkAttention by tier",
                "",
                "Flash uses the original offload policy; Fork uses the fanout-optimized policy for the two-level and three-level rows.",
                "",
                "| Tier | Validity | Flash wall s | Fork wall s | Fork speedup | Flash GPU MiB | Fork GPU MiB | Flash GPU→CPU GiB | Fork GPU→CPU GiB | Flash RSS MiB | Fork RSS MiB |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for tier, flash_name, fork_name in paired:
            flash = report["variants"][flash_name]
            fork = report["variants"][fork_name]
            statuses = {flash["offload_status"], fork["offload_status"]}
            valid = tier == "No offload" or statuses == {"active"}
            speedup = (
                f"{flash['wall_ms'] / fork['wall_ms']:.3f}x" if valid else "invalid"
            )
            lines.append(
                f"| {tier} | {'valid' if valid else 'invalid'} | "
                f"{flash['wall_ms'] / 1000:.3f} | "
                f"{fork['wall_ms'] / 1000:.3f} | "
                f"{speedup} | "
                f"{flash['gpu_peak_used_mib']:.0f} | "
                f"{fork['gpu_peak_used_mib']:.0f} | "
                f"{_gib(flash['offload_store_bytes'])} | "
                f"{_gib(fork['offload_store_bytes'])} | "
                f"{flash['server_tree_rss_peak_mib']:.0f} | "
                f"{fork['server_tree_rss_peak_mib']:.0f} |"
            )
    lines.extend(
        [
            "",
            "## Transfer operation detail",
            "",
            "| Variant | CPU→GPU ops | CPU→GPU avg MiB | GPU→CPU ops | GPU→CPU avg MiB |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in report["variants"].items():
        lines.append(
            f"| {name} | {item['offload_load_operations']:.0f} | "
            f"{item['offload_load_average_mib']:.3f} | "
            f"{item['offload_store_operations']:.0f} | "
            f"{item['offload_store_average_mib']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## PCIe hardware samples",
            "",
            "The integrated values are estimates from sampled instantaneous NVML rates; connector counters above are authoritative for KV bytes.",
            "",
            "| Variant | PCIe RX peak MiB/s | PCIe TX peak MiB/s | PCIe RX sampled GiB | PCIe TX sampled GiB |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, item in report["variants"].items():
        lines.append(
            f"| {name} | {item.get('pcie_rx_peak_mib_s', 0):.1f} | "
            f"{item.get('pcie_tx_peak_mib_s', 0):.1f} | "
            f"{_gib(item.get('pcie_rx_integral_bytes', 0))} | "
            f"{_gib(item.get('pcie_tx_integral_bytes', 0))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.root)
    (args.root / "offload_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(report)
    (args.root / "offload_comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
