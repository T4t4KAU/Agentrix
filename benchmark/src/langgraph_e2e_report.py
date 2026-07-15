from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from cacheblend_rag_report import summarize_run, summarize_runtime_logs


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def quality_summary(payload: dict[str, Any]) -> dict[str, Any]:
    events = payload.get("events", [])
    tool_events = [event for event in events if event.get("stage") == "tool_select"]
    reducer_events = [event for event in events if event.get("stage") == "reduce"]
    valid_calls = 0
    for event in tool_events:
        calls = event.get("response", {}).get("tool_calls", [])
        if len(calls) != 1 or calls[0].get("function", {}).get("name") != "rag_search":
            continue
        try:
            arguments = json.loads(calls[0]["function"].get("arguments", ""))
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(arguments.get("query"), str) and arguments["query"].strip():
            valid_calls += 1
    nonempty_reducers = sum(
        bool(event.get("response", {}).get("content", "").strip())
        for event in reducer_events
    )
    return {
        "tool_calls": len(tool_events),
        "valid_tool_calls": valid_calls,
        "valid_tool_call_rate": valid_calls / len(tool_events) if tool_events else 0.0,
        "reducers": len(reducer_events),
        "nonempty_reducers": nonempty_reducers,
        "reducer_success_rate": (
            nonempty_reducers / len(reducer_events) if reducer_events else 0.0
        ),
    }


def output_lexical_overlap(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> float | None:
    """Bag-of-words F1 over paired reducer outputs; a drift guardrail only."""
    left = {
        str(item.get("case_id")): str(item.get("answer", ""))
        for item in baseline.get("outputs", [])
    }
    right = {
        str(item.get("case_id")): str(item.get("answer", ""))
        for item in candidate.get("outputs", [])
    }
    common = left.keys() & right.keys()
    if not common:
        return None
    scores = []
    for case_id in common:
        left_tokens = Counter(re.findall(r"\w+", left[case_id].lower()))
        right_tokens = Counter(re.findall(r"\w+", right[case_id].lower()))
        overlap = sum((left_tokens & right_tokens).values())
        denominator = sum(left_tokens.values()) + sum(right_tokens.values())
        scores.append(2 * overlap / denominator if denominator else 1.0)
    return sum(scores) / len(scores)


def fork_metrics(path: Path, before_path: Path | None = None) -> dict[str, Any]:
    def total(metric_path: Path, name: str) -> float:
        text = (
            metric_path.read_text(encoding="utf-8", errors="replace")
            if metric_path.is_file()
            else ""
        )
        return sum(
            float(value)
            for value in re.findall(
                rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+(\S+)$", text, re.MULTILINE
            )
        )

    def delta(name: str) -> float:
        before = total(before_path, name) if before_path is not None else 0.0
        return max(total(path, name) - before, 0.0)

    observed = delta("vllm:fork_attention_observed_steps_total")
    active = delta("vllm:fork_attention_active_steps_total")
    shared_ctas = delta("vllm:fork_attention_shared_ctas_total")
    singleton_ctas = delta("vllm:fork_attention_singleton_ctas_total")
    return {
        "observed_steps": observed,
        "active_steps": active,
        "activation_rate": active / observed if observed else 0.0,
        "shared_ctas": shared_ctas,
        "singleton_ctas": singleton_ctas,
        "warmup_subtracted": before_path is not None and before_path.is_file(),
    }


def memory_summary(directory: Path) -> dict[str, Any]:
    samples_path = directory / "memory_samples.csv"
    rows: list[dict[str, str]] = []
    if samples_path.is_file():
        with samples_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

    def values(name: str) -> list[float]:
        output = []
        for row in rows:
            try:
                output.append(float(row.get(name, 0)))
            except (TypeError, ValueError):
                continue
        return output

    def snapshot(name: str) -> float:
        path = directory / name
        if not path.is_file():
            return 0.0
        fields = path.read_text(encoding="utf-8").strip().split(",")
        try:
            return float(fields[1].strip())
        except (IndexError, ValueError):
            return 0.0

    metrics_path = directory / "metrics.prom"
    metrics_text = (
        metrics_path.read_text(encoding="utf-8", errors="replace")
        if metrics_path.is_file()
        else ""
    )
    capacity = re.search(r'kv_cache_size_tokens="(\d+)"', metrics_text)
    capacity_tokens = int(capacity.group(1)) if capacity else 0
    gpu_used = values("gpu_used_mib")
    kv_usage = values("vllm:kv_cache_usage_perc")
    local_cache = values("lmcache:local_cache_usage")
    remote_cache = values("lmcache:remote_cache_usage")
    process_rss = values("process_resident_memory_bytes")
    server_tree_rss = values("server_tree_rss_bytes")
    memory_util = values("memory_controller_util_pct")
    idle_used = snapshot("gpu_before_server.csv")
    warm_used = snapshot("gpu_after_warm.csv")
    peak_gpu = max(gpu_used, default=0.0)
    peak_kv = max(kv_usage, default=0.0)
    local_metric_available = "lmcache:local_cache_usage" in metrics_text
    remote_metric_available = "lmcache:remote_cache_usage" in metrics_text
    return {
        "samples": len(rows),
        "gpu_idle_used_mib": idle_used,
        "gpu_after_warm_used_mib": warm_used,
        "gpu_peak_used_mib": peak_gpu,
        "gpu_peak_delta_from_idle_mib": max(peak_gpu - idle_used, 0.0),
        "gpu_peak_delta_from_warm_mib": max(peak_gpu - warm_used, 0.0),
        "kv_cache_capacity_tokens": capacity_tokens,
        "kv_cache_peak_usage": peak_kv,
        "kv_cache_peak_live_tokens": round(capacity_tokens * peak_kv),
        "frontend_rss_peak_mib": max(process_rss, default=0.0) / 2**20,
        "server_tree_rss_peak_mib": max(server_tree_rss, default=0.0) / 2**20,
        "lmcache_local_metric_available": local_metric_available,
        "lmcache_local_peak_mib": (
            max(local_cache, default=0.0) / 2**20
            if local_metric_available
            else None
        ),
        "lmcache_remote_metric_available": remote_metric_available,
        "lmcache_remote_peak_mib": (
            max(remote_cache, default=0.0) / 2**20
            if remote_metric_available
            else None
        ),
        "memory_controller_peak_pct": max(memory_util, default=0.0),
    }


def build_report(root: Path) -> dict[str, Any]:
    baseline_payload = _read(root / "baseline" / "run.json")
    payloads = {"baseline": baseline_payload}
    variants = (
        "baseline_compact",
        "forkattention",
        "forkattention_compact",
        "cacheblend",
        "cacheblend_compact",
    )
    for variant in variants:
        path = root / variant / "run.json"
        if path.is_file():
            payloads[variant] = _read(path)
    runs = {name: summarize_run(payload) for name, payload in payloads.items()}
    baseline = runs["baseline"]

    def comparison(candidate: dict[str, Any]) -> dict[str, float]:
        return {
            "wall_speedup": baseline["wall_ms"] / candidate["wall_ms"],
            "request_throughput_gain": (
                candidate["request_throughput"] / baseline["request_throughput"]
            ),
            "prompt_token_throughput_gain": (
                candidate["prompt_token_throughput"]
                / baseline["prompt_token_throughput"]
            ),
        }

    report = {
        "schema_version": 1,
        "experiment": {
            "tasks": int(baseline_payload.get("metadata", {}).get("cases", 0)),
            "branches": sum(
                baseline_payload.get("metadata", {}).get("branches_per_case", [])
            ),
            "baseline": "LangGraph + vLLM FLASH_ATTN",
            "forkattention": "LangGraph + Agentrix FORK_ATTN",
            "cacheblend": "LangGraph + vLLM FLASH_ATTN + LMCache CacheBlend",
            "mode": "live end-to-end",
        },
        "runs": runs,
        "quality": {},
        "prompt_compaction": {
            name: payload.get("metadata", {}).get(
                "prompt_compaction_report", {}
            )
            for name, payload in payloads.items()
        },
        "memory": {
            name: memory_summary(root / name) for name in payloads
        },
        "rag_reuse": baseline_payload.get("metadata", {}).get("rag_reuse", {}),
        "comparison": {
            name: comparison(summary)
            for name, summary in runs.items()
            if name != "baseline"
        },
    }
    for name, payload in payloads.items():
        report["quality"][name] = {
            **quality_summary(payload),
            "reducer_lexical_f1_vs_baseline": (
                1.0
                if name == "baseline"
                else output_lexical_overlap(baseline_payload, payload)
            ),
        }
    report["ablation"] = {}
    for compact_name in (
        "baseline_compact",
        "forkattention_compact",
        "cacheblend_compact",
    ):
        raw_name = compact_name.removesuffix("_compact")
        if compact_name not in runs or raw_name not in runs:
            continue
        raw, compact = runs[raw_name], runs[compact_name]
        report["ablation"][compact_name] = {
            "wall_speedup_vs_uncompacted": raw["wall_ms"] / compact["wall_ms"],
            "prompt_token_reduction": (
                1 - compact["prompt_tokens"] / raw["prompt_tokens"]
                if raw["prompt_tokens"]
                else 0.0
            ),
            "reducer_lexical_f1_vs_uncompacted": output_lexical_overlap(
                payloads[raw_name], payloads[compact_name]
            ),
        }
    cache_variants = [name for name in payloads if name.startswith("cacheblend")]
    if cache_variants:
        report["lmcache"] = {
            name: summarize_runtime_logs([root / name / "measured_server.log"])
            for name in cache_variants
        }
    fork_variants = [name for name in payloads if name.startswith("forkattention")]
    if fork_variants:
        report["fork_metrics"] = {
            name: fork_metrics(
                root / name / "metrics.prom",
                root / name / "metrics_before.prom",
            )
            for name in fork_variants
        }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    experiment = report["experiment"]
    runs = report["runs"]
    quality = report["quality"]
    comparison = report["comparison"]
    reuse = report["rag_reuse"]
    labels = {
        "baseline": "baseline",
        "baseline_compact": "baseline + compaction",
        "forkattention": "ForkAttention",
        "forkattention_compact": "ForkAttention + compaction",
        "cacheblend": "CacheBlend",
        "cacheblend_compact": "CacheBlend + compaction",
    }
    lines = [
        f"# LangGraph {experiment['tasks']}-Case Agent End-to-End Result",
        "",
        f"- Tasks: {experiment['tasks']} distinct tasks; {experiment['branches']} total branches",
        "- Execution: live planner → local RAG tool → branch answer → reducer",
        "",
        "| Variant | Wall time | Speedup | Prompt tokens | Saved chars | Reducer lexical F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, run in runs.items():
        speedup = 1.0 if name == "baseline" else comparison[name]["wall_speedup"]
        item_quality = quality[name]
        compact = report["prompt_compaction"].get(name, {})
        lexical = item_quality.get("reducer_lexical_f1_vs_baseline")
        if name in report.get("ablation", {}):
            lexical = report["ablation"][name].get(
                "reducer_lexical_f1_vs_uncompacted"
            )
        row = (
            f"| {labels[name]} | {run['wall_ms'] / 1000:.3f} s "
            f"| {speedup:.2f}x | {run['prompt_tokens']} "
            f"| {compact.get('saved_chars', 0)} "
            f"| {lexical:.3f} |"
            if lexical is not None
            else f"| {labels[name]} | {run['wall_ms'] / 1000:.3f} s "
            f"| {speedup:.2f}x | {run['prompt_tokens']} "
            f"| {compact.get('saved_chars', 0)} | n/a |"
        )
        lines.append(row)
    lines.extend(
        [
            "",
            f"RAG chunk reuse was {reuse.get('reuse_ratio', 0) * 100:.1f}% across task bootstraps.",
        ]
    )
    lines.extend(
        [
            "",
            "| Variant | GPU peak | Server increment | Runtime increment | KV peak | Peak live KV tokens | Process-tree RSS sum | LMCache local peak |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in runs:
        memory = report["memory"][name]
        lmcache_peak = memory["lmcache_local_peak_mib"]
        lmcache_text = f"{lmcache_peak:.1f} MiB" if lmcache_peak is not None else "n/a"
        lines.append(
            f"| {labels[name]} | {memory['gpu_peak_used_mib']:.0f} MiB "
            f"| {memory['gpu_peak_delta_from_idle_mib']:.0f} MiB "
            f"| {memory['gpu_peak_delta_from_warm_mib']:.0f} MiB "
            f"| {memory['kv_cache_peak_usage'] * 100:.1f}% "
            f"| {memory['kv_cache_peak_live_tokens']} "
            f"| {memory['server_tree_rss_peak_mib']:.0f} MiB "
            f"| {lmcache_text} |"
        )
    for name, item in report.get("ablation", {}).items():
        lines.append(
            f"{labels[name]} versus its uncompacted control: "
            f"{item['wall_speedup_vs_uncompacted']:.2f}x wall speedup, "
            f"{item['prompt_token_reduction'] * 100:.1f}% fewer prompt tokens."
        )
    if "lmcache" in report:
        for name, lmcache in report["lmcache"].items():
            lines.append(
                f"{labels[name]} retrieved {lmcache['retrieved_tokens']} tokens "
                f"with a {lmcache['lookup_hit_ratio'] * 100:.1f}% lookup hit ratio."
            )
        lines.extend(
            [
                "",
                "> ForkAttention and CacheBlend are measured separately because "
                "the current LMCache blender rejects ForkAttentionImpl; no "
                "combined-speedup claim is made.",
            ]
        )
    if "fork_metrics" in report:
        for name, physical in report["fork_metrics"].items():
            scope = "measured" if physical["warmup_subtracted"] else "process"
            lines.append(
                f"{labels[name]} physically activated on "
                f"{physical['active_steps']:.0f}/{physical['observed_steps']:.0f} "
                f"{scope} steps ({physical['activation_rate'] * 100:.1f}%), with "
                f"{physical['shared_ctas']:.0f} shared and "
                f"{physical['singleton_ctas']:.0f} singleton CTA plan entries."
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.root)
    (args.root / "comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    markdown = render_markdown(report)
    (args.root / "comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
