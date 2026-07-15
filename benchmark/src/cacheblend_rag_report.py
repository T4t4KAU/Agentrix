from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import statistics
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return ordered[index]


def summarize_run(payload: dict[str, Any]) -> dict[str, Any]:
    events = payload.get("events", [])
    latencies = [float(event["latency_ms"]) for event in events]
    prompt_tokens = sum(
        int(event.get("usage", {}).get("prompt_tokens", 0)) for event in events
    )
    completion_tokens = sum(
        int(event.get("usage", {}).get("completion_tokens", 0)) for event in events
    )
    wall_ms = float(payload["metadata"]["wall_ms"])
    return {
        "requests": len(events),
        "wall_ms": wall_ms,
        "request_throughput": len(events) * 1000 / wall_ms if wall_ms else 0.0,
        "prompt_tokens": prompt_tokens,
        "prompt_token_throughput": (
            prompt_tokens * 1000 / wall_ms if wall_ms else 0.0
        ),
        "completion_tokens": completion_tokens,
        "latency_mean_ms": statistics.fmean(latencies) if latencies else 0.0,
        "latency_p50_ms": _percentile(latencies, 0.5),
        "latency_p95_ms": _percentile(latencies, 0.95),
    }


def _normalized_response(event: dict[str, Any]) -> dict[str, Any]:
    response = event.get("response", {})
    calls = []
    for call in response.get("tool_calls", []):
        function = call.get("function", {})
        calls.append(
            {
                "name": function.get("name"),
                "arguments": function.get("arguments"),
            }
        )
    return {"content": response.get("content") or "", "tool_calls": calls}


def response_match_rate(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> float | None:
    def indexed(payload: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
        return {
            (
                event.get("case_id"),
                event.get("stage"),
                event.get("branch_id"),
                event.get("source_started_ms"),
            ): _normalized_response(event)
            for event in payload.get("events", [])
        }

    left, right = indexed(baseline), indexed(candidate)
    if not left or left.keys() != right.keys():
        return None
    return sum(left[key] == right[key] for key in left) / len(left)


def response_lexical_overlap(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> float | None:
    """Return mean bag-of-words F1 for paired textual responses.

    This is intentionally a drift diagnostic, not a task-quality metric.  Tool
    calls are evaluated separately because their JSON arguments should not be
    treated as prose.
    """

    def indexed(payload: dict[str, Any]) -> dict[tuple[Any, ...], str]:
        return {
            (
                event.get("case_id"),
                event.get("stage"),
                event.get("branch_id"),
                event.get("source_started_ms"),
            ): (event.get("response", {}).get("content") or "")
            for event in payload.get("events", [])
            if event.get("response", {}).get("content")
        }

    left, right = indexed(baseline), indexed(candidate)
    common = left.keys() & right.keys()
    if not common:
        return None
    scores = []
    for key in common:
        left_tokens = Counter(re.findall(r"\w+", left[key].lower()))
        right_tokens = Counter(re.findall(r"\w+", right[key].lower()))
        overlap = sum((left_tokens & right_tokens).values())
        denominator = sum(left_tokens.values()) + sum(right_tokens.values())
        scores.append(2 * overlap / denominator if denominator else 1.0)
    return statistics.fmean(scores)


def valid_rag_tool_call_rate(payload: dict[str, Any]) -> float | None:
    events = [
        event for event in payload.get("events", []) if event.get("stage") == "tool_select"
    ]
    if not events:
        return None
    valid = 0
    for event in events:
        calls = event.get("response", {}).get("tool_calls", [])
        if len(calls) != 1:
            continue
        function = calls[0].get("function", {})
        try:
            arguments = json.loads(function.get("arguments", ""))
        except (TypeError, json.JSONDecodeError):
            continue
        top_k = arguments.get("top_k", 3)
        if (
            function.get("name") == "rag_search"
            and isinstance(arguments.get("query"), str)
            and bool(arguments["query"].strip())
            and isinstance(top_k, int)
            and 1 <= top_k <= 5
        ):
            valid += 1
    return valid / len(events)


def _speedup(baseline: float, candidate: float) -> float:
    return baseline / candidate if candidate else 0.0


def _median_summary(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [summarize_run(payload) for payload in payloads]
    if not summaries:
        return {}
    return {
        key: statistics.median(float(summary[key]) for summary in summaries)
        for key in summaries[0]
    }


def summarize_runtime_logs(paths: list[Path]) -> dict[str, Any]:
    negative_pin_warnings = 0
    total_tokens = 0
    hit_tokens = 0
    retrieved_tokens = 0
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        negative_pin_warnings += sum(
            "Pin count" in line or "Double unpin" in line
            for line in text.splitlines()
        )
        # The adapter can log the same deferred request on multiple scheduler
        # passes.  Count each request id once so the ratio is request-weighted
        # by tokens rather than scheduler iterations.
        requests = {
            request_id: (total, hit)
            for request_id, total, hit in re.findall(
                r"Reqid:\s+([^,]+),\s+Total tokens\s+(\d+).*?"
                r"LMCache hit tokens:\s+(\d+)",
                text,
            )
        }
        for total, hit in requests.values():
            total_tokens += int(total)
            hit_tokens += int(hit)
        for retrieved in re.findall(r"Retrieved\s+(\d+)\s+out of", text):
            retrieved_tokens += int(retrieved)
    return {
        "negative_pin_warnings": negative_pin_warnings,
        "lookup_total_tokens": total_tokens,
        "lookup_hit_tokens": hit_tokens,
        "lookup_hit_ratio": hit_tokens / total_tokens if total_tokens else 0.0,
        "retrieved_tokens": retrieved_tokens,
    }


def collect(root: Path) -> dict[str, Any]:
    scenarios: dict[str, Any] = {}
    for scenario_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        trace_path = scenario_dir / "trace.json"
        baseline_paths = sorted((scenario_dir / "baseline").glob("run*.json"))
        blend_paths = sorted((scenario_dir / "cacheblend").glob("run*.json"))
        if not trace_path.is_file() or not baseline_paths or not blend_paths:
            continue
        trace = _read(trace_path)
        baseline_payloads = [_read(path) for path in baseline_paths]
        blend_payloads = [_read(path) for path in blend_paths]
        base = _median_summary(baseline_payloads)
        candidate = _median_summary(blend_payloads)
        baseline_diagnostics = summarize_runtime_logs(
            sorted((scenario_dir / "baseline").glob("repeat*/measured_server.log"))
        )
        blend_diagnostics = summarize_runtime_logs(
            sorted((scenario_dir / "cacheblend").glob("repeat*/measured_server.log"))
        )
        pair_count = min(len(baseline_payloads), len(blend_payloads))
        match_rates = [
            response_match_rate(baseline_payloads[index], blend_payloads[index])
            for index in range(pair_count)
        ]
        valid_match_rates = [rate for rate in match_rates if rate is not None]
        lexical_rates = [
            response_lexical_overlap(baseline_payloads[index], blend_payloads[index])
            for index in range(pair_count)
        ]
        valid_lexical_rates = [rate for rate in lexical_rates if rate is not None]
        tool_call_rates = [
            valid_rag_tool_call_rate(payload) for payload in blend_payloads
        ]
        valid_tool_call_rates = [rate for rate in tool_call_rates if rate is not None]
        scenarios[scenario_dir.name] = {
            "rag_reuse": trace.get("metadata", {}).get("rag_reuse", {}),
            "repeat_count": pair_count,
            "median_runs": {"baseline": base, "cacheblend": candidate},
            "runtime_diagnostics": {
                "baseline": baseline_diagnostics,
                "cacheblend": blend_diagnostics,
                "production_ready": blend_diagnostics["negative_pin_warnings"] == 0,
            },
            "comparison": {
                "wall_speedup": _speedup(base["wall_ms"], candidate["wall_ms"]),
                "throughput_gain": (
                    candidate["request_throughput"] / base["request_throughput"]
                    if base["request_throughput"]
                    else 0.0
                ),
                "p50_latency_speedup": _speedup(
                    base["latency_p50_ms"], candidate["latency_p50_ms"]
                ),
                "p95_latency_speedup": _speedup(
                    base["latency_p95_ms"], candidate["latency_p95_ms"]
                ),
                "exact_response_match_rate": (
                    statistics.fmean(valid_match_rates)
                    if valid_match_rates
                    else None
                ),
                "lexical_response_overlap": (
                    statistics.fmean(valid_lexical_rates)
                    if valid_lexical_rates
                    else None
                ),
                "valid_rag_tool_call_rate": (
                    statistics.fmean(valid_tool_call_rates)
                    if valid_tool_call_rates
                    else None
                ),
            },
        }
    return {"schema_version": 1, "scenarios": scenarios}


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CacheBlend LangGraph RAG Comparison",
        "",
        "Baseline and CacheBlend use the same captured request trace. Every "
        "measured repeat starts a fresh server and an empty business cache after "
        "an unrelated kernel warmup. The table reports the median repeat.",
        "",
        "| Scenario | RAG reuse | Reordered pairs | Baseline wall (s) | CacheBlend wall (s) | Wall speedup | P50 speedup | LMCache hit ratio | Retrieved tokens | Lexical overlap | Valid RAG calls | Runtime status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, item in report["scenarios"].items():
        reuse = item["rag_reuse"]
        runs = item["median_runs"]
        comparison = item["comparison"]
        lexical = comparison["lexical_response_overlap"]
        tool_calls = comparison["valid_rag_tool_call_rate"]
        diagnostics = item["runtime_diagnostics"]
        blend_runtime = diagnostics["cacheblend"]
        status = (
            "ok"
            if diagnostics["production_ready"]
            else f"exploratory ({diagnostics['cacheblend']['negative_pin_warnings']} pin warnings)"
        )
        lexical_text = "n/a" if lexical is None else f"{lexical * 100:.1f}%"
        tool_call_text = "n/a" if tool_calls is None else f"{tool_calls * 100:.1f}%"
        lines.append(
            f"| {name} | {reuse.get('reuse_ratio', 0) * 100:.1f}% "
            f"| {reuse.get('reordered_pairs', 0)} "
            f"| {runs['baseline']['wall_ms'] / 1000:.3f} "
            f"| {runs['cacheblend']['wall_ms'] / 1000:.3f} "
            f"| {comparison['wall_speedup']:.2f}x "
            f"| {comparison['p50_latency_speedup']:.2f}x "
            f"| {blend_runtime['lookup_hit_ratio'] * 100:.1f}% "
            f"| {blend_runtime['retrieved_tokens']} "
            f"| {lexical_text} | {tool_call_text} | {status} |"
        )
    lines.extend(
        [
            "",
            "> Lexical overlap and valid tool calls are drift diagnostics, not a "
            "semantic task-quality score. The JSON report also retains strict exact "
            "match. CacheBlend selective recomputation still requires task-level "
            "quality evaluation before deployment.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report CacheBlend RAG A/B results")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)
    report = collect(args.root)
    json_output = args.json_output or args.root / "comparison.json"
    markdown_output = args.markdown_output or args.root / "comparison.md"
    json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
