from __future__ import annotations

import json
from pathlib import Path

from langgraph_e2e_report import (
    build_report,
    fork_metrics,
    memory_summary,
    render_markdown,
)


def _run(wall_ms: float) -> dict:
    return {
        "metadata": {
            "wall_ms": wall_ms,
            "cases": 1,
            "branches_per_case": [1],
            "rag_reuse": {"reuse_ratio": 0.5},
            "prompt_compaction_report": {"saved_chars": 10},
        },
        "events": [
            {
                "stage": "tool_select",
                "latency_ms": wall_ms / 2,
                "usage": {"prompt_tokens": 100, "completion_tokens": 5},
                "response": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "rag_search",
                                "arguments": '{"query":"evidence"}',
                            }
                        }
                    ]
                },
            },
            {
                "stage": "reduce",
                "latency_ms": wall_ms / 2,
                "usage": {"prompt_tokens": 100, "completion_tokens": 5},
                "response": {"content": "answer"},
            },
        ],
        "outputs": [{"case_id": "case-0", "answer": "answer"}],
    }


def test_build_live_e2e_report(tmp_path: Path) -> None:
    for variant, wall_ms in (
        ("baseline", 2000),
        ("baseline_compact", 1500),
        ("forkattention", 1000),
        ("cacheblend", 500),
    ):
        directory = tmp_path / variant
        directory.mkdir()
        (directory / "run.json").write_text(json.dumps(_run(wall_ms)), encoding="utf-8")
        (directory / "measured_server.log").write_text("", encoding="utf-8")

    report = build_report(tmp_path)

    assert report["comparison"]["forkattention"]["wall_speedup"] == 2
    assert report["comparison"]["cacheblend"]["wall_speedup"] == 4
    assert report["ablation"]["baseline_compact"][
        "wall_speedup_vs_uncompacted"
    ] == 4 / 3
    assert report["ablation"]["baseline_compact"][
        "reducer_lexical_f1_vs_uncompacted"
    ] == 1
    assert report["quality"]["baseline_compact"][
        "reducer_lexical_f1_vs_baseline"
    ] == 1
    assert report["quality"]["cacheblend"]["valid_tool_call_rate"] == 1
    assert "1-Case Agent End-to-End" in render_markdown(report)


def test_fork_prometheus_metrics(tmp_path: Path) -> None:
    path = tmp_path / "metrics.prom"
    before = tmp_path / "metrics_before.prom"
    before.write_text(
        'vllm:fork_attention_observed_steps_total{engine="0"} 10\n'
        'vllm:fork_attention_active_steps_total{engine="0"} 5\n'
        'vllm:fork_attention_shared_ctas_total{engine="0"} 20\n'
        'vllm:fork_attention_singleton_ctas_total{engine="0"} 5\n',
        encoding="utf-8",
    )
    path.write_text(
        'vllm:fork_attention_observed_steps_total{engine="0"} 110\n'
        'vllm:fork_attention_active_steps_total{engine="0"} 80\n'
        'vllm:fork_attention_shared_ctas_total{engine="0"} 220\n'
        'vllm:fork_attention_singleton_ctas_total{engine="0"} 55\n',
        encoding="utf-8",
    )

    metrics = fork_metrics(path, before)

    assert metrics["activation_rate"] == 0.75
    assert metrics["shared_ctas"] == 200


def test_memory_summary_separates_allocation_and_live_kv(tmp_path: Path) -> None:
    (tmp_path / "gpu_before_server.csv").write_text(
        "2026/07/15 00:00:00, 100, 12000\n", encoding="utf-8"
    )
    (tmp_path / "gpu_after_warm.csv").write_text(
        "2026/07/15 00:00:01, 1000, 11100\n", encoding="utf-8"
    )
    (tmp_path / "memory_samples.csv").write_text(
        "unix_s,gpu_used_mib,gpu_free_mib,gpu_util_pct,"
        "memory_controller_util_pct,server_tree_rss_bytes,"
        "vllm:kv_cache_usage_perc,process_resident_memory_bytes,"
        "lmcache:local_cache_usage,lmcache:remote_cache_usage,"
        "vllm:num_requests_running,vllm:num_requests_waiting\n"
        "1,1100,11000,90,70,2097152000,0.25,100,1048576,0,8,0\n"
        "2,1200,10900,80,60,3145728000,0.5,200,2097152,0,16,0\n",
        encoding="utf-8",
    )
    (tmp_path / "metrics.prom").write_text(
        'vllm:cache_config_info{kv_cache_size_tokens="64000"} 1\n'
        "lmcache:local_cache_usage 2097152\n",
        encoding="utf-8",
    )

    summary = memory_summary(tmp_path)

    assert summary["gpu_peak_delta_from_idle_mib"] == 1100
    assert summary["gpu_peak_delta_from_warm_mib"] == 200
    assert summary["kv_cache_peak_live_tokens"] == 32000
    assert summary["server_tree_rss_peak_mib"] == 3000
    assert summary["lmcache_local_peak_mib"] == 2
