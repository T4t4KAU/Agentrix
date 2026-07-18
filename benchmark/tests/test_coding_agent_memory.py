from pathlib import Path

from coding_agent_memory import cache_capacity_tokens, summarize_memory


def test_cache_capacity_tokens_reads_each_dp_engine(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.prom"
    metrics.write_text(
        "\n".join(
            (
                'vllm:cache_config_info{engine="0",kv_cache_size_tokens="100"} 1.0',
                'vllm:cache_config_info{engine="1",kv_cache_size_tokens="100"} 1.0',
            )
        ),
        encoding="utf-8",
    )

    assert cache_capacity_tokens(metrics) == [100, 100]


def test_summarize_memory_includes_application_and_kv_peaks() -> None:
    samples = [
        {
            "time": 1.0,
            "gpus": [
                {
                    "index": 0,
                    "memory_used_mib": 100,
                    "memory_total_mib": 1000,
                    "memory_controller_utilization_percent": 5,
                },
                {
                    "index": 1,
                    "memory_used_mib": 110,
                    "memory_total_mib": 1000,
                    "memory_controller_utilization_percent": 6,
                },
            ],
            "kv_usage": [0.1, 0.2],
            "process_tree_rss_kib": 1024,
            "application_tree_rss_kib": 0,
        },
        {
            "time": 2.0,
            "gpus": [
                {
                    "index": 0,
                    "memory_used_mib": 300,
                    "memory_total_mib": 1000,
                    "memory_controller_utilization_percent": 90,
                },
                {
                    "index": 1,
                    "memory_used_mib": 350,
                    "memory_total_mib": 1000,
                    "memory_controller_utilization_percent": 80,
                },
            ],
            "kv_usage": [0.5, 0.75],
            "process_tree_rss_kib": 4096,
            "application_tree_rss_kib": 2048,
        },
    ]

    result = summarize_memory(samples, [100, 100])

    assert result["gpu"]["peak_aggregate_used_mib"] == 650
    assert result["gpu"]["peak_delta_from_warm_mib"] == 440
    assert result["server_process_tree"]["peak_rss_mib"] == 4
    assert result["application_process_tree"]["peak_rss_mib"] == 2
    assert result["kv_cache"]["peak_aggregate_usage_percent"] == 62.5
    assert result["kv_cache"]["peak_aggregate_live_tokens"] == 125
