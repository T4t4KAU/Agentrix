import pytest

from telemetry import parse_prometheus, summarize_samples


def test_parse_prometheus_selects_kv_and_queue_metrics() -> None:
    metrics = parse_prometheus(
        """
# HELP vllm:kv_cache_usage_perc GPU cache usage.
vllm:kv_cache_usage_perc{engine="0"} 0.75
vllm:num_requests_running{engine="0"} 3
unrelated_metric 42
"""
    )

    assert metrics == {
        "vllm:kv_cache_usage_perc": [0.75],
        "vllm:num_requests_running": [3.0],
    }


def test_summarize_samples_reports_gpu_and_kv_distributions() -> None:
    samples = [
        {
            "gpus": [
                {
                    "index": "0",
                    "compute_utilization_percent": 50.0,
                    "memory_bandwidth_utilization_percent": 60.0,
                    "memory_used_mib": 1000.0,
                }
            ],
            "vllm": [
                {
                    "metrics": {
                        "vllm:kv_cache_usage_perc": [0.25],
                        "vllm:kv_offload_cpu_cache_occupancy_perc": [0.5],
                        "vllm:num_requests_running": [2.0],
                    }
                }
            ],
        },
        {
            "gpus": [
                {
                    "index": "0",
                    "compute_utilization_percent": 100.0,
                    "memory_bandwidth_utilization_percent": 80.0,
                    "memory_used_mib": 1200.0,
                }
            ],
            "vllm": [
                {
                    "metrics": {
                        "vllm:kv_cache_usage_perc": [0.75],
                        "vllm:kv_offload_cpu_cache_occupancy_perc": [1.0],
                        "vllm:num_requests_running": [4.0],
                    }
                }
            ],
        },
    ]

    summary = summarize_samples(samples)

    assert summary["sample_count"] == 2
    assert summary["raw_sample_count"] == 2
    assert summary["aggregate"]["compute_utilization_percent"]["mean"] == 75.0
    assert summary["aggregate"]["gpu_kv_cache_usage_percent"]["mean"] == 50.0
    assert summary["aggregate"]["gpu_kv_cache_usage_percent"]["max"] == 75.0
    assert summary["aggregate"]["cpu_kv_cache_occupancy_percent"]["max"] == 100.0
    assert summary["gpus"]["0"]["memory_used_mib"]["p95"] == pytest.approx(1190.0)


def test_summarize_samples_aggregates_dp_kv_usage_per_timestamp() -> None:
    samples = [
        {
            "gpus": [],
            "vllm": [
                {
                    "metrics": {
                        "vllm:kv_cache_usage_perc": [1.0, 0.5],
                        "vllm:num_requests_running": [4.0, 2.0],
                    }
                }
            ],
        }
    ]

    summary = summarize_samples(samples)["aggregate"]

    assert summary["gpu_kv_cache_usage_percent"]["max"] == 75
    assert summary["requests_running"]["max"] == 6


def test_summarize_samples_trims_idle_edges_but_keeps_internal_idle() -> None:
    def sample(running: float, compute: float) -> dict:
        return {
            "gpus": [
                {
                    "index": "0",
                    "compute_utilization_percent": compute,
                }
            ],
            "vllm": [{"metrics": {"vllm:num_requests_running": [running]}}],
        }

    summary = summarize_samples(
        [sample(0, 0), sample(1, 80), sample(0, 20), sample(1, 100), sample(0, 0)]
    )

    assert summary["raw_sample_count"] == 5
    assert summary["sample_count"] == 3
    assert summary["aggregate"]["compute_utilization_percent"]["mean"] == pytest.approx(
        200 / 3
    )
