import csv
import json

from main_experiment_report import (
    annotate_baseline_comparisons,
    collect_run,
    render_report,
)


def test_collect_run_combines_latency_kv_and_telemetry(tmp_path) -> None:
    run_root = tmp_path / "fork_tp"
    backend_root = run_root / "fork_attn"
    backend_root.mkdir(parents=True)
    manifest = {
        "mode": "tp_accuracy",
        "model_name": "qwen3-14b",
        "dataset": "appworld",
        "prefix_tokens": 8192,
        "branches": 2,
        "variant": "fork_tp",
        "attention_backend": "FORK_ATTN",
        "offload": "ordinary",
        "dp_replicas": 1,
        "tp_size": 2,
        "use_flashinfer_sampler": False,
        "prefix_aware_policy": True,
        "fanout_admission_window": 16,
        "offload_cpu_gib": 8,
        "max_dataset_records": 32,
        "full_dataset": False,
        "experiment_profile": "fanout_validated",
        "branch_order": "case_major",
        "warm_shared_prefix": True,
        "output_tokens": 32,
        "case_count": 1,
        "enable_forest_cudagraph": True,
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (backend_root / "benchmark_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "baseline_unique_kv",
                "monowire_unique_kv",
                "kv_bytes_per_token",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "baseline_unique_kv": 200,
                "monowire_unique_kv": 120,
                "kv_bytes_per_token": 1024**2,
            }
        )
    raw = [
        {
            "total_latency_ms": 1000,
            "common": {
                "cases": [
                    {
                        "request_latency_ms": 50,
                        "ttft_ms": 20,
                        "tpot_ms": 4,
                        "input_tokens": 80,
                        "output_tokens": 5,
                    }
                ]
            },
            "branches": [
                {
                    "latency_ms": 100,
                    "ttft_ms": 40,
                    "tpot_ms": 5,
                    "input_tokens": 100,
                    "output_tokens": 10,
                },
                {
                    "latency_ms": 200,
                    "ttft_ms": 80,
                    "tpot_ms": 6,
                    "input_tokens": 120,
                    "output_tokens": 20,
                },
            ],
        }
    ]
    (backend_root / "raw_api_results.json").write_text(
        json.dumps(raw), encoding="utf-8"
    )
    profile = {
        "gpu_kv_cache_capacity_gib": 4,
        "kv_offload_load_bytes": 1024**3,
        "kv_offload_load_operations": 16,
        "kv_offload_load_average_mib": 64,
        "kv_offload_store_operations": 32,
        "kv_offload_store_average_mib": 8,
        "num_preemptions": 7,
        "fork_attention_observed_steps": 100,
        "fork_attention_active_steps": 75,
        "fork_attention_active_step_percent": 75,
        "fork_attention_shared_ctas": 600,
        "fork_attention_singleton_ctas": 200,
        "fork_attention_shared_cta_percent": 75,
        "ranks": [
            {
                "fork_dp_prefix_routing": {
                    "requests": 100,
                    "affinity_routes": 75,
                    "rank_routes": [48, 52],
                    "avg_route_us": 125.5,
                }
            }
        ],
        "telemetry": {
            "aggregate": {
                "gpu_kv_cache_usage_percent": {"max": 75},
                "compute_utilization_percent": {"mean": 80},
                "memory_bandwidth_utilization_percent": {"mean": 60},
                "cpu_kv_cache_occupancy_percent": {"max": 50},
            }
        },
    }
    (backend_root / "server_profile.json").write_text(
        json.dumps(profile), encoding="utf-8"
    )
    (backend_root / "vllm_server.log").write_text(
        "GPU KV cache size: 2,048 tokens\n" * 2,
        encoding="utf-8",
    )
    (backend_root / "vllm_server.attempt1.log").write_text(
        "GPU KV cache size: 99,999 tokens\n",
        encoding="utf-8",
    )
    repeatability_root = run_root / "repeatability_vs_fork_run1"
    repeatability_root.mkdir()
    (repeatability_root / "output_agreement.json").write_text(
        json.dumps(
            {
                "normalized_exact_match_percent": 99,
                "mean_token_f1_percent": 99.5,
                "mean_text_similarity_percent": 99.75,
            }
        ),
        encoding="utf-8",
    )

    row = collect_run(run_root / "manifest.json")
    annotate_baseline_comparisons([row])

    assert row["requests"] == 3
    assert row["common_requests"] == 1
    assert row["branch_requests"] == 2
    assert row["output_throughput_tokens_per_s"] == 35
    assert row["branch_output_throughput_tokens_per_s"] == 30
    assert row["ttft_ms_p50"] == 40
    assert row["branch_ttft_ms_p50"] == 60
    assert row["logical_kv_read_tokens"] == 120
    assert row["logical_kv_read_reduction_percent"] == 40
    assert row["estimated_peak_gpu_kv_gib"] == 3
    assert row["estimated_peak_cpu_kv_gib"] == 4
    assert row["estimated_peak_total_kv_gib"] == 7
    assert row["kv_offload_load_gib"] == 1
    assert row["kv_offload_load_operations"] == 16
    assert row["kv_offload_load_average_mib"] == 64
    assert row["num_preemptions"] == 7
    assert row["repeat_exact_match_percent"] == 99
    assert row["num_gpu_blocks_override"] is None
    assert row["use_flashinfer_sampler"] is False
    assert row["prefix_aware_policy"] is True
    assert row["fanout_admission_window"] == 16
    assert row["max_dataset_records"] == 32
    assert row["full_dataset"] is False
    assert row["requested_output_tokens"] == 32
    assert row["cases_per_batch"] == 1
    assert row["enable_forest_cudagraph"] is True
    assert row["dataset_records"] == 1
    assert row["dp_affinity_route_percent"] == 75
    assert row["dp_average_route_us"] == 125.5
    assert row["fork_attention_active_step_percent"] == 75
    assert row["fork_attention_shared_cta_percent"] == 75
    report = render_report([row])
    assert "Memory BW" in report
    assert "Offload Traffic" in report
    assert "16 (64.00)" in report
    assert "Prefix-Aware DP Routing" in report
    assert "Physical ForkAttention Activation" in report
    assert "Provenance" in report
    assert "FlashInfer sampler" in report
    assert "Admission window" in report
    assert "Forest graph" in report
    assert "fanout_validated" in report
    assert "Record cap" in report
    assert "| 32 | no | 1 |" in report


def test_annotate_baselines_compares_policy_to_ordinary_fork() -> None:
    rows = [
        {
            "mode": "single_gpu",
            "model_name": "model",
            "dataset": "data",
            "prefix_tokens": 8192,
            "branches": 8,
            "variant": "fork_ordinary_offload",
            "offload": "ordinary",
            "output_throughput_tokens_per_s": 100,
            "estimated_peak_gpu_kv_gib": 4,
            "kv_offload_load_gib": 8,
            "kv_offload_store_gib": 2,
        },
        {
            "mode": "single_gpu",
            "model_name": "model",
            "dataset": "data",
            "prefix_tokens": 8192,
            "branches": 8,
            "variant": "fork_optimized_offload",
            "offload": "optimized",
            "output_throughput_tokens_per_s": 120,
            "estimated_peak_gpu_kv_gib": 3,
            "kv_offload_load_gib": 6,
            "kv_offload_store_gib": 2,
        },
    ]

    annotate_baseline_comparisons(rows)

    optimized = rows[1]
    assert optimized["baseline_variant"] == "fork_ordinary_offload"
    assert optimized["output_throughput_change_percent"] == 20
    assert optimized["peak_gpu_kv_reduction_vs_baseline_percent"] == 25
    assert optimized["kv_offload_load_reduction_vs_baseline_percent"] == 25


def test_offload_validated_separates_scheduler_and_connector_effects() -> None:
    shared = {
        "mode": "single_gpu",
        "model_name": "model",
        "dataset": "data",
        "prefix_tokens": 8192,
        "branches": 16,
        "offload": "ordinary",
        "estimated_peak_gpu_kv_gib": 2,
        "estimated_peak_total_kv_gib": 6,
    }
    rows = [
        {
            **shared,
            "variant": "fork_ordinary_offload",
            "output_throughput_tokens_per_s": 100,
            "kv_offload_load_gib": 8,
        },
        {
            **shared,
            "variant": "fork_scheduled_ordinary_offload",
            "output_throughput_tokens_per_s": 125,
            "kv_offload_load_gib": 6,
        },
        {
            **shared,
            "variant": "fork_optimized_offload",
            "experiment_profile": "offload_validated",
            "output_throughput_tokens_per_s": 150,
            "kv_offload_load_gib": 3,
        },
    ]

    annotate_baseline_comparisons(rows)

    assert rows[1]["baseline_variant"] == "fork_ordinary_offload"
    assert rows[1]["output_throughput_change_percent"] == 25
    assert rows[2]["baseline_variant"] == "fork_scheduled_ordinary_offload"
    assert rows[2]["output_throughput_change_percent"] == 20
    assert rows[2]["kv_offload_load_reduction_vs_baseline_percent"] == 50
