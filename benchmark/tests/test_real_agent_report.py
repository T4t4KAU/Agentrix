from real_agent_report import build_report


def payload(arm: str):
    optimized = arm == "agentrix"
    return {
        "metadata": {
            "arm": arm,
            "model": "/models/Qwen3-32B",
            "dtype": "bfloat16",
            "attention_backend": "FORK_ATTN" if optimized else "FLASH_ATTN",
            "trimmer": False,
            "offload": False,
            "compaction": False,
            "data_parallel_size": 4,
            "num_gpu_blocks_override": 2600,
            "block_size_tokens": 16,
            "max_model_len": 40960,
            "max_num_seqs": 64,
            "max_num_batched_tokens": 16384,
            "enforce_eager": True,
            "async_scheduling": False,
            "case_manifest_sha256": "abc",
            "question_limit": 1,
            "tool_rounds": 4,
            "tool_delay_ms": 100,
            "action_tokens": 32,
            "answer_tokens": 96,
        },
        "wall_seconds": 10 if optimized else 12,
        "tasks_per_second": 0.1 if optimized else 1 / 12,
        "mean_f1": 0.8,
        "success_rate": 1.0,
        "agent_completion_rate": 1.0,
        "valid_citation_rate": 1.0,
        "tool_call_valid_rate": 1.0,
        "mean_task_latency_seconds": 9 if optimized else 11,
        "mean_turn_ttft_seconds": 1 if optimized else 2,
        "compaction_saved_chars": 0,
        "router_stats": {
            "active": optimized,
            "route_count": 2 if optimized else 0,
        },
        "kv": {
            "fork_execution": {
                "active_steps": 2 if optimized else 0,
                "shared_ctas": 12 if optimized else 0,
            }
        },
        "gpu_memory_samples": [
            {"gpus": [{"used_mib": 80_000 if optimized else 82_000}]}
        ],
        "results": [
            {
                "case_id": "c",
                "source_id": "q",
                "f1": 0.8,
                "prediction": "answer",
                "tool_events": [{}, {}, {}, {}],
            }
        ],
    }


def test_real_agent_report():
    report = build_report(payload("baseline"), payload("agentrix"))
    assert report["fairness"]["strict_equal_gpu_kv_capacity"]
    assert report["core_agentrix_path_observed"]
    assert report["activation"]["ttl_trimmer_disabled"]
    assert report["activation"]["kv_offload_disabled"]
    assert report["activation"]["prompt_compaction_disabled"]
