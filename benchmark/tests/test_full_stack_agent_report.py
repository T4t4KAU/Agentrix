from full_stack_agent_report import build_report


def payload(arm: str, blocks: int = 2600):
    full = arm == "full"
    return {
        "metadata": {
            "arm": arm,
            "model": "/models/Qwen3-32B",
            "dtype": "bfloat16",
            "attention_backend": "FORK_ATTN" if full else "FLASH_ATTN",
            "data_parallel_size": 4,
            "num_gpu_blocks_override": blocks,
            "block_size_tokens": 16,
            "max_model_len": 40960,
            "max_num_seqs": 64,
            "max_num_batched_tokens": 16384,
            "enforce_eager": True,
            "async_scheduling": False,
            "case_manifest_sha256": "abc",
            "question_limit": 1,
            "tool_rounds": 4,
            "tool_delay_ms": 800,
            "action_tokens": 32,
            "answer_tokens": 96,
        },
        "wall_seconds": 10 if full else 12,
        "tasks_per_second": 0.1 if full else 1 / 12,
        "mean_f1": 0.8,
        "success_rate": 1.0,
        "valid_citation_rate": 1.0,
        "mean_task_latency_seconds": 9 if full else 11,
        "mean_turn_ttft_seconds": 1 if full else 2,
        "compaction_saved_chars": 100 if full else 0,
        "trimmer_stats": {
            "trimmed_sessions": 2 if full else 0,
            "released_block_references": 32 if full else 0,
        },
        "router_stats": {"active": full, "route_count": 2 if full else 0},
        "kv": {
            "connector_counters": {
                "vllm:kv_offload_store_bytes": 1024 if full else 0
            },
            "fork_execution": {"active_steps": 2 if full else 0},
        },
        "gpu_memory_samples": [
            {"gpus": [{"used_mib": 80_000 if full else 82_000}]}
        ],
        "results": [
            {"case_id": "c", "source_id": "q", "f1": 0.8}
        ],
    }


def test_report_requires_and_observes_full_stack():
    report = build_report(payload("baseline"), payload("full"))
    assert report["fairness"]["strict_equal_gpu_kv_capacity"]
    assert report["all_five_mechanisms_observed"]
    assert report["delta"]["paired_f1_delta"] == 0
