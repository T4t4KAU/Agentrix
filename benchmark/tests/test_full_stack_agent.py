from __future__ import annotations

import pytest

from full_stack_agent import (
    BM25Index,
    build_passages,
    parse_action,
    validate_equal_kv_metadata,
)


def test_bm25_and_action_parser() -> None:
    passages = build_passages(
        "The capital of France is Paris.\n\n"
        "Berlin is the capital of Germany."
    )
    matches = BM25Index(passages).search("France capital", top_k=1)
    assert "Paris" in matches[0][0].text
    assert parse_action("SEARCH: France capital").kind == "search"
    assert parse_action("READ: P0001").value == "P0001"
    assert parse_action("FINAL: Paris || P0000").value == "Paris"
    assert parse_action("unstructured answer") is None


def test_equal_kv_validation_fails_closed() -> None:
    common = {
        "model": "Qwen3-32B",
        "dtype": "bfloat16",
        "data_parallel_size": 4,
        "num_gpu_blocks_override": 2600,
        "block_size_tokens": 16,
        "max_model_len": 40960,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 16384,
        "enforce_eager": True,
        "async_scheduling": False,
        "case_manifest_sha256": "abc",
        "question_limit": 8,
        "tool_rounds": 4,
        "tool_delay_ms": 600,
        "action_tokens": 32,
        "answer_tokens": 96,
    }
    result = validate_equal_kv_metadata(common, dict(common))
    assert result["total_gpu_blocks"] == 10400
    changed = dict(common, num_gpu_blocks_override=2599)
    with pytest.raises(ValueError, match="strict equal-KV"):
        validate_equal_kv_metadata(common, changed)
