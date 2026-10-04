import copy
import hashlib
import json

import pytest
from benchmark_fork_scale import make_prompts
from summarize_fork_scale import compare_runs, summarize


def run(seed=42, wall=2.0):
    prefix, branches = 1024, 2
    return {
        "valid": True,
        "config": {
            "seed": seed,
            "prefix_tokens": [prefix],
            "branches": [branches],
            "trials": 1,
            "output_tokens": 4,
        },
        "session_owners": {"0": 3, "1": 7},
        "shapes": [
            {
                "prefix_tokens": prefix,
                "branches_per_rank": branches,
                "input_sha256": hashlib.sha256(
                    json.dumps(make_prompts(seed, prefix, branches)).encode()
                ).hexdigest(),
                "trials": [
                    {
                        "trial": 0,
                        "started": 100,
                        "ended": 100 + wall,
                        "wall_seconds": wall,
                        "output_tokens": 16,
                        "rank_counters": {
                            rank: {
                                "tokens_local_cache_hit": 2048,
                                "tokens_local_compute": 256,
                                "preemptions": 0,
                            }
                            for rank in ("0", "1")
                        },
                        "rows": [
                            {
                                "expected_rank": rank,
                                "document": doc,
                                "branch": branch,
                                "prompt_tokens": 1152,
                                "completion_tokens": 4,
                                "output_token_sha256": "a" * 64,
                                "ttft_ms": 100,
                                "e2e_ms": 900,
                            }
                            for rank, doc in (("0", 3), ("1", 7))
                            for branch in (0, 1)
                        ],
                    }
                ],
            }
        ],
    }


def test_equal_length_different_generated_tokens_are_not_accepted():
    a, b = run(), run()
    b["shapes"][0]["trials"][0]["rows"][0]["output_token_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="Generated tokens differ"):
        compare_runs(a, b)
    diagnostic = compare_runs(a, b, diagnose_output_differences=True)[0]
    assert diagnostic["equal_output_latency_comparison"] is False
    assert diagnostic["equal_output_pairs"] == 3
    assert diagnostic["output_pairs"] == 4
    assert diagnostic["output_differences"] == [{"trial": 0, "rank": "0", "branch": 0}]


def test_forged_shared_input_digest_is_not_accepted_even_if_both_arms_match():
    a = run()
    a["shapes"][0]["input_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="Input plan hash mismatch"):
        compare_runs(a, copy.deepcopy(a))


def test_prompt_source_changes_are_not_accepted_even_with_equal_prompt_totals():
    a, b = run(), run()
    counts = b["shapes"][0]["trials"][0]["rank_counters"]["0"]
    counts["tokens_local_compute"] -= 1
    counts["tokens_local_cache_hit"] += 1
    with pytest.raises(ValueError, match="Prompt work differs"):
        compare_runs(a, b)


@pytest.mark.parametrize(
    "fault", ["duplicate", "missing", "preemption", "nonfinite", "wrong_owner"]
)
def test_invalid_formal_batch_is_rejected(fault):
    a, b = run(), run()
    trial = b["shapes"][0]["trials"][0]
    if fault == "duplicate":
        trial["rows"][1] = copy.deepcopy(trial["rows"][0])
    elif fault == "missing":
        trial["rows"].pop()
    elif fault == "preemption":
        trial["rank_counters"]["0"]["preemptions"] = 1
    elif fault == "nonfinite":
        trial["wall_seconds"] = float("nan")
    else:
        trial["rows"][0]["document"] = 7
    with pytest.raises(ValueError):
        compare_runs(a, b)


def test_summary_requires_complete_plan_and_combines_service_means(tmp_path):
    completed = []
    for restart, times in enumerate(((1.0, 0.5), (3.0, 2.5))):
        for arm, wall in zip(("consistent_hash", "consistent_hash_fork"), times):
            label = f"42-{arm}-{restart}-fork-scale"
            completed.append(label)
            (tmp_path / (label + ".json")).write_text(json.dumps(run(wall=wall)))
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({"phase": "benchmark", "completed": completed}))
    with pytest.raises(ValueError, match="incomplete"):
        summarize(tmp_path, [42], 2)
    progress.write_text(json.dumps({"phase": "complete", "completed": completed}))
    result = summarize(tmp_path, [42], 2)
    assert result["service_pairs"] == 2
    assert result["output_pairs"] == 8
    group = result["aggregate"][0]
    assert group["baseline"]["wall_seconds"] == 2.0
    assert group["candidate"]["wall_seconds"] == 1.5
    # Compute change from the two mean durations, not mean percentages (-33.3%).
    assert group["wall_change_pct"] == -25
    assert group["faster_service_pairs"] == 2
    with pytest.raises(ValueError, match="incomplete"):
        summarize(tmp_path, [42], 3)
