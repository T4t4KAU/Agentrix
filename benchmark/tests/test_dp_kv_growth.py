import copy

from benchmark_dp_kv_growth import evaluate, make_plan


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())


def test_growth_preserves_prefix_and_branch_tails_are_isolated():
    plan = make_plan(
        Tokenizer(),
        seed=17,
        sessions=4,
        branches=2,
        rounds=3,
        initial_tokens=6145,
        growth_tokens=1024,
        pressure_requests=2,
    )
    for turn in range(2):
        for a, b in zip(plan[turn]["active"], plan[turn + 1]["active"], strict=True):
            assert b[: len(a)] == a
            assert len(b) == len(a) + 1024
    a, b, c, _ = plan[0]["active"]
    assert a[:4096] == b[:4096]
    assert a[4096:] != b[4096:]
    assert a[:4096] != c[:4096]
    assert plan[0]["pressure"] != plan[1]["pressure"]


def test_recovery_mismatch_is_not_hidden_by_latency_or_preemptions():
    row = {
        "index": 0,
        "input_sha256": "input",
        "token_ids": [1, 2],
        "ttft_ms": 1,
        "end_to_end_ttft_ms": 1,
    }
    phase = {"rows": [row], "counters": {"preemptions": 1}}
    result = {
        "configuration": {"mode": "selective"},
        "rounds": [
            {
                "round": 0,
                "grow": copy.deepcopy(phase),
                "pressure": copy.deepcopy(phase),
                "resume": copy.deepcopy(phase),
            }
        ],
    }
    result["rounds"][0]["resume"]["rows"][0]["token_ids"] = [1, 3]
    evaluate(result)
    assert not result["correctness_passed"]
    assert result["summary"]["preemptions"] == 3
    assert "restored output differs" in result["correctness_failures"][0]


def test_terminal_backup_violation_fails_even_when_outputs_match():
    row = {
        "index": 0,
        "input_sha256": "input",
        "token_ids": [1],
        "ttft_ms": 1,
        "end_to_end_ttft_ms": 1,
    }
    phase = {"rows": [row], "counters": {}}
    result = {
        "configuration": {"mode": "selective"},
        "rounds": [
            {
                "round": 0,
                "grow": copy.deepcopy(phase),
                "pressure": copy.deepcopy(phase),
                "resume": copy.deepcopy(phase),
            }
        ],
    }
    result["rounds"][0]["pressure"]["counters"]["store_bytes"] = 1024
    evaluate(result)
    assert not result["correctness_passed"]
    assert "Terminal requests wrote CPU KV" in result["correctness_failures"]
