from benchmark_fork_scale import make_prompts


def test_shared_input_plan_is_deterministic_and_distinguishes_groups_and_tails():
    groups = make_prompts(42, 32768, 8)
    assert groups == make_prompts(42, 32768, 8)
    assert groups[0]["prime"][:32768] != groups[1]["prime"][:32768]
    for group in groups:
        assert len(group["branches"]) == 8
        prefix = group["prime"][:32768]
        suffixes = set()
        for branch in group["branches"]:
            assert len(branch) == 32768 + 128
            assert branch[:32768] == prefix
            suffixes.add(tuple(branch[32768:]))
        assert len(suffixes) == 8
        assert tuple(group["prime"][32768:]) not in suffixes


def test_branch_scan_preserves_the_same_prefix_and_existing_branches():
    small = make_prompts(7, 65536, 2)
    large = make_prompts(7, 65536, 8)
    for a, b in zip(small, large):
        assert a["prime"] == b["prime"]
        assert a["branches"] == b["branches"][:2]
