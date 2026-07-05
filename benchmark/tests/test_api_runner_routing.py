from api_runner import _branch_rank_map, _common_rank, _rank_counts


def test_round_robin_routes_individual_branches_evenly() -> None:
    route_map = _branch_rank_map(
        case_count=2,
        branch_count=4,
        branch_group_size=2,
        desired_suffixes=[1] * 8,
        dp_size=2,
        dp_routing="round_robin",
    )

    assert _rank_counts(route_map, 2) == [4, 4]
    assert route_map[(0, 0)] == 0
    assert route_map[(0, 1)] == 1


def test_prefix_sticky_keeps_whole_case_on_one_rank() -> None:
    route_map = _branch_rank_map(
        case_count=3,
        branch_count=4,
        branch_group_size=2,
        desired_suffixes=[1] * 12,
        dp_size=2,
        dp_routing="prefix_sticky",
    )

    assert {route_map[(0, branch)] for branch in range(4)} == {0}
    assert {route_map[(1, branch)] for branch in range(4)} == {1}
    assert {route_map[(2, branch)] for branch in range(4)} == {0}


def test_prefix_forest_balances_without_splitting_groups() -> None:
    route_map = _branch_rank_map(
        case_count=1,
        branch_count=8,
        branch_group_size=2,
        desired_suffixes=[100, 100, 80, 80, 60, 60, 40, 40],
        dp_size=2,
        dp_routing="prefix_forest",
    )

    for group_id in range(4):
        ranks = {
            route_map[(0, branch)]
            for branch in range(group_id * 2, group_id * 2 + 2)
        }
        assert len(ranks) == 1
    assert _rank_counts(route_map, 2) == [4, 4]
    assert _common_rank(0, route_map, branch_count=8, dp_size=2) == 0
