from api_runner import (
    _branch_rank_map,
    _client_route_counts,
    _common_rank,
    _internal_dp_headers,
    _rank_counts,
    _reported_route_rank,
    _time_per_output_token,
)


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


def test_prefix_skewed_creates_complementary_minority_branches() -> None:
    route_map = _branch_rank_map(
        case_count=2,
        branch_count=9,
        branch_group_size=1,
        desired_suffixes=[1] * 18,
        dp_size=2,
        dp_routing="prefix_skewed",
    )

    assert [route_map[(0, branch)] for branch in range(9)].count(0) == 8
    assert [route_map[(1, branch)] for branch in range(9)].count(1) == 8
    assert route_map[(0, 0)] == 1
    assert route_map[(1, 0)] == 0
    assert _common_rank(0, route_map, branch_count=9, dp_size=2) == 0
    assert _common_rank(1, route_map, branch_count=9, dp_size=2) == 1


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
            route_map[(0, branch)] for branch in range(group_id * 2, group_id * 2 + 2)
        }
        assert len(ranks) == 1
    assert _rank_counts(route_map, 2) == [4, 4]
    assert _common_rank(0, route_map, branch_count=8, dp_size=2) == 0


def test_time_per_output_token_excludes_first_token_latency() -> None:
    assert _time_per_output_token(100.0, 40.0, 4) == 20.0
    assert _time_per_output_token(40.0, 40.0, 1) == 0.0
    assert _time_per_output_token(100.0, None, 4) is None


def test_internal_dp_only_forces_rank_for_skewed_reload_workload() -> None:
    assert _internal_dp_headers(2, "single", 1) is None
    assert _internal_dp_headers(2, "prefix_forest", 1) is None
    assert _internal_dp_headers(None, "prefix_skewed", 1) is None
    assert _internal_dp_headers(2, "prefix_skewed", 1) == {"X-data-parallel-rank": "1"}


def test_internal_dp_reports_server_owned_routing_as_unknown() -> None:
    route_map = {(0, 0): 0, (0, 1): 1}

    assert _reported_route_rank(2, "single", 0) is None
    assert _reported_route_rank(2, "prefix_skewed", 1) == 1
    assert _reported_route_rank(None, "round_robin", 1) == 1
    assert _client_route_counts(2, "single", route_map, [0], 2) == (None, None)
    assert _client_route_counts(None, "round_robin", route_map, [0], 2) == (
        [1, 1],
        [1, 0],
    )
