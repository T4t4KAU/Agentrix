import json

from run_ascend_router_comparison import comparison_schedule, fork_attention_command


def test_fork_pair_changes_only_enable_and_preserves_other_configuration():
    command = [
        "python",
        "--additional-config",
        json.dumps({"mamba_cache_retention": {"interval": 8192}}),
    ]
    before = list(command)
    arms = [fork_attention_command(command, enabled, 4096) for enabled in (False, True)]
    configs = [json.loads(arm[2]) for arm in arms]
    assert command == before
    for config in configs:
        assert config["mamba_cache_retention"] == {"interval": 8192}
        assert config["fork_attention"]["min_shared_tokens"] == 4096
        assert config["fork_attention"]["diagnostics"] is True
    assert configs[0]["fork_attention"].pop("enabled") is False
    assert configs[1]["fork_attention"].pop("enabled") is True
    assert configs[0] == configs[1]


def test_fork_command_adds_missing_configuration_without_mutation():
    command = ["python", "--enforce-eager"]
    result = fork_attention_command(command, True, 32768)
    assert command == ["python", "--enforce-eager"]
    assert result[:2] == command
    assert result[2] == "--additional-config"
    assert json.loads(result[3])["fork_attention"]["enabled"] is True


def test_independent_restarts_keep_seed_pairs_adjacent_and_reverse_order():
    policies = ["fork", "native"]
    result = list(comparison_schedule([101, 202], policies, 2))
    assert result == [
        (101, "fork", 0),
        (101, "native", 0),
        (202, "native", 0),
        (202, "fork", 0),
        (101, "native", 1),
        (101, "fork", 1),
        (202, "fork", 1),
        (202, "native", 1),
    ]
    assert policies == ["fork", "native"]
    assert len(set(result)) == len(result)


def test_default_single_trial_preserves_the_existing_router_order():
    assert list(comparison_schedule([1, 2], ["hash", "native", "cache"], 1)) == [
        (1, "hash", 0),
        (1, "native", 0),
        (1, "cache", 0),
        (2, "cache", 0),
        (2, "native", 0),
        (2, "hash", 0),
    ]
