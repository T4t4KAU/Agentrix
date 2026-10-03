import json

from run_ascend_router_comparison import fork_attention_command


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
