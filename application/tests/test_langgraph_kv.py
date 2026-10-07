from types import SimpleNamespace

import pytest
from agentrix_application.langgraph_kv import LangGraphKVHints, shared_prefix_length


def policy(**kwargs):
    return LangGraphKVHints(
        SimpleNamespace(checkpointer=None), namespace="test", **kwargs
    )


def test_thread_affinity_survives_fanout_retry_and_resume():
    hints = policy(disposable=True)
    configs = [
        {"configurable": {"thread_id": "会话", **extra}}
        for extra in (
            {},
            {"checkpoint_ns": "branch:one"},
            {"checkpoint_ns": "branch:two", "checkpoint_id": "resume"},
        )
    ]
    keys = [hints.options(c)["extra_headers"]["X-Session-ID"] for c in configs]
    assert len(set(keys)) == 1
    assert "会话" not in keys[0]
    other = hints.options({"configurable": {"thread_id": "other"}})
    assert other["extra_headers"]["X-Session-ID"] != keys[0]
    tenant = LangGraphKVHints(hints.graph, namespace="other")
    assert tenant.options(configs[0])["extra_headers"]["X-Session-ID"] != keys[0]


def test_default_and_durable_graphs_keep_backup_even_at_end():
    config = {"configurable": {"thread_id": "t"}}
    assert "extra_body" not in policy().options(config, reuse="none")
    hints = policy(disposable=True)
    hints.graph.checkpointer = object()
    assert "extra_body" not in hints.options(config, reuse="none")
    hints.graph.checkpointer = None
    for extra in (
        {"checkpoint_id": "prior"},
        {"__pregel_checkpointer": object()},
        {"checkpoint_ns": "unknown-runtime:node"},
    ):
        assert "extra_body" not in hints.options(
            {"configurable": {"thread_id": "t", **extra}}, reuse="none"
        )


def test_shared_cap_uses_token_identity_not_length_and_never_deletes():
    hints = policy(disposable=True)
    config = {"configurable": {"thread_id": "t"}}
    options = hints.options(
        config,
        reuse="shared_prefix",
        request_tokens=[1, 2, 9, 10],
        shared_tokens=[1, 2, 3],
    )
    assert options["extra_body"] == {"kv_transfer_params": {"max_offload_tokens": 2}}
    assert hints.options(config) == {"extra_headers": options["extra_headers"]}
    assert shared_prefix_length([1, 2], [1, 2, 3]) == 2
    assert shared_prefix_length([1], [2]) == 0
    assert shared_prefix_length([], []) == 0
    with pytest.raises(ValueError, match="both rendered"):
        hints.options(config, reuse="shared_prefix")


@pytest.mark.parametrize("tokens", [[True], [-1], ["1"], [1.5]])
def test_invalid_tokens_cannot_silently_choose_a_backup_boundary(tokens):
    with pytest.raises(ValueError):
        shared_prefix_length([1], tokens)


@pytest.mark.parametrize("thread_id", [None, "", 1, False])
def test_missing_identity_is_not_replaced_with_a_shared_fallback(thread_id):
    with pytest.raises(ValueError, match="thread_id"):
        policy().options({"configurable": {"thread_id": thread_id}})
