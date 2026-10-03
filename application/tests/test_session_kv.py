import pytest
from agentrix_application.session_kv import session_kv_options


def test_known_terminal_skips_only_new_backup_and_keeps_routing_identity():
    active = session_kv_options("opaque-session")
    terminal = session_kv_options("opaque-session", terminal=True)
    assert active == {"extra_headers": {"X-Session-ID": "opaque-session"}}
    assert terminal["extra_headers"] == active["extra_headers"]
    assert terminal["extra_body"] == {"kv_transfer_params": {"max_offload_tokens": 0}}
    terminal["extra_headers"]["X-Session-ID"] = "changed"
    assert session_kv_options("opaque-session") == active


@pytest.mark.parametrize(
    "identity", ["", "a\nb", "a\rb", "a b", "a\x00b", "a\x7fb", "会话", None]
)
def test_reject_invalid_header_identity(identity):
    with pytest.raises(ValueError):
        session_kv_options(identity)


@pytest.mark.parametrize("terminal", ["false", "true", 0, 1, None])
def test_ambiguous_terminal_signal_cannot_suppress_backup(terminal):
    with pytest.raises(TypeError):
        session_kv_options("opaque-session", terminal=terminal)
