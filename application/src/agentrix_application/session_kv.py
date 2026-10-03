"""Opt-in request options for official sticky routing and selective KV backup.

This adapter does not evict cached blocks or implement an engine policy.
The backend must support the official max_offload_tokens request field.
"""

from __future__ import annotations


def session_kv_options(session_id: str, *, terminal: bool = False) -> dict:
    """Build OpenAI SDK kwargs from application-known session lifecycle.

    Keep the same opaque identity across turns and retries. Set terminal only
    when the application knows *before dispatch* that this request will not be
    revisited. Unknown reuse stays on the backend's default backup policy.
    A finished LLM turn, a tool call, or a finished child branch is not by itself
    proof that its prefix cannot be reused by another turn or branch.

    Skipping backup does not delete existing copies or prevent CPU cache reads.
    Merge these options explicitly if the caller already supplies extra_body
    or extra_headers; do not overwrite unrelated request settings.
    """
    if not isinstance(session_id, str) or not session_id or not session_id.isascii():
        raise ValueError("session_id must be a nonempty ASCII string")
    if any(ord(character) <= 32 or ord(character) == 127 for character in session_id):
        raise ValueError("session_id cannot contain whitespace or control characters")
    if type(terminal) is not bool:
        raise TypeError("terminal must be a boolean, not an inferred truthy value")
    options: dict = {"extra_headers": {"X-Session-ID": session_id}}
    if terminal:
        options["extra_body"] = {"kv_transfer_params": {"max_offload_tokens": 0}}
    return options
