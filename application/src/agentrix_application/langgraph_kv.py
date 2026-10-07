"""Map graph-owned reuse information to official routing and KV backup APIs.

There is deliberately no LangGraph dependency here. Pass the config received by
a node and bind the policy to its compiled graph. Graph structure alone cannot
prove that a conversation, checkpoint, or branch will never be revisited.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .session_kv import session_kv_options

KVReuse = Literal["unknown", "shared_prefix", "none"]


def shared_prefix_length(request: Sequence[int], shared: Sequence[int]) -> int:
    """Compare actual rendered token IDs, including chat-template boundaries."""
    for tokens in (request, shared):
        if isinstance(tokens, (str, bytes)) or not isinstance(tokens, Sequence):
            raise TypeError("token IDs must be sequences of nonnegative integers")
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("token IDs must be nonnegative integers")
    for index, (left, right) in enumerate(zip(request, shared)):
        if left != right:
            return index
    return min(len(request), len(shared))


class LangGraphKVHints:
    """A conservative, opt-in policy for a compiled graph.

    ``disposable`` is an application contract: full private transcripts will
    not be used in a later invocation. Checkpointed graphs always retain the
    backend's default backup policy. END, tool completion and Send task IDs are
    never interpreted as proof of this contract.

    Use a stable deployment/tenant namespace. All branches and retries in one
    thread share affinity; checkpoint namespaces and task IDs do not fragment
    it. The hash is an opaque routing key, not an authentication boundary.

    A prefix cap neither creates recurrent-state checkpoints nor pins existing
    backups. Keep parent requests on the normal backup policy; the backend
    decides which aligned attention/state boundaries are actually restorable.
    """

    def __init__(self, graph: Any, *, namespace: str, disposable: bool = False):
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("namespace must be a nonempty string")
        if type(disposable) is not bool:
            raise TypeError("disposable must be a boolean")
        if not hasattr(graph, "checkpointer"):
            raise TypeError("bind the policy to a compiled LangGraph graph")
        self.graph = graph
        self.namespace = namespace
        self.disposable = disposable

    def can_limit_backup(self, config: Mapping[str, Any]) -> bool:
        configurable = config.get("configurable", {})
        # The compiled graph covers explicit persistence. This additional
        # guard recognizes an inherited checkpointer in LangGraph's node config.
        # checkpoint_ns itself is unsuitable: even ephemeral Send tasks have it.
        # An unrecognized namespaced config retains the default backup policy.
        return (
            self.disposable
            and self.graph.checkpointer in (None, False)
            and not configurable.get("checkpoint_id")
            and (
                not configurable.get("checkpoint_ns")
                or "__pregel_checkpointer" in configurable
            )
            and configurable.get("__pregel_checkpointer") in (None, False)
        )

    def options(
        self,
        config: Mapping[str, Any],
        *,
        reuse: KVReuse = "unknown",
        request_tokens: Sequence[int] | None = None,
        shared_tokens: Sequence[int] | None = None,
    ) -> dict:
        if reuse not in ("unknown", "shared_prefix", "none"):
            raise ValueError("unknown KV reuse policy")
        thread_id = config.get("configurable", {}).get("thread_id")
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("LangGraph config requires a nonempty string thread_id")
        identity = json.dumps([self.namespace, thread_id], ensure_ascii=True)
        session_id = "lg-" + hashlib.sha256(identity.encode()).hexdigest()
        options = session_kv_options(session_id)
        if reuse == "unknown" or not self.can_limit_backup(config):
            return options
        if reuse == "none":
            cap = 0
        else:
            if request_tokens is None or shared_tokens is None:
                raise ValueError("shared_prefix requires both rendered token sequences")
            cap = shared_prefix_length(request_tokens, shared_tokens)
        options["extra_body"] = {"kv_transfer_params": {"max_offload_tokens": cap}}
        return options
