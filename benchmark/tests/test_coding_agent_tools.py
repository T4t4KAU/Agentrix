import json
import subprocess
from pathlib import Path

import pytest

from coding_agent_tools import RepositoryTools, ToolError


def make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src" / "value.c").write_text("int value = 1;\n")
    subprocess.run(("git", "init", "-q"), cwd=workspace, check=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.com"),
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Test"), cwd=workspace, check=True
    )
    subprocess.run(("git", "add", "."), cwd=workspace, check=True)
    subprocess.run(("git", "commit", "-qm", "baseline"), cwd=workspace, check=True)
    return workspace


def test_tools_record_search_read_and_patch(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    tools = RepositoryTools(workspace, {"allowed_paths": ["src/value.c"]})
    assert "value" in tools.search("value", "*.c")["content"]
    assert "int value" in tools.read("src/value.c")["content"]
    tools.apply_patch(
        """diff --git a/src/value.c b/src/value.c
--- a/src/value.c
+++ b/src/value.c
@@ -1 +1 @@
-int value = 1;
+int value = 2;
"""
    )
    assert "value = 2" in (workspace / "src" / "value.c").read_text()
    assert [event["tool"] for event in tools.events] == [
        "search",
        "read",
        "apply_patch",
    ]


def test_tools_reject_escape_and_out_of_scope_patch(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    tools = RepositoryTools(workspace, {"allowed_paths": ["src/value.c"]})
    with pytest.raises(ToolError):
        tools.read("../secret")
    with pytest.raises(ToolError):
        tools.apply_patch(
            """diff --git a/README b/README
--- /dev/null
+++ b/README
@@ -0,0 +1 @@
+no
"""
        )
