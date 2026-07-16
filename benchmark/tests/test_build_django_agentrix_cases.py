import json
import importlib.util
import subprocess
from argparse import Namespace
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "build_django_agentrix_cases.py"
)
SPEC = importlib.util.spec_from_file_location("build_django_agentrix_cases", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
BRANCH_TASKS = MODULE.BRANCH_TASKS
build = MODULE.build


def test_builds_frozen_case_with_shared_parent(tmp_path: Path) -> None:
    repo = tmp_path / "django"
    repo.mkdir()
    subprocess.run(("git", "init", "-q", str(repo)), check=True)
    subprocess.run(("git", "-C", str(repo), "config", "user.email", "test@example.com"), check=True)
    subprocess.run(("git", "-C", str(repo), "config", "user.name", "Test"), check=True)
    source = repo / "django" / "sample.py"
    source.parent.mkdir()
    source.write_text("def sample():\n    return 1\n" * 100)
    subprocess.run(("git", "-C", str(repo), "add", "."), check=True)
    subprocess.run(("git", "-C", str(repo), "commit", "-qm", "fixture"), check=True)
    specs = tmp_path / "specs.json"
    specs.write_text(
        json.dumps(
            [
                {
                    "case_id": "case",
                    "subsystem": "sample",
                    "task": "Investigate sample.",
                    "paths": ["django/sample.py"],
                }
            ]
        )
    )
    cases = build(
        Namespace(repo=repo, specs=specs, target_tokens=256, encoding="o200k_base")
    )
    assert len(cases) == 1
    assert len(cases[0]["branches"]) == len(BRANCH_TASKS) == 16
    assert cases[0]["shared_parent_tokens"] >= 256
    assert cases[0]["shared_parent_sha256"]
    assert cases[0]["source_segments"]
    trajectories = [branch["trajectory"] for branch in cases[0]["branches"]]
    assert [len(item) for item in trajectories] == [1] * 4 + [2] * 8 + [3] * 4
    assert sum(
        turn.get("tool") == "repository_search"
        for item in trajectories
        for turn in item
    ) == 12
    assert sum(
        turn.get("tool") == "reviewer_feedback"
        for item in trajectories
        for turn in item
    ) == 4
