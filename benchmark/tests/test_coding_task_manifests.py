import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_indexed_coding_tasks_have_required_artifacts() -> None:
    task_root = ROOT / "coding_tasks"
    index = json.loads((task_root / "index.json").read_text())
    identifiers = set()
    for entry in index["tasks"]:
        manifest_path = task_root / entry["manifest"]
        task = json.loads(manifest_path.read_text())
        assert task["task_id"] == entry["task_id"]
        assert task["task_id"] not in identifiers
        identifiers.add(task["task_id"])
        task_dir = manifest_path.parent
        assert (task_dir / task["mutation_patch"]).is_file()
        assert (task_dir / task["public_test"]).is_file()
        assert (task_dir / task["hidden_test"]).resolve().is_file()
        assert task["allowed_paths"]
        assert task["configure"]
        assert task["build"]
        assert task["timeout_seconds"] > 0


def test_historical_tasks_declare_provenance() -> None:
    task_root = ROOT / "coding_tasks"
    index = json.loads((task_root / "index.json").read_text())
    historical = [
        item
        for item in index["tasks"]
        if item["classification"] == "historical_regression"
    ]
    assert historical
    for entry in historical:
        task = json.loads((task_root / entry["manifest"]).read_text())
        assert task["provenance_commit"]
