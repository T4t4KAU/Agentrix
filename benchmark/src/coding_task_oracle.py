from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any


def load_task(path: Path) -> tuple[dict[str, Any], Path]:
    task_path = path.resolve()
    return json.loads(task_path.read_text(encoding="utf-8")), task_path.parent


def run_command(
    spec: dict[str, Any], workspace: Path, timeout: int
) -> dict[str, Any]:
    started = time.perf_counter()
    argv = [
        value.format(python=sys.executable, workspace=str(workspace))
        for value in spec["argv"]
    ]
    result = subprocess.run(
        tuple(argv),
        cwd=workspace / spec.get("cwd", "."),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return {
        "argv": argv,
        "cwd": spec.get("cwd", "."),
        "returncode": result.returncode,
        "wall_time_ms": (time.perf_counter() - started) * 1000,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def export_revision(repo: Path, revision: str, workspace: Path) -> None:
    if not (repo / ".git").exists():
        # Server deployments may use a trusted pinned git-archive snapshot to
        # avoid transferring repository history. The caller still supplies the
        # revision recorded by the task manifest for result provenance.
        shutil.copytree(repo, workspace, dirs_exist_ok=True)
        return
    archive = subprocess.run(
        ("git", "-C", str(repo), "archive", "--format=tar", revision),
        capture_output=True,
        check=True,
    ).stdout
    with tempfile.NamedTemporaryFile(suffix=".tar") as handle:
        handle.write(archive)
        handle.flush()
        with tarfile.open(handle.name) as bundle:
            bundle.extractall(workspace, filter="data")


def prepare(task_path: Path, repo: Path, workspace: Path) -> dict[str, Any]:
    task, task_dir = load_task(task_path)
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError(f"workspace must not exist or be empty: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    export_revision(repo.resolve(), task["revision"], workspace)
    mutation = (task_dir / task["mutation_patch"]).resolve()
    subprocess.run(
        ("git", "apply", str(mutation)), cwd=workspace, check=True
    )
    agentrix_dir = workspace / ".agentrix"
    agentrix_dir.mkdir()
    public_source = (task_dir / task["public_test"]).resolve()
    public_target = agentrix_dir / "public_test.py"
    shutil.copy2(public_source, public_target)
    public_digest = hashlib.sha256(public_target.read_bytes()).hexdigest()
    state = {
        "task_id": task["task_id"],
        "revision": task["revision"],
        "public_test_sha256": public_digest,
    }
    (agentrix_dir / "task_state.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )
    subprocess.run(("git", "init", "-q"), cwd=workspace, check=True)
    subprocess.run(
        ("git", "config", "user.email", "oracle@invalid.local"),
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Agentrix Task Oracle"),
        cwd=workspace,
        check=True,
    )
    subprocess.run(("git", "add", "."), cwd=workspace, check=True)
    subprocess.run(
        ("git", "commit", "-qm", "seed task baseline"),
        cwd=workspace,
        check=True,
    )
    (workspace / "build").mkdir()
    commands = []
    for command in (*task.get("configure", []), *task.get("build", [])):
        result = run_command(command, workspace, task["timeout_seconds"])
        commands.append(result)
        if result["returncode"] != 0:
            raise RuntimeError(json.dumps(result, indent=2))
    public = subprocess.run(
        tuple(
            value.format(python=sys.executable, workspace=str(workspace))
            for value in task["public_test_command"]
        ),
        cwd=workspace,
        text=True,
        capture_output=True,
        timeout=task["timeout_seconds"],
        check=False,
    )
    if public.returncode == 0:
        raise RuntimeError("seeded defect unexpectedly passes its public test")
    return {
        "task_id": task["task_id"],
        "workspace": str(workspace.resolve()),
        "issue": task["issue"],
        "seed_public_test_returncode": public.returncode,
        "commands": commands,
    }


def evaluate(task_path: Path, workspace: Path) -> dict[str, Any]:
    task, task_dir = load_task(task_path)
    workspace = workspace.resolve()
    state_path = workspace / ".agentrix" / "task_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    public_path = workspace / ".agentrix" / "public_test.py"
    public_unchanged = (
        hashlib.sha256(public_path.read_bytes()).hexdigest()
        == state["public_test_sha256"]
    )
    changed = subprocess.check_output(
        ("git", "diff", "--name-only", "HEAD"), cwd=workspace, text=True
    ).splitlines()
    allowed = set(task["allowed_paths"])
    scope_valid = set(changed).issubset(allowed)
    command_results = []
    for command in task.get("build", []):
        result = run_command(command, workspace, task["timeout_seconds"])
        command_results.append(result)
        if result["returncode"] != 0:
            break
    build_passed = all(x["returncode"] == 0 for x in command_results)

    test_results = {}
    if build_passed and public_unchanged and scope_valid:
        commands = {
            "public": task["public_test_command"],
            "hidden": [
                value.format(
                    python=sys.executable,
                    hidden_test=str((task_dir / task["hidden_test"]).resolve()),
                    workspace=str(workspace),
                )
                for value in task["hidden_test_command"]
            ],
        }
        for name, argv in commands.items():
            result = run_command(
                {"argv": argv}, workspace, task["timeout_seconds"]
            )
            test_results[name] = result
    resolved = (
        build_passed
        and public_unchanged
        and scope_valid
        and bool(changed)
        and all(x["returncode"] == 0 for x in test_results.values())
        and set(test_results) == {"public", "hidden"}
    )
    return {
        "schema_version": 1,
        "task_id": task["task_id"],
        "resolved": resolved,
        "changed_paths": changed,
        "scope_valid": scope_valid,
        "public_test_unchanged": public_unchanged,
        "build_passed": build_passed,
        "build": command_results,
        "tests": test_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or score coding tasks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--task", type=Path, required=True)
    prepare_parser.add_argument("--repo", type=Path, required=True)
    prepare_parser.add_argument("--workspace", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--task", type=Path, required=True)
    evaluate_parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.task, args.repo, args.workspace)
    else:
        result = evaluate(args.task, args.workspace)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
