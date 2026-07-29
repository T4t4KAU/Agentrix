#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from coding_task_oracle import evaluate, load_task, prepare


SOURCE_DIRECTORIES = {
    "django/django": "django",
    "sqlite/sqlite": "sqlite",
    "FFmpeg/FFmpeg": "ffmpeg",
}


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "workload": "coding_oracle_preflight",
        "task_count": len(rows),
        "passed": sum(row["status"] == "passed" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "tasks": rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate every formal coding oracle by injecting its defect and "
            "then applying the exact inverse mutation as a gold repair"
        )
    )
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    index = json.loads((args.task_root / "index.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for entry in index["tasks"]:
        if not entry.get("score_in_formal_accuracy", False):
            continue
        task_path = (args.task_root / entry["manifest"]).resolve()
        task, task_dir = load_task(task_path)
        started = time.perf_counter()
        row: dict[str, Any] = {
            "task_id": task["task_id"],
            "repository": task["repository"],
            "revision": task["revision"],
        }
        try:
            source = args.source_root / SOURCE_DIRECTORIES[task["repository"]]
            with tempfile.TemporaryDirectory(
                prefix=f"agentrix_oracle_{task['task_id']}_."
            ) as temporary:
                workspace = Path(temporary)
                seeded = prepare(task_path, source, workspace)
                subprocess.run(
                    (
                        "git",
                        "apply",
                        "--reverse",
                        str((task_dir / task["mutation_patch"]).resolve()),
                    ),
                    cwd=workspace,
                    check=True,
                )
                score = evaluate(task_path, workspace)
            row.update(
                {
                    "status": "passed" if score["resolved"] else "failed",
                    "seed_public_test_returncode": seeded[
                        "seed_public_test_returncode"
                    ],
                    "gold_score": score,
                }
            )
            if not score["resolved"]:
                row["error"] = "inverse mutation did not satisfy the full oracle"
        except Exception as error:
            row.update({"status": "failed", "error": repr(error)})
        row["wall_time_ms"] = (time.perf_counter() - started) * 1000
        rows.append(row)
        write_report(args.output, rows)
        print(f"{row['status']}: {row['task_id']}", flush=True)

    if any(row["status"] == "failed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
