#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


REPOSITORIES = {
    "django/django": "django",
    "sqlite/sqlite": "sqlite",
    "FFmpeg/FFmpeg": "ffmpeg",
}


def revisions(task_root: Path) -> dict[str, str]:
    index = json.loads((task_root / "index.json").read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for entry in index["tasks"]:
        task = json.loads(
            (task_root / entry["manifest"]).read_text(encoding="utf-8")
        )
        repository = str(task["repository"])
        revision = str(task["revision"])
        previous = result.setdefault(repository, revision)
        if previous != revision:
            raise ValueError(f"multiple revisions requested for {repository}")
    return result


def download_snapshot(repository: str, revision: str, destination: Path) -> None:
    url = f"https://codeload.github.com/{repository}/tar.gz/{revision}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agentrix_source_") as temporary:
        archive = Path(temporary) / "source.tar.gz"
        urllib.request.urlretrieve(url, archive)
        extract_root = Path(temporary) / "extract"
        extract_root.mkdir()
        with tarfile.open(archive) as bundle:
            bundle.extractall(extract_root, filter="data")
        children = list(extract_root.iterdir())
        if len(children) != 1 or not children[0].is_dir():
            raise RuntimeError(f"unexpected source archive layout for {repository}")
        shutil.copytree(children[0], destination)
    (destination / ".agentrix_revision").write_text(
        f"{repository}\n{revision}\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download pinned source snapshots for executable coding tasks"
    )
    parser.add_argument(
        "--task-root",
        type=Path,
        default=Path("benchmark/coding_tasks"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for repository, revision in revisions(args.task_root).items():
        destination = args.output_root / REPOSITORIES[repository]
        marker = destination / ".agentrix_revision"
        expected = f"{repository}\n{revision}\n"
        if marker.exists() and marker.read_text(encoding="utf-8") == expected:
            print(f"reuse {repository}@{revision}: {destination}")
            continue
        if destination.exists():
            raise ValueError(
                f"{destination} exists without the expected revision marker"
            )
        print(f"download {repository}@{revision}: {destination}")
        download_snapshot(repository, revision, destination)


if __name__ == "__main__":
    main()
