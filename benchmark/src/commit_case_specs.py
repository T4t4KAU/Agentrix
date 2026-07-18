from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


UNSAFE_PATTERN = re.compile(
    r"security|vulnerab\w*|cve[- ]?\d*|exploit\w*|attack\w*|malicious|"
    r"password\w*|credential\w*|authoriz\w*|permission\w*|authenticat\w*|"
    r"crash\w*|corrupt\w*|malformed|fuzz\w*|sanitizer|segfault|"
    r"overflow|underflow|overread|out[- ]of[- ]bounds|\boob\b|"
    r"buffer|pointer|memcpy|\bnull\b|freed?|allocat\w*|memory[- ]safety|"
    r"use[- ]after[- ]free|double[- ]free|jit[_ -]protect|large enough|"
    r"sha\d*sum|checksum|pgp|keyserver|control character|"
    r"outside.{0,20}range|static analy|validat\w*|input limits|"
    r"(?:^|/)(?:auth|csrf|crypto|hash|password|security)[^/]*\.",
    re.IGNORECASE,
)


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def is_safe_commit(subject: str, paths: list[str]) -> bool:
    searchable = f"{subject}\n" + "\n".join(paths)
    return UNSAFE_PATTERN.search(searchable) is None


def existing_context_paths(
    repo: Path,
    changed_paths: list[str],
    *,
    allowed_suffixes: set[str],
    max_paths: int,
) -> list[str]:
    tracked = [
        path
        for path in git_output(repo, "ls-files").splitlines()
        if Path(path).suffix.lower() in allowed_suffixes
        and (repo / path).is_file()
        and not UNSAFE_PATTERN.search(path)
    ]
    changed = [
        path
        for path in changed_paths
        if path in tracked and Path(path).suffix.lower() in allowed_suffixes
    ]
    if not changed:
        return []
    directories = {str(Path(path).parent) for path in changed}
    neighbors = [
        path
        for path in tracked
        if path not in changed and str(Path(path).parent) in directories
    ]
    remaining = [
        path for path in tracked if path not in changed and path not in neighbors
    ]
    return (changed + sorted(neighbors) + sorted(remaining))[:max_paths]


def generate_specs(
    *,
    repo: Path,
    repository_slug: str,
    count: int,
    allowed_suffixes: set[str],
    max_changed_files: int,
    max_context_paths: int,
    history_limit: int,
) -> list[dict[str, Any]]:
    specs = []
    commits = git_output(
        repo, "rev-list", f"--max-count={history_limit}", "--no-merges", "HEAD"
    ).splitlines()
    prefix = repository_slug.lower().replace("/", "_").replace("-", "_")
    for commit in commits:
        subject = git_output(repo, "show", "-s", "--format=%s", commit)
        changed_paths = [
            path
            for path in git_output(
                repo, "diff-tree", "--no-commit-id", "--name-only", "-r", commit
            ).splitlines()
            if path
        ]
        if (
            not changed_paths
            or len(changed_paths) > max_changed_files
            or not is_safe_commit(subject, changed_paths)
        ):
            continue
        paths = existing_context_paths(
            repo,
            changed_paths,
            allowed_suffixes=allowed_suffixes,
            max_paths=max_context_paths,
        )
        if not paths:
            continue
        short = commit[:12]
        specs.append(
            {
                "case_id": f"{prefix}_commit_{short}",
                "case_kind": "commit_analysis",
                "source_commit": commit,
                "source_subject": subject,
                "subsystem": f"Ordinary functional commit {short}",
                "task": (
                    f"Perform a read-only analysis of historical commit {short}: "
                    f"{subject}. Explain the ordinary functional behavior, affected "
                    "execution path, module relationships, and relevant tests using "
                    "the frozen source context. Keep the response strictly "
                    "read-only and do not suggest code modifications."
                ),
                "paths": paths,
            }
        )
        if len(specs) == count:
            break
    if len(specs) != count:
        raise RuntimeError(
            f"selected {len(specs)} safe functional commits, expected {count}"
        )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic read-only coding cases from safe commits"
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--repository-slug", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--allowed-suffixes", default=".py,.c,.h")
    parser.add_argument("--max-changed-files", type=int, default=12)
    parser.add_argument("--max-context-paths", type=int, default=16)
    parser.add_argument("--history-limit", type=int, default=2000)
    args = parser.parse_args()
    specs = generate_specs(
        repo=args.repo.resolve(),
        repository_slug=args.repository_slug,
        count=args.count,
        allowed_suffixes={
            suffix.strip().lower()
            for suffix in args.allowed_suffixes.split(",")
            if suffix.strip()
        },
        max_changed_files=args.max_changed_files,
        max_context_paths=args.max_context_paths,
        history_limit=args.history_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(specs, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "case_count": len(specs),
                "commits": [spec["source_commit"] for spec in specs],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
