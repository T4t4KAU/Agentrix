#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import tiktoken


BRANCH_TASKS = (
    "Trace the primary execution path and identify the most likely state transition responsible for the regression.",
    "Search for ownership, cloning, copying, or mutation invariants that constrain a safe correction.",
    "Analyze the public API and backwards-compatibility contract implicated by the task.",
    "Inspect existing tests and identify the smallest fail-to-pass regression test shape.",
    "Look for analogous implementations elsewhere in the repository that establish the intended design pattern.",
    "Analyze error handling and cleanup paths, including behavior after a partial failure.",
    "Check synchronous versus asynchronous execution and identify any divergent behavior.",
    "Check database-backend or platform-specific assumptions without requiring a non-SQLite backend.",
    "Propose candidate patch A, prioritizing the smallest localized change.",
    "Propose candidate patch B, prioritizing explicit state ownership even if it touches more than one function.",
    "Design adversarial edge cases that could make a plausible patch incomplete.",
    "Review likely performance implications and identify work that must not move into a hot loop.",
    "Review serialization, deepcopy, pickling, or cached-property implications where relevant.",
    "Identify regression risks in neighboring subsystems and select focused pass-to-pass tests.",
    "Act as a skeptical code reviewer: challenge the task assumptions and the most obvious proposed fix.",
    "Synthesize an independent diagnosis with file and symbol evidence suitable for the parent coding agent.",
)

COMMIT_ANALYSIS_BRANCH_TASKS = (
    "Trace the affected execution path and summarize the observable behavior.",
    "Explain the ownership and state-transition invariants touched by the change.",
    "Describe the public API contract and compatibility considerations.",
    "Inspect related tests and explain which behavior they establish.",
    "Find analogous implementations that clarify the intended design pattern.",
    "Analyze error handling and cleanup behavior without proposing code changes.",
    "Compare synchronous and asynchronous paths where both exist.",
    "Identify platform or backend assumptions visible in the supplied context.",
    "Summarize the data flow across the affected functions and modules.",
    "Explain how the change interacts with cached or derived state.",
    "List ordinary functional edge cases relevant to understanding the behavior.",
    "Assess likely performance implications along the affected hot paths.",
    "Explain serialization or persistence implications where relevant.",
    "Identify neighboring subsystems whose behavior should remain unchanged.",
    "Act as a skeptical reviewer and separate confirmed evidence from inference.",
    "Synthesize a read-only design analysis with file and symbol evidence.",
)


def build_trajectory(
    *,
    index: int,
    subsystem: str,
    task: str,
    paths: list[str],
    source_sections: list[dict[str, Any]],
    branch_tasks: tuple[str, ...],
    analysis_only: bool,
) -> list[dict[str, Any]]:
    trajectory = [
        {
            "stage": "triage",
            "instruction": (
                f"Private investigation {index + 1}/{len(branch_tasks)} "
                f"for {subsystem}:\n{task}"
            ),
        }
    ]
    # Four agents terminate after triage, eight use one tool observation, and
    # four difficult branches proceed to an independent review stage.
    if index >= 4:
        inspected = paths[index % len(paths)]
        repeated_section = source_sections[index % len(source_sections)]
        trajectory.append(
            {
                "stage": "tool_followup",
                "tool": "repository_search",
                "tool_observation": (
                    f"Repository search completed in {inspected}. The frozen "
                    "snapshot is authoritative; no unlisted source was loaded."
                ),
                "tool_sections": [repeated_section],
                "instruction": (
                    "Re-evaluate the hypothesis using this tool observation, "
                    "reject unsupported claims, and identify the strongest "
                    "discriminating test."
                ),
            }
        )
    if index >= 12:
        trajectory.append(
            {
                "stage": "review",
                "tool": "reviewer_feedback",
                "tool_observation": (
                    "The reviewer requires confirmed evidence to be separated "
                    "from inference and all ownership and cleanup risks resolved."
                ),
                "instruction": (
                    "Return the final read-only behavior analysis and evidence "
                    "summary while preserving the required JSON schema."
                    if analysis_only
                    else "Return the final implementation recommendation and an "
                    "ordered test plan while preserving the required JSON schema."
                ),
            }
        )
    return trajectory


SYSTEM_PROMPT = """You are a read-only repository coding subagent working for a parent coding agent.
Analyze the frozen repository snapshot and the assigned private investigation. Do not invent files or symbols.
Return a detailed JSON report with keys hypothesis, evidence, affected_symbols, recommended_change, tests, and risks.
Evidence entries must cite repository-relative paths and symbol names. Do not emit a patch."""

COMMIT_ANALYSIS_SYSTEM_PROMPT = """You are a read-only repository analysis subagent working for a parent coding agent.
Analyze the frozen repository snapshot and the assigned ordinary functional commit-analysis question.
Do not suggest or perform code modifications.
Return a detailed JSON report with keys hypothesis, evidence, affected_symbols, behavior, tests, and risks.
Evidence entries must cite repository-relative paths and symbol names."""


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()


def repository_map(repo: Path, paths: list[str]) -> str:
    directories = sorted({str(Path(path).parent) for path in paths})
    lines: list[str] = []
    for directory in directories:
        root = repo / directory
        if not root.is_dir():
            continue
        entries = sorted(
            str(path.relative_to(repo))
            for path in root.iterdir()
            if path.is_file()
            and path.suffix in {".c", ".h", ".mak", ".py", ".rst", ".txt", ".y"}
        )
        lines.extend(entries)
    return "\n".join(lines)


def render_section(path: str, content: str) -> str:
    return (
        f"\n\n===== BEGIN FILE: {path} =====\n{content}\n===== END FILE: {path} ====="
    )


def fit_parent(
    *,
    repo: Path,
    spec: dict[str, Any],
    target_tokens: int,
    encoding: Any,
    repository_name: str,
) -> tuple[str, list[dict[str, Any]]]:
    commit = git_output(repo, "rev-parse", "HEAD")
    analysis_only = spec.get("case_kind") == "commit_analysis"
    constraints = (
        "- Treat this checkout as a frozen read-only snapshot.\n"
        "- Explain ordinary functional behavior using supplied source and tests.\n"
        "- Do not suggest or perform code modifications."
        if analysis_only
        else "- Treat this checkout as a frozen read-only snapshot.\n"
        "- Base every claim on the supplied source or tests.\n"
        "- Preserve documented behavior and backwards compatibility.\n"
        "- Prefer focused SQLite-compatible regression tests."
    )
    prefix = f"""{repository_name.upper()} CODING CASE
Case: {spec["case_id"]}
Subsystem: {spec["subsystem"]}
Repository commit: {commit}

Parent task:
{spec["task"]}

Constraints:
{constraints}

REPOSITORY MAP
{repository_map(repo, spec["paths"])}

FROZEN SOURCE CONTEXT"""
    text = prefix
    segments: list[dict[str, Any]] = []
    sources: list[tuple[str, str, list[int]]] = []
    for relative in spec["paths"]:
        source = repo / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        content = source.read_text(encoding="utf-8", errors="replace")
        section = render_section(relative, content)
        sources.append((relative, content, encoding.encode(section)))
    budget = max(0, target_tokens - len(encoding.encode(prefix)))
    allocations = [0] * len(sources)
    active = set(range(len(sources)))
    while budget > 0 and active:
        share = max(1, budget // len(active))
        progressed = False
        for index in list(active):
            available = len(sources[index][2]) - allocations[index]
            take = min(share, available, budget)
            allocations[index] += take
            budget -= take
            progressed = progressed or take > 0
            if allocations[index] >= len(sources[index][2]):
                active.remove(index)
            if budget <= 0:
                break
        if not progressed:
            break
    for (relative, content, tokens), allocated in zip(
        sources, allocations, strict=True
    ):
        if allocated <= 0:
            continue
        section = encoding.decode(tokens[:allocated])
        text += section
        rendered_tokens = len(encoding.encode(section))
        segments.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(content.encode()).hexdigest(),
                "rendered_tokens": rendered_tokens,
                "truncated": rendered_tokens < len(tokens),
                "prompt_section": {
                    "segment_id": f"source:{relative}",
                    "content": section,
                    "heading": None,
                },
            }
        )
    return text, segments


def build(args: argparse.Namespace) -> list[dict[str, Any]]:
    repo = args.repo.resolve()
    if not (repo / ".git").exists() and not (repo / ".git").is_file():
        raise ValueError(f"not a git worktree: {repo}")
    if git_output(repo, "status", "--porcelain"):
        raise ValueError(f"repository checkout must be clean: {repo}")
    specs = json.loads(args.specs.read_text(encoding="utf-8"))
    encoding = tiktoken.get_encoding(args.encoding)
    cases = []
    for spec in specs:
        analysis_only = spec.get("case_kind") == "commit_analysis"
        branch_tasks = COMMIT_ANALYSIS_BRANCH_TASKS if analysis_only else BRANCH_TASKS
        parent, segments = fit_parent(
            repo=repo,
            spec=spec,
            target_tokens=args.target_tokens,
            encoding=encoding,
            repository_name=getattr(args, "repository_name", "Django"),
        )
        messages = [
            {
                "role": "system",
                "content": (
                    COMMIT_ANALYSIS_SYSTEM_PROMPT if analysis_only else SYSTEM_PROMPT
                ),
            },
            {"role": "user", "content": parent},
        ]
        cases.append(
            {
                "schema_version": 2,
                "case_id": spec["case_id"],
                "oracle_task_id": spec.get("oracle_task_id"),
                "case_kind": spec.get("case_kind", "functional_task"),
                "source_commit": spec.get("source_commit"),
                "source_subject": spec.get("source_subject"),
                "subsystem": spec["subsystem"],
                "repo": getattr(args, "repo_id", "django/django"),
                "repo_commit": git_output(repo, "rev-parse", "HEAD"),
                "source_manifest": (
                    (repo / args.manifest_file).read_text(encoding="utf-8").strip()
                    if getattr(args, "manifest_file", "")
                    else None
                ),
                "tokenizer": args.encoding,
                "shared_parent_tokens": sum(
                    len(encoding.encode(message["content"])) for message in messages
                ),
                "shared_parent_sha256": hashlib.sha256(
                    json.dumps(messages, sort_keys=True).encode()
                ).hexdigest(),
                "shared_messages": messages,
                "source_segments": segments,
                "known_prompt_sections": [
                    segment["prompt_section"] for segment in segments
                ],
                "branches": [
                    {
                        "branch_id": index,
                        "private_instruction": (
                            f"Private investigation {index + 1}/{len(branch_tasks)} "
                            f"for {spec['subsystem']}:\n{task}"
                        ),
                        "trajectory": build_trajectory(
                            index=index,
                            subsystem=spec["subsystem"],
                            task=task,
                            paths=spec["paths"],
                            source_sections=[
                                segment["prompt_section"] for segment in segments
                            ],
                            branch_tasks=branch_tasks,
                            analysis_only=analysis_only,
                        ),
                    }
                    for index, task in enumerate(branch_tasks)
                ],
            }
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build frozen long-prefix repository coding-agent cases"
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=30000)
    parser.add_argument("--encoding", default="o200k_base")
    parser.add_argument("--repo-id", default="django/django")
    parser.add_argument("--repository-name", default="Django")
    parser.add_argument("--manifest-file", default="")
    args = parser.parse_args()
    cases = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "cases": len(cases),
                "branches": sum(len(case["branches"]) for case in cases),
                "shared_parent_tokens": [
                    case["shared_parent_tokens"] for case in cases
                ],
                "repo_commit": cases[0]["repo_commit"] if cases else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
