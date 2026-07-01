from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


DEFAULT_PATHS = {
    "swebench": Path("data/swebench_verified.jsonl"),
    "agencybench": Path("data/agencybench_v2.jsonl"),
}


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def load_records(dataset: str, path: Path | None = None) -> list[dict[str, Any]]:
    source = path or DEFAULT_PATHS[dataset]
    if not source.exists():
        raise FileNotFoundError(f"dataset not found: {source}")
    return list(iter_jsonl(source))


def record_to_prompt(dataset: str, record: dict[str, Any]) -> str:
    if dataset == "swebench":
        return (
            "你正在分析一个真实软件缺陷。\n\n"
            f"仓库：{record['repo']}\n"
            f"基础提交：{record['base_commit']}\n"
            f"实例：{record['instance_id']}\n\n"
            f"问题描述：\n{record['problem_statement']}\n\n"
            f"FAIL_TO_PASS：\n{record.get('FAIL_TO_PASS', '')}\n\n"
            f"PASS_TO_PASS：\n{record.get('PASS_TO_PASS', '')}\n\n"
            "请分析根因、影响范围、验证方法，并提出修复方案。不要修改真实仓库。"
        )
    if dataset == "agencybench":
        subtasks = [
            record.get(f"subtask{index}", "")
            for index in range(1, int(record.get("subtask_count", 5)) + 1)
            if record.get(f"subtask{index}")
        ]
        return (
            "请分析以下长程 Agent 场景，规划完成所有子任务的可靠步骤。\n\n"
            f"类别：{record.get('category', '')}\n"
            f"场景：{record.get('scenario', '')}\n"
            f"场景 ID：{record.get('scenario_id', '')}\n\n"
            + "\n\n".join(
                f"子任务 {index}：\n{text}"
                for index, text in enumerate(subtasks, 1)
            )
        )
    raise ValueError(f"unsupported dataset: {dataset}")


def inspect_datasets(paths: dict[str, Path] | None = None) -> list[dict[str, Any]]:
    paths = paths or DEFAULT_PATHS
    report = []
    for name, path in paths.items():
        records = load_records(name, path)
        report.append(
            {
                "dataset": name,
                "path": str(path),
                "records": len(records),
                "fields": sorted(records[0]) if records else [],
            }
        )
    return report
