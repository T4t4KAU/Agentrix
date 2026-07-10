from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


DEFAULT_PATHS = {
    "swebench": Path("data/swebench_verified.jsonl"),
    "agencybench": Path("data/agencybench_v2.jsonl"),
    "agentboard": Path("data/AgentBoard"),
    "appworld": Path("data/appworld"),
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
    if dataset == "agentboard":
        if source.is_file():
            return list(iter_jsonl(source))
        return _load_agentboard(source)
    if dataset == "appworld":
        if source.is_file():
            return list(iter_jsonl(source))
        return _load_appworld(source)
    return list(iter_jsonl(source))


def _load_agentboard(source: Path) -> list[dict[str, Any]]:
    prompt_root = source / "agentboard" / "prompts"
    records: list[dict[str, Any]] = []
    for path in sorted((prompt_root / "VanillaAgent").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        domains = (
            payload.items() if isinstance(payload, dict) else [(path.stem, payload)]
        )
        for domain, value in domains:
            if not isinstance(value, dict):
                continue
            instruction = value.get("instruction") or value.get("system_message")
            examples = value.get("examples") or value.get("in_context_examples") or []
            if instruction or examples:
                records.append(
                    {
                        "task": path.stem,
                        "domain": str(domain),
                        "instruction": instruction or "",
                        "examples": examples,
                    }
                )
    for path in sorted((prompt_root / "Raw").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        records.append(
            {
                "task": path.stem,
                "domain": "tool",
                "instruction": payload.get("system_message", ""),
                "tools": payload.get("tool_set_message", []),
                "examples": payload.get("in_context_examples", []),
            }
        )
    if not records:
        raise ValueError(f"no AgentBoard prompts found under {prompt_root}")
    return records


def _load_appworld(source: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    demos_path = (
        source / "experiments" / "prompts" / "function_calling_agent" / "demos.json"
    )
    if demos_path.exists():
        messages = json.loads(demos_path.read_text(encoding="utf-8"))
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if content:
                records.append(
                    {
                        "task_id": f"function-calling-demo-{index}",
                        "agent": "function_calling_agent",
                        "instruction": content,
                    }
                )
    for path in sorted(
        (source / "experiments" / "prompts").glob("*/*instructions.txt")
    ):
        instruction = path.read_text(encoding="utf-8").strip()
        if instruction:
            records.append(
                {
                    "task_id": path.parent.name,
                    "agent": path.parent.name,
                    "instruction": instruction,
                }
            )
    if not records:
        raise ValueError(f"no AppWorld prompts found under {source}")
    return records


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
    if dataset == "agentboard":
        return (
            "You are operating in an AgentBoard environment.\n\n"
            f"Task family: {record.get('task', '')}\n"
            f"Domain: {record.get('domain', '')}\n\n"
            f"Agent instructions:\n{record.get('instruction', '')}\n\n"
            f"Available tools:\n{json.dumps(record.get('tools', []), ensure_ascii=False)}\n\n"
            f"Demonstrations:\n{json.dumps(record.get('examples', []), ensure_ascii=False)}\n\n"
            "Plan the next reliable actions while respecting the environment rules."
        )
    if dataset == "appworld":
        return (
            "You are solving an AppWorld task with stateful applications and APIs.\n\n"
            f"Agent: {record.get('agent', '')}\n"
            f"Task ID: {record.get('task_id', '')}\n\n"
            f"Instructions:\n{record.get('instruction', '')}\n\n"
            "Produce a concise, verifiable action plan before executing API calls."
        )
    raise ValueError(f"unsupported dataset: {dataset}")


def inspect_datasets(paths: dict[str, Path] | None = None) -> list[dict[str, Any]]:
    paths = paths or DEFAULT_PATHS
    report = []
    for name, path in paths.items():
        if not path.exists():
            report.append(
                {
                    "dataset": name,
                    "path": str(path),
                    "records": 0,
                    "fields": [],
                    "available": False,
                }
            )
            continue
        records = load_records(name, path)
        report.append(
            {
                "dataset": name,
                "path": str(path),
                "records": len(records),
                "fields": sorted(records[0]) if records else [],
                "available": True,
            }
        )
    return report
