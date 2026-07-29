from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from coding_agent_tools import RepositoryTools, ToolError
from coding_task_oracle import evaluate, load_task, prepare
from django_agentrix_runner import Runtime, build_graph


ACTION_PROMPT = """You are the parent coding agent. Use the subagent reports and repository tools to resolve the task.
Return exactly one JSON object per turn, without Markdown.
Allowed actions:
{"action":"search","pattern":"...","glob":"*.py"}
{"action":"read","path":"relative/path","start_line":1,"end_line":240}
{"action":"apply_patch","patch":"unified git diff"}
{"action":"public_test"}
{"action":"diff"}
{"action":"final","summary":"..."}
Inspect evidence before editing. Run the public test after editing. Do not edit tests.
Search for exact symbols before reading files. Read narrow line ranges and never repeat an identical action.
Once the faulty implementation is identified, apply a focused patch and run public_test; do not spend all steps browsing.
For apply_patch, emit a valid unified git diff using exact context you just read. Omit invented index hashes.
A final action is accepted only after a patch is applied and its latest public_test passes."""


def parse_action(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        stripped = stripped.rsplit("```", 1)[0].strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response contains no JSON object")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict) or not isinstance(value.get("action"), str):
        raise ValueError("response is not an action object")
    return value


def execute_action(tools: RepositoryTools, action: dict[str, Any]) -> dict[str, Any]:
    name = action["action"]
    if name == "search":
        return tools.search(str(action["pattern"]), str(action.get("glob", "*")))
    if name == "read":
        return tools.read(
            str(action["path"]),
            int(action.get("start_line", 1)),
            int(action.get("end_line", 400)),
        )
    if name == "apply_patch":
        return tools.apply_patch(str(action["patch"]))
    if name == "public_test":
        return tools.public_test()
    if name == "diff":
        return tools.diff()
    raise ToolError(f"unknown action: {name}")


def public_test_passed(event: dict[str, Any]) -> bool:
    """Return whether a repository public-test event completed successfully."""
    try:
        payload = json.loads(event["content"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return False
    return payload.get("returncode") == 0


def finalization_blocker(
    *, patch_applied: bool, passing_public_test: bool
) -> str | None:
    """Require a real edit-and-test loop before accepting a final action."""
    if not patch_applied:
        return (
            "Cannot finish yet: no patch has been applied. Inspect the faulty "
            "implementation and use apply_patch."
        )
    if not passing_public_test:
        return (
            "Cannot finish yet: the latest public test after the patch has not "
            "passed. Run public_test and revise the patch if needed."
        )
    return None


def select_case(
    cases: list[dict[str, Any]],
    *,
    case_id: str | None,
    task_id: str | None,
) -> dict[str, Any]:
    """Select exactly one case by case ID or executable-oracle task ID."""
    if bool(case_id) == bool(task_id):
        raise ValueError("provide exactly one of case_id and task_id")
    key = "case_id" if case_id else "oracle_task_id"
    value = case_id or task_id
    matches = [case for case in cases if case.get(key) == value]
    if len(matches) != 1:
        raise ValueError(f"expected one case for {key}={value!r}, found {len(matches)}")
    return matches[0]


def summarize_request_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Return quality-adjusted performance inputs from one executable task."""
    latencies = [float(item["latency_ms"]) for item in metrics]
    ttfts = [
        float(item["ttft_ms"])
        for item in metrics
        if item.get("ttft_ms") is not None
    ]
    return {
        "request_count": len(metrics),
        "input_tokens": sum(int(item["input_tokens"]) for item in metrics),
        "output_tokens": sum(int(item["output_tokens"]) for item in metrics),
        "request_latency_mean_ms": (
            statistics.fmean(latencies) if latencies else None
        ),
        "ttft_mean_ms": statistics.fmean(ttfts) if ttfts else None,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = select_case(cases, case_id=args.case_id, task_id=args.task_id)
    index = json.loads((args.task_root / "index.json").read_text(encoding="utf-8"))
    entry = next(
        item for item in index["tasks"] if item["task_id"] == selected["oracle_task_id"]
    )
    task_path = (args.task_root / entry["manifest"]).resolve()
    task, _ = load_task(task_path)
    workspace = args.workspace or Path(
        tempfile.mkdtemp(prefix=f"agentrix_e2e_{args.case_id}_.")
    )
    seeded = prepare(task_path, args.repo, workspace)

    client = AsyncOpenAI(api_key="local", base_url=args.base_url, timeout=900)
    runtime = Runtime(client, args.model, 1, {
        round_index: sum(
            min(len(branch.get("trajectory", [])) or 1, args.rounds) >= round_index
            for branch in selected["branches"]
        )
        for round_index in range(1, args.rounds + 1)
    })
    graph = build_graph(
        runtime,
        args.branch_output_tokens,
        args.rounds,
        args.trajectory_mode,
        args.prompt_compaction,
    )
    started = time.perf_counter()
    state = await graph.ainvoke({"case": selected, "branch_results": []})
    branch_wall_ms = (time.perf_counter() - started) * 1000

    reports = [
        {
            "branch_id": item["branch_id"],
            "final_report": item["text"][: args.max_report_chars],
        }
        for item in state["branch_results"]
    ]
    base_messages = [
        *selected["shared_messages"],
        {
            "role": "user",
            "content": (
                f"Coding task:\n{task['issue']}\n\n"
                f"Subagent reports:\n{json.dumps(reports, ensure_ascii=False)}\n\n"
                f"{ACTION_PROMPT}"
            ),
        },
    ]
    tools = RepositoryTools(workspace, task, max_output_bytes=args.max_tool_output_bytes)
    actions = []
    history: list[dict[str, str]] = []
    final_summary = ""
    parse_failures = 0
    seen_actions: set[str] = set()
    patch_applied = False
    passing_public_test = False
    for step in range(args.max_tool_steps):
        compact_summary = []
        if len(history) > 4:
            compact_summary.append(
                {
                    "role": "user",
                    "content": "Earlier actions: "
                    + ", ".join(str(action["action"]) for action in actions[:-2]),
                }
            )
        if step >= args.max_tool_steps // 2:
            compact_summary.append(
                {
                    "role": "user",
                    "content": (
                        f"Only {args.max_tool_steps - step} actions remain. "
                        "Stop broad or repeated reading. Locate the exact symbol, "
                        "apply the smallest patch, then run public_test."
                    ),
                }
            )
        messages = [*base_messages, *compact_summary, *history[-4:]]
        text, _ = await runtime.complete(
            case_id=selected["case_id"],
            stage="parent_decision",
            branch_id=None,
            round_index=step + 1,
            messages=messages,
            max_tokens=args.parent_output_tokens,
        )
        history.append({"role": "assistant", "content": text})
        try:
            action = parse_action(text)
        except (ValueError, json.JSONDecodeError) as error:
            parse_failures += 1
            history.append(
                {"role": "user", "content": f"Invalid action JSON: {error}. Return one valid action."}
            )
            continue
        actions.append(action)
        if action["action"] == "final":
            blocker = finalization_blocker(
                patch_applied=patch_applied,
                passing_public_test=passing_public_test,
            )
            if blocker:
                history.append({"role": "user", "content": blocker})
                continue
            final_summary = str(action.get("summary", ""))
            break
        signature = json.dumps(action, sort_keys=True, ensure_ascii=False)
        if signature in seen_actions:
            observation = "Tool error: identical action already executed. Choose a different, more targeted action."
        else:
            seen_actions.add(signature)
            try:
                event = execute_action(tools, action)
                observation = event["content"]
                if action["action"] == "apply_patch":
                    patch_applied = True
                    passing_public_test = False
                elif action["action"] == "public_test" and patch_applied:
                    passing_public_test = public_test_passed(event)
            except (ToolError, KeyError, ValueError) as error:
                observation = f"Tool error: {error}"
        history.append(
            {
                "role": "user",
                "content": f"Tool result for `{action['action']}`:\n{observation}\nChoose the next action.",
            }
        )
    await client.close()
    score = evaluate(task_path, workspace)
    request_metrics = [asdict(metric) for metric in runtime.metrics]
    total_wall_time_ms = (time.perf_counter() - started) * 1000
    return {
        "schema_version": 1,
        "workload": "coding_agent_executable_e2e",
        "case_id": selected["case_id"],
        "task_id": task["task_id"],
        "repository": task["repository"],
        "workspace": str(workspace),
        "seed_public_test_returncode": seeded["seed_public_test_returncode"],
        "resolved": score["resolved"],
        "score": score,
        "branch_wall_time_ms": branch_wall_ms,
        "total_wall_time_ms": total_wall_time_ms,
        "subagent_count": len(state["branch_results"]),
        "action_count": len(actions),
        "parse_failures": parse_failures,
        "actions": actions,
        "tool_events": tools.events,
        "final_summary": final_summary,
        "requests": request_metrics,
        "request_summary": summarize_request_metrics(request_metrics),
        "prompt_compaction": args.prompt_compaction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run executable coding-agent E2E task")
    parser.add_argument("--base-url", default="http://127.0.0.1:9000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case-id")
    selection.add_argument("--task-id")
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--trajectory-mode", choices=("live", "replay"), default="live")
    parser.add_argument("--prompt-compaction", action="store_true")
    parser.add_argument("--branch-output-tokens", type=int, default=128)
    parser.add_argument("--parent-output-tokens", type=int, default=512)
    parser.add_argument("--max-tool-steps", type=int, default=14)
    parser.add_argument("--max-tool-output-bytes", type=int, default=4096)
    parser.add_argument("--max-report-chars", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                k: payload[k]
                for k in (
                    "case_id",
                    "task_id",
                    "resolved",
                    "subagent_count",
                    "action_count",
                    "parse_failures",
                    "branch_wall_time_ms",
                    "total_wall_time_ms",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
