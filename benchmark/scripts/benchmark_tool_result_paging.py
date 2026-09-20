#!/usr/bin/env python3
"""Closed-loop tool paging ablation with immutable reports and an exact oracle.

The seeded reports are a controlled synthetic workload, not AgentX or a coding
quality benchmark. Both modes expose identical tools and data. The model chooses
which pages to read; the harness never supplies target rows automatically.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "application/src"), str(ROOT / "benchmark/src")]
from agentrix_application import PagedToolStore  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from resource_sampler import process_tree_rss_kib, sample_gpu  # noqa: E402
import aiohttp  # noqa: E402

SYSTEM = """You audit build reports. Use only supplied reports or tool results.
Return JSON actions with no Markdown or explanations. You may put parallel
search actions on separate lines. Return exactly one final action when finished.
Tools available in both modes:
{"action":"search","source":"builds","needle":"job-0001"}
{"action":"search","source":"checks","needle":"job-0001"}
{"action":"read","source":"builds","offset":0,"limit":768}
The source must be exactly "builds" or exactly "checks". Replace the example job
with the requested job. You may search each report using that exact job ID.
If the supplied reports already contain the evidence, answer directly without
calling tools. Tools are optional; use them only to retrieve missing evidence.
Search returns a bounded excerpt from the exact immutable report, not a live file.
When you have evidence from BOTH reports, return one final JSON object with:
action: "final"; revision: the exact eight-character revision from builds;
score: the actual integer from checks; decision: "reject" when that score is
less than 50, or "accept" when that score is 50 or greater.
Compute the decision from the observed score; do not copy example values.
Reports marked external contain only a preview: search or read them to get evidence.
"""


def reports(seed: int, rows: int):
    rng = random.Random(seed)
    builds, checks, truth = [], [], {}
    for index in range(rows):
        job = f"job-{index:04d}"
        revision = f"{rng.getrandbits(32):08x}"
        score = rng.randrange(100)
        builds.append(
            json.dumps(
                dict(
                    job=job,
                    revision=revision,
                    component=f"worker-{index % 17}",
                    timestamp=f"2026-09-19T12:{index % 60:02d}:00",
                )
            )
        )
        checks.append(
            json.dumps(
                dict(
                    job=job,
                    score=score,
                    suite=f"integration-{index % 13}",
                    checks=64,
                    status="complete",
                )
            )
        )
        truth[job] = dict(
            revision=revision,
            score=score,
            decision="accept" if score >= 50 else "reject",
        )
    return {"builds": "\n".join(builds), "checks": "\n".join(checks)}, truth


def parse(text):
    start, end = text.find("{"), text.rfind("}")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def parse_actions(text):
    decoder = json.JSONDecoder()
    remaining = text.strip()
    actions = []
    while remaining:
        value, end = decoder.raw_decode(remaining)
        if not isinstance(value, dict):
            raise ValueError("expected an action object")
        actions.append(value)
        remaining = remaining[end:].strip()
    if not actions or len(actions) > 4:
        raise ValueError("expected 1..4 actions")
    return actions


def validate_decision(action):
    """Check the public business rule against the model's OWN proposed score.

    Never consult the hidden report/oracle. Wrong evidence can still pass this
    consistency check and will count as a benchmark failure.
    """
    score = action.get("score")
    if type(score) is not int:
        raise ValueError("final score must be an integer from the tool evidence")
    if action.get("decision") != ("reject" if score < 50 else "accept"):
        raise ValueError(
            "decision contradicts your proposed score. Recheck the "
            "tool evidence: score < 50 requires reject, otherwise accept."
        )


def restore_reports(store, session_id, handles, max_chars):
    """Restore exact source reports within a bound; never consult an answer."""
    sizes = {
        name: store.read(session_id, handle, limit=1)["total_chars"]
        for name, handle in handles.items()
    }
    if sum(sizes.values()) > max_chars:
        return None
    return "\n\n".join(
        f"Report {name}:\n"
        + "".join(
            store.read(session_id, handle, offset=offset, limit=16384)["content"]
            for offset in range(0, sizes[name], 16384)
        )
        for name, handle in handles.items()
    )


async def run(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count_tokens = None
    if args.tokenizer:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))

        def count_tokens(text):
            return len(tokenizer.encode(text, add_special_tokens=False).ids)

    client = AsyncOpenAI(
        base_url=args.base_url.rstrip("/") + "/v1",
        api_key="local",
        timeout=180,
        max_retries=0,
    )
    requests, branches, reducers, memory = [], [], [], []
    transcripts, corpus_hashes, lifecycle = [], [], []
    semaphore = asyncio.Semaphore(args.concurrency)
    workflow_semaphore = asyncio.Semaphore(args.workflow_concurrency)
    stop = asyncio.Event()
    start = time.perf_counter()
    errors = []
    hardware = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    model_info = (await client.models.list()).model_dump()

    async def monitor():
        async with aiohttp.ClientSession() as http:
            while not stop.is_set():
                record = {"elapsed": time.perf_counter() - start}
                try:
                    async with http.get(
                        args.base_url.rstrip("/") + "/metrics",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as response:
                        metrics = await response.text()
                    record["kv_usage"] = max(
                        (
                            float(line.rsplit(" ", 1)[1])
                            for line in metrics.splitlines()
                            if line.startswith("vllm:kv_cache_usage_perc{")
                        ),
                        default=0,
                    )
                    for name in ("num_requests_running", "num_requests_waiting"):
                        record[name] = sum(
                            float(line.rsplit(" ", 1)[1])
                            for line in metrics.splitlines()
                            if line.startswith(f"vllm:{name}{{")
                        )
                except Exception as error:
                    record["error"] = str(error)
                memory.append(record)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=0.05)
                except asyncio.TimeoutError:
                    pass

    async def monitor_processes():
        # Driver and /proc queries can take a second. Keep them off the KV
        # sampling path so they do not hide short-lived working-set peaks.
        while not stop.is_set():
            record = {"elapsed": time.perf_counter() - start}
            try:
                record["gpus"] = await asyncio.to_thread(sample_gpu, {args.gpu})
                if args.server_pid:
                    record["server_rss_kib"] = await asyncio.to_thread(
                        process_tree_rss_kib, args.server_pid
                    )
                record["application_rss_kib"] = resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
            except Exception as error:
                record["error"] = str(error)
            memory.append(record)
            try:
                await asyncio.wait_for(stop.wait(), timeout=1)
            except asyncio.TimeoutError:
                pass

    async def complete(messages, identity):
        async with semaphore:
            began = time.perf_counter()
            first = None
            text = []
            usage = None
            stream = await client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=0,
                seed=args.seed,
                max_tokens=256,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            async for chunk in stream:
                if chunk.usage:
                    usage = chunk.usage.model_dump()
                if chunk.choices and chunk.choices[0].delta.content:
                    first = first or time.perf_counter()
                    text.append(chunk.choices[0].delta.content)
            if usage is None:
                raise RuntimeError("missing response usage")
            finished = time.perf_counter()
            content = "".join(text)
            requests.append(
                dict(
                    identity=identity,
                    latency=finished - began,
                    ttft=(first or finished) - began,
                    usage=usage,
                )
            )
            transcripts.append(dict(identity=identity, response=content))
            return content

    async def case(case_id, directory):
        bodies, expected = reports(args.seed + case_id, args.rows)
        corpus_hashes.append(
            {
                name: hashlib.sha256(body.encode()).hexdigest()
                for name, body in bodies.items()
            }
        )
        store = PagedToolStore(directory / f"case-{case_id}.sqlite")
        root = f"case-{case_id}"
        store.open_session(root)
        handles = {name: store.put(root, body) for name, body in bodies.items()}
        original_bytes = store.stats()["stored_bytes"]
        peak_stored_bytes = original_bytes
        targets = [
            f"job-{int((branch + 1) * args.rows / (args.branches + 1)):04d}"
            for branch in range(args.branches)
        ]
        evidence = "\n\n".join(
            f"Report {name}:\n"
            + (
                body
                if args.mode == "inline"
                else json.dumps(
                    dict(
                        external=True, source=name, chars=len(body), preview=body[:160]
                    )
                )
            )
            for name, body in bodies.items()
        )
        # Full bodies are not retained by the paged Agent conversation.
        del bodies
        shared = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": evidence},
        ]
        del evidence

        async def audit_stage(index, target, sid, stage):
            nonlocal peak_stored_bytes
            messages = [
                *shared,
                *([{"role": "user", "content": ""}] if count_tokens else []),
                {
                    "role": "user",
                    "content": f"Audit {target}. Find its revision in builds and its score in checks, then decide.",
                },
            ]
            answer = None
            tool_calls = 0
            validation_retries = 0
            context_restorations = 0
            context_samples = []
            began = time.perf_counter()
            try:
                for step in range(args.max_steps):
                    if count_tokens:
                        context = store.render_context(
                            sid,
                            count_tokens=count_tokens,
                            max_tokens=args.tool_context_tokens,
                        )
                        messages[2] = {"role": "user", "content": context.pop("text")}
                        context_samples.append(context)
                    content = await complete(
                        messages, f"{sid}/stage-{stage}/turn-{step}"
                    )
                    messages.append({"role": "assistant", "content": content})
                    try:
                        actions = parse_actions(content)
                        if len(actions) == 1 and actions[0].get("action") == "final":
                            try:
                                validate_decision(actions[0])
                            except ValueError:
                                validation_retries += 1
                                # Invalid output is not committed to Agent history.
                                messages.pop()
                                if (
                                    args.mode == "paged"
                                    and args.restore_on_validation_error
                                    and not context_restorations
                                ):
                                    # The public validator only checks a score
                                    # decision. Restore its source, not unrelated
                                    # build metadata. Existing retrieved build
                                    # evidence and tools stay available.
                                    restored = restore_reports(
                                        store,
                                        sid,
                                        {"checks": handles["checks"]},
                                        args.max_restore_chars,
                                    )
                                    if restored is not None:
                                        messages[1] = {
                                            "role": "user",
                                            "content": restored,
                                        }
                                        context_restorations += 1
                                        continue
                                raise
                            answer = actions[0]
                            break
                        observations = []
                        for action in actions:
                            if action["action"] == "list_observations":
                                observation = {
                                    "catalog": store.list_observations(
                                        sid, offset=action.get("offset", 0), limit=8
                                    )
                                }
                                tool_calls += 1
                                observations.append(observation)
                                continue
                            if action["action"] == "read_observation":
                                observation = store.read(
                                    sid,
                                    action["result_id"],
                                    offset=action.get("offset", 0),
                                    limit=action.get("limit", 768),
                                )
                                tool_calls += 1
                                observations.append(observation)
                                continue
                            source = handles[action["source"]]
                            if action["action"] == "search":
                                observation = store.search(
                                    sid,
                                    source,
                                    action["needle"],
                                    limit=args.search_excerpt_chars or 768,
                                )
                                if (
                                    not args.search_excerpt_chars
                                    and observation["match_offset"] is not None
                                ):
                                    relative = (
                                        observation["match_offset"]
                                        - observation["offset"]
                                    )
                                    body = observation["content"]
                                    begin = body.rfind("\n", 0, relative) + 1
                                    end = body.find("\n", relative)
                                    observation = {
                                        "matched_row": json.loads(
                                            body[begin : end if end >= 0 else None]
                                        )
                                    }
                            elif action["action"] == "read":
                                observation = store.read(
                                    sid,
                                    source,
                                    offset=action.get("offset", 0),
                                    limit=action.get("limit", 768),
                                )
                            else:
                                raise ValueError("unknown action")
                            tool_calls += 1
                            observation.pop("result_id", None)
                            observations.append(
                                {"source": action["source"], **observation}
                            )
                        feedback = json.dumps(observations)
                        if count_tokens:
                            store.record_observation(
                                sid, f"{target} step {step}", feedback
                            )
                            peak_stored_bytes = max(
                                peak_stored_bytes, store.stats()["stored_bytes"]
                            )
                            feedback = "Tool observations are in recent_results above. Choose the next action."
                    except (ValueError, KeyError, TypeError) as error:
                        feedback = "Tool/action error: " + str(error)
                    messages.append({"role": "user", "content": feedback})
            except Exception as error:
                errors.append(dict(identity=sid, error=str(error)))
            result = dict(
                case=case_id,
                branch=index,
                stage=stage,
                target=target,
                answer=answer,
                correct=answer is not None
                and all(answer.get(k) == v for k, v in expected[target].items()),
                expected=expected[target],
                tool_calls=tool_calls,
                validation_retries=validation_retries,
                context_restorations=context_restorations,
                tool_context_samples=context_samples,
                wall_seconds=time.perf_counter() - began,
            )
            branches.append(result)
            return result

        async def branch(index, target):
            sid = root + f"/branch-{index}"
            store.open_session(sid, parent=root)
            try:
                result = None
                for stage in range(args.context_rounds):
                    # Revisit the first job after older evidence has left the
                    # resident set. The model must choose how to recover it.
                    current = (
                        target
                        if stage == args.context_rounds - 1
                        else (f"job-{(int(target[4:]) + stage * 7) % args.rows:04d}")
                    )
                    result = await audit_stage(index, current, sid, stage)
                return result
            finally:
                store.release_session(sid)

        branch_tasks = [
            asyncio.create_task(branch(i, target)) for i, target in enumerate(targets)
        ]
        try:
            decisions = await asyncio.gather(*branch_tasks)
            summary = [
                {"job": row["target"], "answer": row["answer"]} for row in decisions
            ]
            best = sorted(
                targets, key=lambda target: (-expected[target]["score"], target)
            )[0]
            reducer_messages = [
                {
                    "role": "system",
                    "content": 'Select the job with the highest score from these branch reports. Break ties by lexicographically smallest job. Return only {"job":"job-xxxx"}.',
                },
                {"role": "user", "content": json.dumps(summary)},
            ]
            try:
                answer = parse(await complete(reducer_messages, root + "/reduce"))
                reducers.append(
                    dict(
                        case=case_id,
                        correct=answer.get("job") == best,
                        answer=answer,
                        expected=best,
                    )
                )
            except Exception as error:
                errors.append(dict(identity=root + "/reduce", error=str(error)))
                reducers.append(dict(case=case_id, correct=False))
        finally:
            for task in branch_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*branch_tasks, return_exceptions=True)
            store.release_session(root)
            lifecycle.append(
                dict(
                    case=case_id,
                    original_bytes=original_bytes,
                    peak_stored_bytes=peak_stored_bytes,
                    final=store.stats(),
                    disk_bytes=(directory / f"case-{case_id}.sqlite").stat().st_size,
                )
            )
            store.close()

    async def run_case(case_id, directory):
        async with workflow_semaphore:
            await case(case_id, directory)
            print(
                args.mode,
                "completed cases",
                len(lifecycle),
                "/",
                args.cases,
                "correct branches",
                sum(row["correct"] for row in branches),
                "/",
                len(branches),
                flush=True,
            )

    samplers = []
    try:
        # Unrelated warmup is identical for every arm and excluded from metrics.
        await client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": "Reply with hello."}],
            temperature=0,
            max_tokens=16,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        # A fresh cache per arm; startup/warmup are excluded from measured work.
        async with aiohttp.ClientSession() as http:
            async with http.post(
                args.base_url.rstrip("/") + "/reset_prefix_cache?reset_external=true",
                json={},
            ) as response:
                response.raise_for_status()
                if not (await response.json()).get("success"):
                    raise RuntimeError("prefix cache reset rejected")
            async with http.get(args.base_url.rstrip("/") + "/metrics") as response:
                metrics = await response.text()
                cache_config = next(
                    line
                    for line in metrics.splitlines()
                    if line.startswith("vllm:cache_config_info{")
                )
                if f'kv_cache_memory_bytes="{args.kv_cache_bytes}"' not in cache_config:
                    raise ValueError(
                        "benchmark KV budget differs from server configuration"
                    )
                gpu_blocks = re.search(r'num_gpu_blocks="(\d+)"', cache_config)
                num_gpu_blocks = int(gpu_blocks.group(1)) if gpu_blocks else None
                preemptions_before = sum(
                    float(line.rsplit(" ", 1)[1])
                    for line in metrics.splitlines()
                    if line.startswith("vllm:num_preemptions_total{")
                )
        start = time.perf_counter()
        samplers = [
            asyncio.create_task(monitor()),
            asyncio.create_task(monitor_processes()),
        ]
        with tempfile.TemporaryDirectory(prefix="agentrix-tool-pages-") as temporary:
            tasks = [
                asyncio.create_task(run_case(case_id, Path(temporary)))
                for case_id in range(args.cases)
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Let each workflow release its snapshots before removing the
                # temporary directory, including when a sibling has failed.
                await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        await asyncio.gather(*samplers)
        await client.close()
    async with aiohttp.ClientSession() as http:
        async with http.get(args.base_url.rstrip("/") + "/metrics") as response:
            response.raise_for_status()
            metrics = await response.text()
    preemptions_after = sum(
        float(line.rsplit(" ", 1)[1])
        for line in metrics.splitlines()
        if line.startswith("vllm:num_preemptions_total{")
    )
    payload = dict(
        schema_version=1,
        workload="seeded build-report tool workflow; scripted ingestion, model-selected retrieval, fanout and reduction",
        mode=args.mode,
        seed=args.seed,
        hardware=hardware,
        model=args.model,
        model_info=model_info,
        rows=args.rows,
        cases=args.cases,
        branches_per_case=args.branches,
        concurrency=args.concurrency,
        workflow_concurrency=args.workflow_concurrency,
        context_rounds=args.context_rounds,
        tool_context_tokens=args.tool_context_tokens,
        search_excerpt_chars=args.search_excerpt_chars,
        tokenizer_sha256=(
            hashlib.sha256((args.tokenizer / "tokenizer.json").read_bytes()).hexdigest()
            if args.tokenizer
            else None
        ),
        kv_cache_bytes=args.kv_cache_bytes,
        restore_on_validation_error=args.restore_on_validation_error,
        validation_restore_sources=["checks"],
        max_restore_chars=args.max_restore_chars,
        cache_config_info=cache_config,
        corpus_sha256=corpus_hashes,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__).resolve(),
                ROOT / "application/src/agentrix_application/prompt_compactor.py",
            ]
        },
        branches=branches,
        reducers=reducers,
        requests=requests,
        transcripts=transcripts,
        memory_samples=memory,
        lifecycle=lifecycle,
        errors=errors,
    )
    payload["summary"] = dict(
        wall_seconds=elapsed,
        branch_correct=sum(row["correct"] for row in branches),
        branch_total=len(branches),
        reducer_correct=sum(row["correct"] for row in reducers),
        reducer_total=len(reducers),
        workflow_correct=sum(
            reducer["correct"]
            and all(
                branch["correct"]
                for branch in branches
                if branch["case"] == reducer["case"]
            )
            for reducer in reducers
        ),
        input_tokens=sum(row["usage"]["prompt_tokens"] for row in requests),
        output_tokens=sum(row["usage"]["completion_tokens"] for row in requests),
        request_count=len(requests),
        tool_calls=sum(row["tool_calls"] for row in branches),
        validation_retries=sum(row["validation_retries"] for row in branches),
        context_restorations=sum(row["context_restorations"] for row in branches),
        peak_tool_context_tokens=max(
            (
                sample["tokens"]
                for row in branches
                for sample in row["tool_context_samples"]
            ),
            default=None,
        ),
        peak_archived_observations=max(
            (
                sample["archived_observations"]
                for row in branches
                for sample in row["tool_context_samples"]
            ),
            default=0,
        ),
        mean_ttft=statistics.mean(row["ttft"] for row in requests)
        if requests
        else None,
        peak_gpu_mib=max(
            (gpu["memory_used_mib"] for row in memory for gpu in row.get("gpus", [])),
            default=None,
        ),
        peak_server_rss_kib=max(
            (row.get("server_rss_kib", 0) for row in memory), default=0
        ),
        peak_application_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        sampled_peak_live_kv_bytes=args.kv_cache_bytes
        * max((row.get("kv_usage", 0) for row in memory), default=0),
        sampled_peak_live_kv_blocks=(
            round(
                (num_gpu_blocks - 1)
                * max((row.get("kv_usage", 0) for row in memory), default=0)
            )
            if num_gpu_blocks is not None
            else None
        ),
        sampled_peak_running_requests=max(
            (row.get("num_requests_running", 0) for row in memory), default=0
        ),
        sampled_peak_waiting_requests=max(
            (row.get("num_requests_waiting", 0) for row in memory), default=0
        ),
        preemptions=preemptions_after - preemptions_before,
        all_snapshots_reclaimed=all(
            row["final"]["objects"] == 0 and row["final"]["sessions"] == 0
            for row in lifecycle
        ),
        errors=len(errors),
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:19000")
    parser.add_argument("--model", default="agentrix-paging")
    parser.add_argument("--mode", choices=["inline", "paged"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--branches", type=int, default=3)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument(
        "--workflow-concurrency",
        type=int,
        default=1,
        help="Concurrent workflows; --concurrency still caps in-flight model requests.",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--context-rounds",
        type=int,
        default=1,
        help="Sequential audits per branch; the final stage revisits its first job.",
    )
    parser.add_argument(
        "--tool-context-tokens",
        type=int,
        help="Cumulative tool observation budget; omit for an unbounded baseline.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Model directory containing tokenizer.json; enables archived observations.",
    )
    parser.add_argument(
        "--search-excerpt-chars",
        type=int,
        default=0,
        help="Return an excerpt instead of just the matched JSON row (0).",
    )
    parser.add_argument("--restore-on-validation-error", action="store_true")
    parser.add_argument("--max-restore-chars", type=int, default=131072)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--kv-cache-bytes", type=int, required=True)
    args = parser.parse_args()
    if (
        min(
            args.cases,
            args.branches,
            args.rows,
            args.concurrency,
            args.workflow_concurrency,
            args.max_steps,
            args.context_rounds,
            args.max_restore_chars,
            args.kv_cache_bytes,
        )
        < 1
        or args.branches >= args.rows
    ):
        parser.error("positive bounds and rows > branches are required")
    if (
        args.context_rounds > 1 or args.tool_context_tokens is not None
    ) and not args.tokenizer:
        parser.error(
            "--tokenizer is required for context budgets or multi-stage audits"
        )
    if args.tool_context_tokens is not None and args.tool_context_tokens < 1:
        parser.error("--tool-context-tokens must be positive")
    if not 0 <= args.search_excerpt_chars <= 16384:
        parser.error("--search-excerpt-chars must be in 0..16384")
    if args.tokenizer:
        if args.mode != "paged" or args.restore_on_validation_error:
            parser.error(
                "archived context requires --mode paged without full-report restoration"
            )
        global SYSTEM
        SYSTEM += """
Tool history is a bounded recent_results catalog. Old observations remain on disk.
Retrieve their metadata with {"action":"list_observations","offset":0} and exact
content with {"action":"read_observation","result_id":"...","offset":0,"limit":768}.
Partial entries contain only an exact prefix; next_offset is the next unread character.
You can also search either original report again. Each stage audits only its current job.
"""
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
