"""Trace-timed and legacy TraceLab replay with strict token accounting."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple, TextIO

import aiohttp
from tracelab_workload import PromptBuilder


class TimedRequest(NamedTuple):
    arrival_s: float
    session: dict[str, Any]
    row: dict[str, Any]
    round_index: int
    prompt: list[int]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


async def generate(
    client: aiohttp.ClientSession, endpoint: str, payload: dict[str, Any]
) -> tuple[dict[str, Any], list[int]]:
    started = time.perf_counter()
    first_token_at = None
    output_ids: list[int] = []
    usage = None
    done = False
    async with client.post(endpoint + "/v1/completions", json=payload) as response:
        if response.status != 200:
            raise RuntimeError(
                f"HTTP {response.status}: {(await response.text())[:500]}"
            )
        async for raw_line in response.content:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                done = True
                break
            chunk = json.loads(line[6:])
            if chunk.get("error"):
                raise RuntimeError(str(chunk["error"]))
            for choice in chunk.get("choices", []):
                ids = choice.get("token_ids") or []
                if ids and first_token_at is None:
                    first_token_at = time.perf_counter()
                output_ids.extend(ids)
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
    ended = time.perf_counter()
    if not done or first_token_at is None or usage is None:
        raise RuntimeError(
            "Incomplete stream: token IDs, final usage and DONE required"
        )
    if usage["prompt_tokens"] != len(payload["prompt"]):
        raise RuntimeError("Server changed the prompt token count")
    if len(output_ids) != payload["max_tokens"] or usage["completion_tokens"] != len(
        output_ids
    ):
        raise RuntimeError("Server did not generate the exact trace output length")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    result = {
        "ttft_ms": (first_token_at - started) * 1000,
        "e2e_ms": (ended - started) * 1000,
        "tpot_ms": (
            (ended - first_token_at) * 1000 / (len(output_ids) - 1)
            if len(output_ids) > 1
            else None
        ),
        "prompt_tokens": usage["prompt_tokens"],
        "output_tokens": len(output_ids),
        "cached_tokens": cached,
        "output_sha256": hashlib.sha256(json.dumps(output_ids).encode()).hexdigest(),
    }
    return result, output_ids


async def run_request(
    client: aiohttp.ClientSession,
    args: argparse.Namespace,
    session: dict[str, Any],
    row: dict[str, Any],
    index: int,
    prompt: list[int],
    run_start: float,
    log: TextIO,
    *,
    planned_s: float | None = None,
) -> tuple[dict[str, Any], list[int]]:
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": row["output_len"],
        "temperature": 0,
        "seed": args.seed,
        "ignore_eos": True,
        "return_token_ids": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "vllm_xargs": {
            "agentrix_session_id": session["session_id"],
            "agentrix_turn": index,
            "agentrix_history_tokens": (
                row["prefix_len"] if index and not row.get("context_reset") else 0
            ),
        },
    }
    submitted_s = time.perf_counter() - run_start
    identity = {
        "session_id": session["session_id"],
        "provider": session["provider"],
        "round_index": index,
        "source_row": row["source_row"],
        "submitted_s": submitted_s,
    }
    try:
        result, output = await generate(client, args.base_url, payload)
    except Exception as error:
        log.write(json.dumps(dict(identity, error=repr(error))) + "\n")
        log.flush()
        raise
    result.update(
        identity,
        completed_s=time.perf_counter() - run_start,
        planned_prefix_tokens=row["prefix_len"],
        tool_wait_ms=row["tool_wait_after_ms"] if planned_s is None else 0,
    )
    if planned_s is not None:
        lag_ms = max(0.0, submitted_s - planned_s) * 1000
        result.update(
            planned_arrival_s=planned_s,
            arrival_lag_ms=lag_ms,
            arrival_ttft_ms=lag_ms + result["ttft_ms"],
            arrival_e2e_ms=lag_ms + result["e2e_ms"],
        )
    log.write(json.dumps(result) + "\n")
    log.flush()
    return result, output


async def replay_session(
    client: aiohttp.ClientSession,
    args: argparse.Namespace,
    session: dict[str, Any],
    run_start: float,
    log: TextIO,
) -> list[dict[str, Any]]:
    delay = session["arrival_time_ms"] / 1000 - (time.perf_counter() - run_start)
    await asyncio.sleep(max(0.0, delay))
    builder = PromptBuilder(session["session_id"], args.seed)
    results = []
    for index, row in enumerate(session["rounds"]):
        prompt = builder.build(row)
        result, output = await run_request(
            client, args, session, row, index, prompt, run_start, log
        )
        results.append(result)
        builder.commit(prompt, output)
        if row["tool_wait_after_ms"]:
            await asyncio.sleep(row["tool_wait_after_ms"] / 1000)
    print(f"Completed {session['provider']} session: {len(results)} rounds", flush=True)
    return results


def prepare_timeline(workload: dict[str, Any]) -> list[TimedRequest]:
    """Build response-independent prompts before starting the replay clock.

    Public traces omit prompt/output contents. Reuse known synthetic prompt
    history and fill missing history; never wait for this engine's output or
    invent shared prefixes between distinct sessions or load copies.
    """
    prepared = []
    for session in workload["sessions"]:
        builder = None
        for index, row in enumerate(session["rounds"]):
            if builder is None or row["context_reset"]:
                builder = PromptBuilder(
                    f"{session['session_id']}:reset-{row['source_row']}",
                    workload["metadata"]["seed"],
                )
            prompt = builder.build(row)
            builder.commit(prompt, [])
            prepared.append(
                TimedRequest(row["arrival_time_ms"] / 1000, session, row, index, prompt)
            )
    return sorted(
        prepared,
        key=lambda item: (item.arrival_s, item.session["session_id"], item.round_index),
    )


async def replay_timeline(
    client: aiohttp.ClientSession,
    args: argparse.Namespace,
    prepared: list[TimedRequest],
    run_start: float,
    log: TextIO,
) -> list[dict[str, Any]]:
    active = 0

    async def submit(item: TimedRequest) -> dict[str, Any]:
        nonlocal active
        planned_s, session, row, index, prompt = item
        await asyncio.sleep(max(0.0, run_start + planned_s - time.perf_counter()))
        if active >= args.max_inflight:
            raise RuntimeError(
                "Client in-flight limit exceeded; refusing to shift arrivals"
            )
        active += 1
        try:
            result, _ = await run_request(
                client,
                args,
                session,
                row,
                index,
                prompt,
                run_start,
                log,
                planned_s=planned_s,
            )
            return result
        finally:
            active -= 1

    tasks = [asyncio.create_task(submit(item)) for item in prepared]
    try:
        return await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def sample_metrics(
    client: aiohttp.ClientSession, endpoint: str, output: Path, run_start: float
) -> None:
    """Observe running/queued requests; these gauges are not exact kernel batches."""
    names = (
        "vllm:num_requests_running{",
        "vllm:num_requests_waiting{",
        "vllm:kv_cache_usage_perc{",
    )
    with output.open("x") as log:
        while True:
            sample: dict[str, Any] = {"elapsed_s": time.perf_counter() - run_start}
            try:
                async with client.get(
                    endpoint + "/metrics", timeout=aiohttp.ClientTimeout(total=2)
                ) as response:
                    response.raise_for_status()
                    sample["gauges"] = {
                        line.rsplit(" ", 1)[0]: float(line.rsplit(" ", 1)[1])
                        for line in (await response.text()).splitlines()
                        if line.startswith(names)
                    }
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                sample["error"] = str(error)
            log.write(json.dumps(sample) + "\n")
            log.flush()
            await asyncio.sleep(1)


def prompt_source_delta(before: str, after: str) -> dict[str, float]:
    """Separate GPU reuse from external KV transfer using engine counters."""
    totals: dict[str, float] = defaultdict(float)
    for sign, text in ((-1, before), (1, after)):
        for line in text.splitlines():
            if not line.startswith("vllm:prompt_tokens_by_source_total{"):
                continue
            source = line.split('source="', 1)[1].split('"', 1)[0]
            totals[source] += sign * float(line.rsplit(" ", 1)[1])
    return dict(totals)


def summarize(rows: list[dict[str, Any]], makespan_s: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "requests": len(rows),
        "makespan_s": makespan_s,
        "requests_per_second": len(rows) / makespan_s,
        "output_tokens_per_second": sum(row["output_tokens"] for row in rows)
        / makespan_s,
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "cached_tokens": sum(row["cached_tokens"] for row in rows),
    }
    for label, subset in (
        ("all", rows),
        ("followup", [row for row in rows if row["round_index"] > 0]),
    ):
        if not subset:
            continue
        result[f"{label}_cached_token_rate"] = sum(
            row["cached_tokens"] for row in subset
        ) / sum(row["prompt_tokens"] for row in subset)
        for metric in ("ttft_ms", "e2e_ms"):
            values = [row[metric] for row in subset]
            result[f"{label}_{metric}_mean"] = statistics.fmean(values)
            for quantile in (50, 95, 99):
                result[f"{label}_{metric}_p{quantile}"] = percentile(
                    values, quantile / 100
                )
    tpot = [row["tpot_ms"] for row in rows if row.get("tpot_ms") is not None]
    if tpot:
        result["tpot_ms_p50"] = percentile(tpot, 0.5)
        result["tpot_ms_p95"] = percentile(tpot, 0.95)
    for metric in ("arrival_lag_ms", "arrival_ttft_ms", "arrival_e2e_ms"):
        values = [row[metric] for row in rows if metric in row]
        if values:
            result[f"{metric}_max"] = max(values)
            for quantile in (50, 95, 99):
                result[f"{metric}_p{quantile}"] = percentile(values, quantile / 100)
    events = sorted(
        event
        for row in rows
        if "submitted_s" in row
        for event in (
            (row["submitted_s"], 1, row["session_id"]),
            (row["completed_s"], -1, row["session_id"]),
        )
    )
    if events:
        active_sessions: dict[str, int] = defaultdict(int)
        active = peak = peak_sessions = 0
        for _, change, session_id in events:
            active += change
            active_sessions[session_id] += change
            if active_sessions[session_id] == 0:
                del active_sessions[session_id]
            peak = max(peak, active)
            peak_sessions = max(peak_sessions, len(active_sessions))
        result["peak_inflight_requests"] = peak
        result["peak_inflight_sessions"] = peak_sessions
    return result


async def main_async(args: argparse.Namespace) -> None:
    workload = json.loads(args.workload.read_text())
    args.seed = workload["metadata"]["seed"]
    trace_timed = workload["metadata"].get("timing_mode") == "trace_open_loop"
    prepared = prepare_timeline(workload) if trace_timed else None
    args.output.mkdir(parents=True, exist_ok=False)
    timeout = aiohttp.ClientTimeout(total=1800, sock_read=1800)
    # Tool waits can exceed server keep-alive timeouts. Use a fresh loopback
    # connection per request instead of retrying ambiguous POST failures.
    connector = aiohttp.TCPConnector(force_close=True, limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as client:
        # Warm kernels using disjoint input, then cold-start every measured replay.
        await generate(
            client,
            args.base_url,
            {
                "model": args.model,
                "prompt": [31000 + index % 1000 for index in range(1024)],
                "max_tokens": 16,
                "ignore_eos": True,
                "temperature": 0,
                "stream": True,
                "return_token_ids": True,
                "stream_options": {"include_usage": True},
            },
        )
        async with client.post(args.base_url + "/reset_prefix_cache") as response:
            response.raise_for_status()
        await asyncio.sleep(1)
        async with client.get(args.base_url + "/metrics") as response:
            response.raise_for_status()
            metrics_before = await response.text()
            (args.output / "metrics_before.prom").write_text(metrics_before)
        started = time.perf_counter()
        sampler = asyncio.create_task(
            sample_metrics(
                client, args.base_url, args.output / "metrics_samples.jsonl", started
            )
        )
        with (args.output / "requests.jsonl").open("x") as log:
            tasks = []
            try:
                if prepared is not None:
                    rows = await replay_timeline(client, args, prepared, started, log)
                else:
                    tasks = [
                        asyncio.create_task(
                            replay_session(client, args, session, started, log)
                        )
                        for session in workload["sessions"]
                    ]
                    rows = [
                        row for batch in await asyncio.gather(*tasks) for row in batch
                    ]
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                sampler.cancel()
                await asyncio.gather(sampler, return_exceptions=True)
        elapsed = time.perf_counter() - started
        async with client.get(args.base_url + "/metrics") as response:
            response.raise_for_status()
            metrics_after = await response.text()
            (args.output / "metrics_after.prom").write_text(metrics_after)
    if len(rows) != workload["metadata"]["selected_rounds"]:
        raise RuntimeError("Replay did not complete the entire selected workload")
    result = {
        "label": args.label,
        "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        "workload": workload["metadata"],
        "summary": summarize(rows, elapsed),
        "server_prompt_sources": prompt_source_delta(metrics_before, metrics_after),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8145")
    parser.add_argument("--model", default="tracelab-qwen")
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-inflight", type=int, default=512)
    args = parser.parse_args()
    if args.max_inflight < 1:
        parser.error("max-inflight must be positive")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
