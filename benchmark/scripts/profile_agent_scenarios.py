#!/usr/bin/env python3
"""Controlled GPU-only agent serving exploration; immutable token workloads."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark/src"))
from profile_tracelab import require_free_port, service, wait_for_service
from tracelab_replay import generate, percentile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare(args):
    from data import record_to_prompt
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    cases = []
    sources = {}

    def source(path):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        return [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]

    coding = {}
    for name in ("django", "sqlite", "ffmpeg"):
        row = source(ROOT / f"benchmark/data/{name}_agentrix/cases_30k_b16.jsonl")[0]
        text = "\n\n".join(m["content"] for m in row["shared_messages"])
        coding[name] = (row, encode(text))
        prefix = encode(text)[:28000]
        suffixes = [encode("\n\n" + b["private_instruction"]) for b in row["branches"]]
        cases.append(
            {
                "name": f"{name}_native_b16",
                "kind": "coding_fanout",
                "dataset": name,
                "source_case": row["case_id"],
                "prefix": prefix,
                "suffixes": suffixes,
                "rounds": 1,
                "output_tokens": 256,
                "truncated": len(encode(text)) > 28000,
            }
        )
    for length, branches in (
        (2048, 16),
        (8192, 16),
        (24576, 16),
        (24576, 32),
        (24576, 64),
        (8192, 32),
        (16384, 8),
        (16384, 16),
        (24576, 4),
        (24576, 8),
    ):
        cases.append(
            {
                "name": f"shape_p{length}_b{branches}",
                "kind": "controlled_fanout",
                "dataset": "django",
                "prefix": coding["django"][1][:length],
                "suffixes": [
                    encode(
                        f"\n\nIndependent investigation branch {i}: analyze correctness, call paths and tests."
                    )
                    for i in range(branches)
                ],
                "rounds": 1,
                "output_tokens": 256,
                "truncated": True,
            }
        )
    for name in ("swebench", "agencybench", "agentboard", "appworld"):
        filename = {
            "swebench": "swebench_verified",
            "agencybench": "agencybench_v2",
        }.get(name, name)
        rows = source(ROOT / f"benchmark/data/{filename}.jsonl")
        # First record: no performance-driven sample selection or length padding.
        prefix = encode(record_to_prompt(name, rows[0]))[:28000]
        cases.append(
            {
                "name": f"{name}_native_b16",
                "kind": "dataset_fanout",
                "dataset": name,
                "prefix": prefix,
                "suffixes": [
                    encode(
                        f"\n\nInvestigation {i}: independently propose and verify the next actions."
                    )
                    for i in range(16)
                ],
                "rounds": 1,
                "output_tokens": 256,
                "truncated": False,
            }
        )
    for name in ("django", "sqlite", "ffmpeg"):
        rows = source(ROOT / f"benchmark/data/{name}_agentrix/cases_30k_b16.jsonl")
        sessions = [
            encode("\n\n".join(m["content"] for m in r["shared_messages"]))[:24000]
            for r in rows
        ]
        for length in (12000, 24000):
            cases.append(
                {
                    "name": f"{name}_forest4_p{length}_b8",
                    "kind": "forest",
                    "dataset": name,
                    "roots": [p[:length] for p in sessions],
                    "rounds": 1,
                    "output_tokens": 256,
                    "suffixes": [
                        [
                            encode("\n\n" + b["private_instruction"])
                            for b in r["branches"][:8]
                        ]
                        for r in rows
                    ],
                }
            )
        cases.append(
            {
                "name": f"{name}_sessions4_turns4",
                "kind": "multi_turn",
                "dataset": name,
                "sessions": sessions,
                "rounds": 4,
                "output_tokens": 128,
                "followup": encode(
                    "\nTool observation: the repository snapshot is unchanged. Review the preceding analysis and identify another relevant code path and regression test.\n"
                ),
            }
        )
    # Causal multi-turn sessions: later prompts include this engine's generated IDs.
    for name in ("swebench", "agencybench"):
        filename = {"swebench": "swebench_verified", "agencybench": "agencybench_v2"}[
            name
        ]
        rows = source(ROOT / f"benchmark/data/{filename}.jsonl")[:8]
        sessions = [encode(record_to_prompt(name, r))[:10000] for r in rows]
        cases.append(
            {
                "name": f"{name}_sessions8_turns4",
                "kind": "multi_turn",
                "dataset": name,
                "sessions": sessions,
                "rounds": 4,
                "output_tokens": 128,
                "followup": encode(
                    "\nTool observation: the proposed check has completed. Reconsider assumptions and propose the next verification step.\n"
                ),
            }
        )
    value = {
        "model": args.model,
        "sources": sources,
        "cases": cases,
        "protocol": "Frozen inputs; bootstrap then concurrent branches. Multi-turn appends actual generated IDs; fixed output length, no tool execution or task-quality scoring.",
    }
    args.workload.parent.mkdir(parents=True, exist_ok=True)
    with args.workload.open("x") as f:
        json.dump(value, f)
    print(
        json.dumps(
            [
                {
                    "name": c["name"],
                    "prefix": len(c.get("prefix", [])),
                    "sessions": len(c.get("sessions", [])),
                    "branches": len(c.get("suffixes", [])),
                }
                for c in cases
            ],
            indent=2,
        )
    )


async def metrics(client, url):
    async with client.get(url + "/metrics") as response:
        response.raise_for_status()
        return await response.text()


async def sample_metrics(client, url, stop, samples):
    started = time.perf_counter()
    while not stop.is_set():
        body = await metrics(client, url)
        values = {}
        for line in body.splitlines():
            if line.startswith(
                (
                    "vllm:num_requests_running{",
                    "vllm:num_requests_waiting{",
                    "vllm:kv_cache_usage_perc{",
                )
            ):
                key, value = line.rsplit(" ", 1)
                values[key] = float(value)
        samples.append({"time_s": time.perf_counter() - started, "values": values})
        try:
            await asyncio.wait_for(stop.wait(), timeout=1)
        except TimeoutError:
            pass


def counters(body):
    result = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.rsplit(" ", 1)
        if any(
            s in key
            for s in (
                "preemptions_total",
                "prompt_tokens_by_source_total",
                "prefix_cache_hits_total",
                "prefix_cache_queries_total",
            )
        ):
            result[key] = float(value)
    return result


async def replay(args, workload):
    import aiohttp

    url = f"http://127.0.0.1:{args.port}"
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=900),
        connector=aiohttp.TCPConnector(limit=0),
    ) as client:

        async def request(prompt, output, session="warmup", turn=0):
            payload = {
                "model": "agent-scenarios",
                "prompt": prompt,
                "max_tokens": output,
                "temperature": 0,
                "seed": 20260905,
                "ignore_eos": True,
                "return_token_ids": True,
                "stream": True,
                "stream_options": {"include_usage": True},
                "vllm_xargs": {
                    "agentrix_session_id": session,
                    "agentrix_turn": turn,
                    "agentrix_history_tokens": len(prompt) if turn else 0,
                },
            }
            return await generate(client, url, payload)

        await request([1000 + i % 100 for i in range(256)], 16)
        for case in workload["cases"]:
            if args.cases and case["name"] not in args.cases.split(","):
                continue
            for repeat in range(args.repeats):
                case_dir = args.output / f"{case['name']}_r{repeat}"
                case_dir.mkdir()
                async with client.post(url + "/reset_prefix_cache") as response:
                    response.raise_for_status()
                await asyncio.sleep(1.1)
                before = await metrics(client, url)
                (case_dir / "metrics_before.txt").write_text(before)
                records = []
                samples = []
                stop = asyncio.Event()
                sampler = asyncio.create_task(
                    sample_metrics(client, url, stop, samples)
                )
                start = time.perf_counter()

                async def measured(
                    prompt, output, sid, turn, stage, *, start=start, records=records
                ):
                    submitted = time.perf_counter() - start
                    row, ids = await request(prompt, output, sid, turn)
                    row.update(
                        session=sid,
                        turn=turn,
                        stage=stage,
                        submitted_s=submitted,
                        completed_s=time.perf_counter() - start,
                        prompt_sha256=digest(prompt),
                    )
                    records.append(row)
                    return ids

                if case["kind"] == "multi_turn":

                    async def session(i, prompt, *, case=case, repeat=repeat):
                        for turn in range(case["rounds"]):
                            ids = await measured(
                                prompt,
                                case["output_tokens"],
                                f"{case['name']}-{repeat}-{i}",
                                turn,
                                "turn",
                            )
                            prompt = prompt + ids + case["followup"]

                    await asyncio.gather(
                        *(session(i, p.copy()) for i, p in enumerate(case["sessions"]))
                    )
                    branch_start = start
                elif case["kind"] == "forest":
                    await asyncio.gather(
                        *(
                            measured(p, 32, f"root-{repeat}-{i}", 0, "bootstrap")
                            for i, p in enumerate(case["roots"])
                        )
                    )
                    jobs = [
                        (i, j, p + s)
                        for i, (p, suffixes) in enumerate(
                            zip(case["roots"], case["suffixes"])
                        )
                        for j, s in enumerate(suffixes)
                    ]
                    random.Random(20260905).shuffle(jobs)
                    branch_start = time.perf_counter()
                    await asyncio.gather(
                        *(
                            measured(
                                p,
                                case["output_tokens"],
                                f"root-{repeat}-{i}",
                                1,
                                "branch",
                            )
                            for i, j, p in jobs
                        )
                    )
                else:
                    await measured(case["prefix"], 32, f"root-{repeat}", 0, "bootstrap")
                    # Frozen branch inputs isolate backend arithmetic from output drift.
                    if args.profile:
                        async with client.post(url + "/start_profile") as response:
                            response.raise_for_status()
                    branch_start = time.perf_counter()
                    await asyncio.gather(
                        *(
                            measured(
                                case["prefix"] + suffix,
                                case["output_tokens"],
                                f"branch-{repeat}-{i}",
                                0,
                                "branch",
                            )
                            for i, suffix in enumerate(case["suffixes"])
                        )
                    )
                end = time.perf_counter()
                stop.set()
                await sampler
                (case_dir / "metrics_samples.json").write_text(json.dumps(samples))
                await asyncio.sleep(1.1)
                after = await metrics(client, url)
                (case_dir / "metrics_after.txt").write_text(after)
                a, b = counters(before), counters(after)
                branch = [r for r in records if r["stage"] != "bootstrap"]
                summary = {
                    "case": case["name"],
                    "label": args.label,
                    "repeat": repeat,
                    "wall_s": end - start,
                    "branch_wall_s": end - branch_start,
                    "output_tps": sum(r["output_tokens"] for r in records)
                    / (end - start),
                    "branch_output_tps": sum(r["output_tokens"] for r in branch)
                    / (end - branch_start),
                    "ttft_p50_ms": statistics.median(r["ttft_ms"] for r in branch),
                    "ttft_p95_ms": percentile([r["ttft_ms"] for r in branch], 0.95),
                    "tpot_p50_ms": statistics.median(r["tpot_ms"] for r in branch),
                    "cached_tokens": sum(r["cached_tokens"] for r in records),
                    "prompt_tokens": sum(r["prompt_tokens"] for r in records),
                    "requests": len(records),
                    "counter_deltas": {k: b[k] - a.get(k, 0) for k in b},
                }
                (case_dir / "requests.json").write_text(json.dumps(records, indent=2))
                (case_dir / "summary.json").write_text(json.dumps(summary, indent=2))
                print(json.dumps(summary), flush=True)


def run(args):
    workload = json.loads(args.workload.read_text())
    if Path(workload["model"]).resolve() != Path(args.model).resolve():
        raise ValueError("Prepare a workload with the tokenizer of the selected model")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "harness.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "gpu_identity.csv").write_text(
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv",
            ],
            text=True,
        )
    )
    require_free_port(args.port)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("VLLM_", "LMCACHE_")) and k != "PYTHONPATH"
    }
    env.update(
        PYTHONPATH=str(args.runtime_root.resolve()),
        CUDA_VISIBLE_DEVICES=args.gpus,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONHASHSEED="0",
        VLLM_USE_FLASHINFER_SAMPLER="0",
        VLLM_SERVER_DEV_MODE="1",
        VLLM_PLUGINS="",
        VLLM_AGENTRIX_DP_ROUTING_POLICY=args.policy,
        VLLM_FORK_ATTN_DP_PREFIX_MIN_BLOCKS=str(args.prefix_min_blocks),
        VLLM_FORK_ATTN_DP_WORK_SLACK_TOKENS=str(args.work_slack),
        VLLM_FORK_ATTN_ENABLE_FOREST="1",
        VLLM_FORK_ATTN_ENABLE_FOREST_CUDAGRAPH="1",
        VLLM_FORK_ATTN_FANOUT_SCHEDULING_ENABLED="0",
    )
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        "agent-scenarios",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        args.backend,
        "--generation-config",
        "vllm",
        "--data-parallel-size",
        str(len(args.gpus.split(","))),
        "--api-server-count",
        "1",
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--no-async-scheduling",
        "--block-size",
        "16",
        "--gpu-memory-utilization",
        "0.90",
        "--max-model-len",
        "32768",
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        "64",
        "--num-gpu-blocks-override",
        str(args.gpu_blocks),
    ]
    model_config = json.loads((Path(args.model) / "config.json").read_text())
    if "vision_config" in model_config:
        command += ["--limit-mm-per-prompt", '{"image":0,"video":0}']
    if args.eager:
        command += ["--enforce-eager"]
    if args.profile:
        command += [
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": str((args.output / "profile").resolve()),
                    "torch_profiler_with_stack": False,
                    "ignore_frontend": True,
                    "max_iterations": 12,
                }
            ),
        ]
    identity = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import vllm,torch; print(vllm.__file__,vllm.__version__,torch.__version__)",
        ],
        env=env,
        text=True,
    )
    (args.output / "configuration.json").write_text(
        json.dumps(
            {
                "command": command,
                "environment": {
                    k: v
                    for k, v in env.items()
                    if k.startswith(("VLLM_", "CUDA_VISIBLE", "PYTHONPATH"))
                },
                "runtime_identity": identity,
                "workload_sha256": digest(workload),
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "arguments": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "source_sha256": {
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (
                        ROOT / "benchmark/src/tracelab_replay.py",
                        ROOT / "benchmark/scripts/profile_tracelab.py",
                        ROOT / "vllm/vllm/v1/engine/prefix_router.py",
                        ROOT / "vllm/vllm/v1/attention/backends/fork_attn.py",
                    )
                },
            },
            indent=2,
        )
    )
    with service(command, env, args.output / "server.log") as server:
        wait_for_service(server, args.port, health=True)
        with service(
            ["nvidia-smi", "dmon", "-i", args.gpus, "-s", "pucvmet", "-d", "1"],
            env,
            args.output / "gpu_samples.log",
        ):
            asyncio.run(replay(args, workload))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument(
        "--model", default="/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--label", default="explore")
    parser.add_argument(
        "--backend", default="FLASH_ATTN", choices=("FLASH_ATTN", "FORK_ATTN")
    )
    parser.add_argument(
        "--policy",
        default="native",
        choices=("native", "prefix_aware", "session_aware"),
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--port", type=int, default=8150)
    parser.add_argument("--gpu-blocks", type=int, default=4608)
    parser.add_argument("--prefix-min-blocks", type=int, default=4)
    parser.add_argument("--work-slack", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cases", default="")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    prepare(args) if args.prepare else run(args)
