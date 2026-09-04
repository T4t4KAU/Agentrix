#!/usr/bin/env python3
"""Black-box characterization of multimodal encoder locality and interference."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer


@dataclass
class Result:
    experiment: str
    case: str
    workload: str
    repetition: int
    rank: int
    cache_state: str
    ttft_ms: float
    total_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    image_path: str


def stream_chat(port: int, content: Any, timeout: float = 300.0) -> tuple[float, float, dict]:
    payload = {
        "model": "qwen35",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token = None
    usage: dict[str, int] = {}
    with httpx.Client(timeout=timeout) as client:
        with client.stream(
            "POST", f"http://127.0.0.1:{port}/v1/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if (delta.get("content") or delta.get("reasoning_content")) and first_token is None:
                        first_token = time.perf_counter()
    ended = time.perf_counter()
    if first_token is None:
        first_token = ended
    return (first_token - started) * 1000, (ended - started) * 1000, usage


def image_content(path: Path) -> list[dict[str, Any]]:
    return [
        {"type": "image_url", "image_url": {"url": path.resolve().as_uri()}},
        {"type": "text", "text": "Describe the image in one word."},
    ]


def make_victim_prompt(tokenizer_path: Path, target_tokens: int) -> str:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    seed = "multimodal scheduling benchmark context " * (target_tokens // 3 + 100)
    ids = tokenizer.encode(seed, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def record(
    rows: list[Result], experiment: str, case: str, workload: str,
    repetition: int, rank: int, cache_state: str, measurement: tuple[float, float, dict],
    image_path: Path | None = None,
) -> None:
    ttft, total, usage = measurement
    rows.append(Result(
        experiment=experiment,
        case=case,
        workload=workload,
        repetition=repetition,
        rank=rank,
        cache_state=cache_state,
        ttft_ms=ttft,
        total_ms=total,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        image_path=str(image_path or ""),
    ))


def concurrent_pair(
    victim_content: str,
    attacker_path: Path,
    attacker_rank: int,
    delay_ms: float,
) -> tuple[tuple[float, float, dict], tuple[float, float, dict]]:
    started = threading.Event()

    def attack() -> tuple[float, float, dict]:
        started.set()
        return stream_chat(8000 + attacker_rank, image_content(attacker_path))

    def victim() -> tuple[float, float, dict]:
        started.wait()
        time.sleep(delay_ms / 1000)
        return stream_chat(8000, victim_content)

    with ThreadPoolExecutor(max_workers=2) as pool:
        attack_future = pool.submit(attack)
        victim_future = pool.submit(victim)
        return victim_future.result(), attack_future.result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--victim-tokens", type=int, default=4096)
    parser.add_argument("--attacker-head-start-ms", type=float, default=30.0)
    parser.add_argument(
        "--workloads", nargs="+", default=["small", "medium", "large", "xlarge"]
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    victim_content = make_victim_prompt(args.model, args.victim_tokens)
    rows: list[Result] = []

    for rank in (0, 1):
        stream_chat(8000 + rank, "Warm up. Reply with one word.")

    for repetition in range(args.repetitions):
        measurement = stream_chat(8000, victim_content)
        record(rows, "interference", "baseline", "text", repetition, 0, "n/a", measurement)

    # Experiment 2 first so each workload is guaranteed cold on both ranks.
    for workload in args.workloads:
        path = sorted(args.assets.glob(f"{workload}_*_v0.jpg"))[0]
        cold0 = stream_chat(8000, image_content(path))
        hot0 = stream_chat(8000, image_content(path))
        cold1 = stream_chat(8001, image_content(path))
        record(rows, "locality", "rank0", workload, 0, 0, "cold", cold0, path)
        record(rows, "locality", "rank0", workload, 0, 0, "hot", hot0, path)
        record(rows, "locality", "rank1", workload, 0, 1, "cross_rank_cold", cold1, path)

    # Use unique image bytes for every pair so the attacker always runs Vision Encode.
    for workload in args.workloads:
        for repetition in range(args.repetitions):
            same_path = sorted(args.assets.glob(f"{workload}_*_v{1 + repetition}.jpg"))[0]
            diff_path = sorted(args.assets.glob(f"{workload}_*_v{1 + args.repetitions + repetition}.jpg"))[0]
            victim, attacker = concurrent_pair(
                victim_content, same_path, attacker_rank=0,
                delay_ms=args.attacker_head_start_ms,
            )
            record(rows, "interference", "same_rank_victim", workload, repetition, 0, "cold", victim)
            record(rows, "interference", "same_rank_attacker", workload, repetition, 0, "cold", attacker, same_path)
            victim, attacker = concurrent_pair(
                victim_content, diff_path, attacker_rank=1,
                delay_ms=args.attacker_head_start_ms,
            )
            record(rows, "interference", "different_rank_victim", workload, repetition, 0, "cold", victim)
            record(rows, "interference", "different_rank_attacker", workload, repetition, 1, "cold", attacker, diff_path)

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    baseline = statistics.median(
        row.ttft_ms for row in rows if row.case == "baseline"
    )
    print(f"baseline median TTFT: {baseline:.2f} ms")
    for workload in args.workloads:
        for case in ("same_rank_victim", "different_rank_victim"):
            values = [
                row.ttft_ms for row in rows
                if row.workload == workload and row.case == case
            ]
            print(
                f"{workload:8s} {case:23s} median={statistics.median(values):8.2f} ms "
                f"slowdown={statistics.median(values) / baseline:6.2f}x"
            )
    print(args.output)


if __name__ == "__main__":
    main()
