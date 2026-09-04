#!/usr/bin/env python3
"""Measure vLLM's multimodal preprocessing and encoder stages offline."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from PIL import Image

from vllm import LLM, SamplingParams
from vllm.benchmarks.mm_processor import get_timing_stats_from_engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--mm-processor-cache-gb", type=float, default=4.0)
    parser.add_argument(
        "--workloads", nargs="+", default=["small", "medium", "large", "xlarge"]
    )
    args = parser.parse_args()

    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=16384,
        gpu_memory_utilization=0.88,
        enable_prefix_caching=False,
        mm_processor_cache_gb=args.mm_processor_cache_gb,
        enable_mm_processor_stats=True,
    )
    sampling = SamplingParams(temperature=0, max_tokens=1)

    def run(path: Path) -> tuple[float, int, dict[str, float | int]]:
        with Image.open(path) as source:
            image = source.convert("RGB")
        request = {
            "prompt": "<|vision_start|><|image_pad|><|vision_end|>Describe in one word.",
            "multi_modal_data": {"image": image},
        }
        start = time.perf_counter()
        outputs = llm.generate(request, sampling, use_tqdm=False)
        total_ms = (time.perf_counter() - start) * 1000
        stats = get_timing_stats_from_engine(llm.llm_engine)
        merged: dict[str, float | int] = {}
        for item in stats.values():
            merged.update(item)
        prompt_tokens = len(outputs[0].prompt_token_ids)
        return total_ms, prompt_tokens, merged

    # Clear one-time kernels and timing registries before recording.
    warm_path = sorted(args.assets.glob("small_*_v7.jpg"))[0]
    run(warm_path)

    rows: list[dict[str, object]] = []
    for workload in args.workloads:
        for repetition in range(args.repetitions):
            path = next(args.assets.glob(f"{workload}_*_v{8 + repetition}.jpg"))
            cold_total, prompt_tokens, cold_stats = run(path)
            hot_total, hot_prompt_tokens, hot_stats = run(path)
            for cache_state, total_ms, tokens, stats in (
                ("cold", cold_total, prompt_tokens, cold_stats),
                ("hot", hot_total, hot_prompt_tokens, hot_stats),
            ):
                rows.append({
                    "workload": workload,
                    "repetition": repetition,
                    "cache_state": cache_state,
                    "prompt_tokens": tokens,
                    "total_ms": total_ms,
                    "encoder_forward_ms": float(stats.get("encoder_forward_secs", 0)) * 1000,
                    "num_encoder_calls": int(stats.get("num_encoder_calls", 0)),
                    "preprocessor_total_ms": float(stats.get("preprocessor_total_secs", 0)) * 1000,
                    "apply_hf_processor_ms": float(stats.get("apply_hf_processor_secs", 0)) * 1000,
                    "image_path": str(path),
                })
                print(rows[-1])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
