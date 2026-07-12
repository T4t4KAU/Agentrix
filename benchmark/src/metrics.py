from __future__ import annotations

import math
import statistics
from typing import Iterable

from models import BenchmarkTrace


def _validate(prefix_tokens: int, suffix_tokens: Iterable[int], tile_k: int) -> list[int]:
    suffixes = list(suffix_tokens)
    if prefix_tokens < 0 or any(value < 0 for value in suffixes):
        raise ValueError("token counts must be non-negative")
    if not suffixes:
        raise ValueError("suffix_tokens must not be empty")
    if tile_k <= 0:
        raise ValueError("tile_k must be positive")
    return suffixes


def baseline_metrics(
    prefix_tokens: int, suffix_tokens: Iterable[int], tile_k: int = 128
) -> dict[str, int]:
    suffixes = _validate(prefix_tokens, suffix_tokens, tile_k)
    lengths = [prefix_tokens + suffix for suffix in suffixes]
    return {
        "valid_qk": sum(lengths),
        "unique_kv_tokens": sum(lengths),
        "scheduled_tiles": sum(math.ceil(length / tile_k) for length in lengths),
        "logical_launches": len(lengths),
    }

def monowire_metrics(
    prefix_tokens: int, suffix_tokens: Iterable[int], tile_k: int = 128
) -> dict[str, int]:
    suffixes = _validate(prefix_tokens, suffix_tokens, tile_k)
    return {
        "valid_qk": sum(prefix_tokens + suffix for suffix in suffixes),
        "unique_kv_tokens": prefix_tokens + sum(suffixes),
        "scheduled_tiles": math.ceil(prefix_tokens / tile_k)
        + math.ceil(sum(suffixes) / tile_k),
        "logical_launches": 1,
    }


def compare_trace(
    trace: BenchmarkTrace,
    tile_k: int = 128,
    kv_bytes_per_token: int = 0,
) -> dict[str, int | float | str]:
    suffixes = [branch.suffix_tokens for branch in trace.branches]
    baseline = baseline_metrics(trace.prefix_tokens, suffixes, tile_k)
    monowire = monowire_metrics(trace.prefix_tokens, suffixes, tile_k)
    if baseline["valid_qk"] != monowire["valid_qk"]:
        raise AssertionError("Baseline and Monowire valid Q-K work must match")

    def ratio(left: int, right: int) -> float:
        return left / right if right else 1.0

    baseline_kv = baseline["unique_kv_tokens"]
    optimized_kv = monowire["unique_kv_tokens"]
    kv_tokens_saved = baseline_kv - optimized_kv
    result: dict[str, int | float | str] = {
        "case_id": trace.case_id,
        "prefix_tokens": trace.prefix_tokens,
        "branch_count": len(suffixes),
        "suffix_distribution": trace.suffix_distribution,
        "suffix_mean": statistics.fmean(suffixes),
        "suffix_std": statistics.pstdev(suffixes),
        "baseline_valid_qk": baseline["valid_qk"],
        "monowire_valid_qk": monowire["valid_qk"],
        "baseline_unique_kv": baseline["unique_kv_tokens"],
        "monowire_unique_kv": monowire["unique_kv_tokens"],
        "kv_tokens_saved": kv_tokens_saved,
        "kv_reduction_percent": 100 * kv_tokens_saved / baseline_kv,
        "kv_reduction": ratio(
            baseline["unique_kv_tokens"], monowire["unique_kv_tokens"]
        ),
        "baseline_tiles": baseline["scheduled_tiles"],
        "monowire_tiles": monowire["scheduled_tiles"],
        "tile_reduction": ratio(
            baseline["scheduled_tiles"], monowire["scheduled_tiles"]
        ),
        "baseline_launches": baseline["logical_launches"],
        "monowire_launches": monowire["logical_launches"],
        "launch_reduction": ratio(
            baseline["logical_launches"], monowire["logical_launches"]
        ),
    }
    if kv_bytes_per_token > 0:
        kv_bytes_saved = kv_tokens_saved * kv_bytes_per_token
        result["kv_bytes_per_token"] = kv_bytes_per_token
        result["kv_bytes_saved"] = kv_bytes_saved
        result["kv_gib_saved"] = kv_bytes_saved / 1024**3
    return result
