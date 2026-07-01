from __future__ import annotations

import math
import random
from typing import Any

from models import BenchmarkTrace, BranchTrace


def sample_suffixes(
    count: int, distribution: str, mean: int, rng: random.Random
) -> list[int]:
    if count <= 0 or mean < 0:
        raise ValueError("count must be positive and mean must be non-negative")
    if distribution == "equal":
        values = [float(mean)] * count
    elif distribution == "uniform":
        values = [rng.uniform(0.5 * mean, 1.5 * mean) for _ in range(count)]
    elif distribution == "lognormal":
        sigma = 0.75
        mu = math.log(max(mean, 1)) - sigma * sigma / 2
        values = [rng.lognormvariate(mu, sigma) for _ in range(count)]
    elif distribution in {"long_tail", "long-tail"}:
        values = [rng.paretovariate(2.0) for _ in range(count)]
    else:
        raise ValueError(f"unsupported suffix distribution: {distribution}")
    if distribution != "equal":
        observed_mean = sum(values) / count
        scale = mean / observed_mean if observed_mean else 0
        values = [value * scale for value in values]
    return [max(0, round(value)) for value in values]


def traces_from_config(config: dict[str, Any]) -> list[BenchmarkTrace]:
    rng = random.Random(int(config.get("seed", 2026)))
    suffix_mean = int(config.get("suffix_mean", 768))
    output_tokens = int(config.get("output_tokens", 256))
    arrival_mode = str(config.get("arrival_mode", "simultaneous"))
    traces: list[BenchmarkTrace] = []
    for prefix in config["prefix_tokens"]:
        for branches in config["branch_counts"]:
            for distribution in config["suffix_distributions"]:
                suffixes = sample_suffixes(
                    int(branches), str(distribution), suffix_mean, rng
                )
                case_id = f"p{int(prefix) // 1024}k_b{branches}_{distribution}"
                traces.append(
                    BenchmarkTrace(
                        case_id=case_id,
                        prefix_tokens=int(prefix),
                        branches=[
                            BranchTrace(index, suffix, output_tokens)
                            for index, suffix in enumerate(suffixes)
                        ],
                        suffix_distribution=str(distribution),
                        output_tokens=output_tokens,
                        arrival_mode=arrival_mode,
                    )
                )
    return traces
