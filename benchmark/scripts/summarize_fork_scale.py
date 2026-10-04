"""Validate paired ForkAttention runs and average independent service repeats.

Run on the experiment server. Input/output identity and equal prompt work are
required before latency is summarized. Batch repeats are not independent runs.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from benchmark_fork_scale import make_prompts


def require(condition, message):
    if not condition:
        raise ValueError(message)


def indexed(items, key):
    result = {}
    for item in items:
        identity = key(item)
        require(identity not in result, f"Duplicate identity: {identity}")
        result[identity] = item
    return result


def validate_run(run):
    require(run.get("valid") is True, "Incomplete or invalid run")
    config = run["config"]
    owners = run["session_owners"]
    require(set(owners) == {"0", "1"}, "Expected two DP ranks")
    require(len(set(owners.values())) == 2, "Session owners overlap")
    shapes = indexed(
        run["shapes"], lambda s: (s["prefix_tokens"], s["branches_per_rank"])
    )
    require(
        set(shapes)
        == {(p, b) for p in config["prefix_tokens"] for b in config["branches"]},
        "Missing or unexpected shapes",
    )
    require(bool(shapes) and config["trials"] > 0, "Empty experiment")
    for (prefix, branches), shape in shapes.items():
        digest = hashlib.sha256(
            json.dumps(make_prompts(config["seed"], prefix, branches)).encode()
        ).hexdigest()
        require(shape["input_sha256"] == digest, "Input plan hash mismatch")
        trials = indexed(shape["trials"], lambda t: t["trial"])
        require(set(trials) == set(range(config["trials"])), "Incomplete batch repeats")
        for trial in trials.values():
            require(
                math.isfinite(trial["started"])
                and math.isfinite(trial["ended"])
                and trial["ended"] > trial["started"],
                "Invalid measurement window",
            )
            rows = indexed(trial["rows"], lambda r: (r["expected_rank"], r["branch"]))
            require(
                set(rows) == {(rank, b) for rank in owners for b in range(branches)},
                "Missing or unexpected branches",
            )
            require(set(trial["rank_counters"]) == set(owners), "Missing rank counters")
            for rank, counts in trial["rank_counters"].items():
                require(
                    all(math.isfinite(v) and v >= 0 for v in counts.values()),
                    "Invalid counters",
                )
                require(counts.get("preemptions", 0) == 0, "Preempted batch")
                require(
                    sum(
                        counts.get("tokens_" + source, 0)
                        for source in (
                            "local_compute",
                            "local_cache_hit",
                            "external_kv_transfer",
                        )
                    )
                    == branches * (prefix + 128),
                    "Incomplete prompt accounting",
                )
                require(
                    counts.get("tokens_local_cache_hit", 0) >= branches * prefix,
                    "Cold shared prefix",
                )
            for (rank, _), row in rows.items():
                require(row["document"] == owners[rank], "Wrong session owner")
                require(row["prompt_tokens"] == prefix + 128, "Input length mismatch")
                require(
                    row["completion_tokens"] == config["output_tokens"],
                    "Output length mismatch",
                )
                require(
                    isinstance(row["output_token_sha256"], str)
                    and len(row["output_token_sha256"]) == 64
                    and all(
                        c in "0123456789abcdef" for c in row["output_token_sha256"]
                    ),
                    "Missing output token digest",
                )
                require(
                    math.isfinite(row["e2e_ms"])
                    and 0 < row["ttft_ms"] <= row["e2e_ms"],
                    "Invalid request timing",
                )
            require(
                trial["output_tokens"] == len(rows) * config["output_tokens"],
                "Wrong output total",
            )
            require(
                math.isfinite(trial["wall_seconds"]) and trial["wall_seconds"] > 0,
                "Invalid batch timing",
            )
    return shapes


def compare_runs(baseline, candidate, *, diagnose_output_differences=False):
    a, b = validate_run(baseline), validate_run(candidate)
    ignored = {"output", "base_url", "control_url"}
    require(
        {k: v for k, v in baseline["config"].items() if k not in ignored}
        == {k: v for k, v in candidate["config"].items() if k not in ignored},
        "Benchmark configurations differ",
    )
    require(
        baseline["session_owners"] == candidate["session_owners"],
        "Session placement differs",
    )
    result = []
    for key in sorted(a):
        at = indexed(a[key]["trials"], lambda t: t["trial"])
        bt = indexed(b[key]["trials"], lambda t: t["trial"])
        count = 0
        differences = []
        for trial in at:
            require(
                at[trial]["rank_counters"] == bt[trial]["rank_counters"],
                "Prompt work differs",
            )
            ar = indexed(at[trial]["rows"], lambda r: (r["expected_rank"], r["branch"]))
            br = indexed(bt[trial]["rows"], lambda r: (r["expected_rank"], r["branch"]))
            differences.extend(
                {"trial": trial, "rank": rank, "branch": branch}
                for rank, branch in ar
                if ar[(rank, branch)]["output_token_sha256"]
                != br[(rank, branch)]["output_token_sha256"]
            )
            count += len(ar)
        require(
            diagnose_output_differences or not differences, "Generated tokens differ"
        )
        row = {
            "prefix_tokens": key[0],
            "branches_per_rank": key[1],
            "output_pairs": count,
            "equal_output_pairs": count - len(differences),
            "output_differences": differences,
            "equal_output_latency_comparison": not differences,
            "within_service_variable_branches": {},
        }
        for name, trials in (("baseline", at), ("candidate", bt)):
            rows = [r for t in trials.values() for r in t["rows"]]
            outputs = {}
            for r in rows:
                outputs.setdefault((r["expected_rank"], r["branch"]), set()).add(
                    r["output_token_sha256"]
                )
            row["within_service_variable_branches"][name] = sum(
                len(v) > 1 for v in outputs.values()
            )
            row[name] = {
                "wall_seconds": statistics.fmean(
                    t["wall_seconds"] for t in trials.values()
                ),
                "ttft_ms": statistics.fmean(r["ttft_ms"] for r in rows),
                "decode_response_ms": statistics.fmean(
                    r["e2e_ms"] - r["ttft_ms"] for r in rows
                ),
            }
        row["wall_change_pct"] = (
            row["candidate"]["wall_seconds"] / row["baseline"]["wall_seconds"] - 1
        ) * 100
        result.append(row)
    return result


def summarize(runs, seeds, restarts, *, diagnose_output_differences=False):
    require(
        len(set(seeds)) == len(seeds) and bool(seeds) and restarts > 0,
        "Invalid repeat plan",
    )
    progress = json.loads((runs / "progress.json").read_text())
    expected = {
        f"{seed}-{arm}-{restart}-fork-scale"
        for seed in seeds
        for restart in range(restarts)
        for arm in ("consistent_hash", "consistent_hash_fork")
    }
    require(
        progress["phase"] == "complete" and set(progress["completed"]) == expected,
        "Experiment plan is incomplete",
    )
    pairs = {}
    groups = {}
    reference_config = None
    for seed in seeds:
        for restart in range(restarts):
            arms = [
                json.loads(
                    (runs / f"{seed}-{arm}-{restart}-fork-scale.json").read_text()
                )
                for arm in ("consistent_hash", "consistent_hash_fork")
            ]
            require(
                all(r["config"]["seed"] == seed for r in arms),
                "Seed differs from run label",
            )
            shared_config = {
                k: v
                for k, v in arms[0]["config"].items()
                if k not in {"seed", "output", "base_url", "control_url"}
            }
            if reference_config is None:
                reference_config = shared_config
            require(
                shared_config == reference_config,
                "Configuration changed across service repeats",
            )
            rows = compare_runs(
                *arms, diagnose_output_differences=diagnose_output_differences
            )
            pairs[f"{seed}/{restart}"] = rows
            for row in rows:
                groups.setdefault(
                    (row["prefix_tokens"], row["branches_per_rank"]), []
                ).append(row)
    aggregate = []
    for (prefix, branches), rows in sorted(groups.items()):
        require(len(rows) == len(seeds) * restarts, "Unbalanced shape repeats")
        group = {
            "prefix_tokens": prefix,
            "branches_per_rank": branches,
            "service_pairs": len(rows),
            "output_pairs": sum(r["output_pairs"] for r in rows),
            "equal_output_pairs": sum(r["equal_output_pairs"] for r in rows),
            "equal_output_latency_comparison": all(
                r["equal_output_latency_comparison"] for r in rows
            ),
        }
        for arm in ("baseline", "candidate"):
            group[arm] = {
                metric: statistics.fmean(r[arm][metric] for r in rows)
                for metric in rows[0][arm]
            }
        group["wall_change_pct"] = (
            group["candidate"]["wall_seconds"] / group["baseline"]["wall_seconds"] - 1
        ) * 100
        group["service_pair_change_range_pct"] = [
            min(r["wall_change_pct"] for r in rows),
            max(r["wall_change_pct"] for r in rows),
        ]
        group["faster_service_pairs"] = sum(r["wall_change_pct"] < 0 for r in rows)
        aggregate.append(group)
    return {
        "scope": "Equal-weight service means, balanced seeds/restarts; batch repeats are not independent. Shapes with output differences are diagnostic timing only, not equal-output speedups. No significance or QA claim. Runtime/config isolation and memory audited separately.",
        "service_pairs": len(pairs),
        "output_pairs": sum(r["output_pairs"] for rows in pairs.values() for r in rows),
        "equal_output_pairs": sum(
            r["equal_output_pairs"] for rows in pairs.values() for r in rows
        ),
        "pairs": pairs,
        "aggregate": aggregate,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--restarts", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnose-output-differences",
        action="store_true",
        help="Keep mismatched outputs as explicitly ineligible diagnostics; never mark them as equal-output speedups",
    )
    args = parser.parse_args()
    result = summarize(
        args.runs,
        args.seeds,
        args.restarts,
        diagnose_output_differences=args.diagnose_output_differences,
    )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=2))
