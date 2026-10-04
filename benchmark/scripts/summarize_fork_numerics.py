"""Audit full-token diagnostics and original-question QA on the experiment server.

Logprobs are compared only until the generated histories diverge. Diagnostic
request timings are never used as performance evidence. QA gates are evaluated
per service pair before combining seeds; pooled scores cannot hide a regression.
"""

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

from benchmark_fork_numerics import compare_outputs
from longbench_qa import score_answer
from summarize_fork_scale import indexed, require, validate_run


def numeric_rows(run):
    shapes = validate_run(run)
    result = {}
    for shape, data in shapes.items():
        for trial in data["trials"]:
            for row in trial["rows"]:
                require(
                    row["output_token_sha256"]
                    == hashlib.sha256(
                        json.dumps(row["token_ids"]).encode()
                    ).hexdigest(),
                    "Token digest does not match captured IDs",
                )
                require(
                    len(row["token_ids"]) == run["config"]["output_tokens"]
                    and row["logprob_tokens"]
                    == [f"token_id:{token}" for token in row["token_ids"]],
                    "Token IDs and logprob labels are not aligned",
                )
                # Validate even rows with no partner (e.g. a single trial).
                compare_outputs(row, row)
                result[
                    (*shape, trial["trial"], row["expected_rank"], row["branch"])
                ] = row
    return result


def compare_numeric(a, b=None):
    ar = numeric_rows(a)
    pairs = []
    if b is None:
        for p, n in sorted({key[:2] for key in ar}):
            for t1, t2 in itertools.combinations(range(a["config"]["trials"]), 2):
                pairs.extend(
                    (key, (p, n, t2, rank, branch))
                    for key in ar
                    for prefix, branches, trial, rank, branch in [key]
                    if (prefix, branches, trial) == (p, n, t1)
                )
        br = ar
    else:
        br = numeric_rows(b)
        ignored = {"output", "base_url", "control_url", "request_mode"}
        require(
            {k: v for k, v in a["config"].items() if k not in ignored}
            == {k: v for k, v in b["config"].items() if k not in ignored},
            "Numeric configurations differ",
        )
        require(a["session_owners"] == b["session_owners"], "Session placement differs")
        require(ar.keys() == br.keys(), "Numeric request sets differ")
        ac = {
            (s["prefix_tokens"], s["branches_per_rank"], t["trial"]): t["rank_counters"]
            for s in a["shapes"]
            for t in s["trials"]
        }
        bc = {
            (s["prefix_tokens"], s["branches_per_rank"], t["trial"]): t["rank_counters"]
            for s in b["shapes"]
            for t in s["trials"]
        }
        require(ac == bc, "Prompt work differs")
        pairs = [(key, key) for key in sorted(ar)]
    groups = {}
    for ka, kb in pairs:
        value = compare_outputs(ar[ka], br[kb])
        group = groups.setdefault(
            ka[:2],
            {"pairs": 0, "different_pairs": [], "max_common_logprob_difference": 0.0},
        )
        group["pairs"] += 1
        group["max_common_logprob_difference"] = max(
            group["max_common_logprob_difference"],
            value["max_common_logprob_difference"] or 0.0,
        )
        if not value["tokens_equal"]:
            group["different_pairs"].append(
                {"baseline_key": ka, "candidate_key": kb, **value}
            )
    return [
        {
            "prefix_tokens": p,
            "branches_per_rank": n,
            "equal_pairs": g["pairs"] - len(g["different_pairs"]),
            **g,
        }
        for (p, n), g in sorted(groups.items())
    ]


def qa_rows(run, cases):
    expected = indexed(
        [
            {
                "case_id": c["case_id"],
                "context_sha256": c["context_sha256"],
                "context_tokens": c["context_tokens"],
                "phase": "first" if i == 0 else "followup",
                **q,
            }
            for c in cases
            for i, q in enumerate(c["questions"])
        ],
        lambda r: (r["case_id"], r["source_id"]),
    )
    rows = indexed(run["results"], lambda r: (r["case_id"], r["source_id"]))
    require(rows.keys() == expected.keys(), "QA questions are missing or duplicated")
    require(
        run["case_count"] == len(cases) and run["question_count"] == len(rows),
        "QA totals differ",
    )
    require(
        run["prime_first_question"] is True
        and not run["question_waves"]
        and not run["coalesce_prefill"],
        "Unexpected QA arrival pattern",
    )
    for key, row in rows.items():
        require(not row.get("error"), "QA request error")
        for field in (
            "question",
            "answers",
            "context_sha256",
            "context_tokens",
            "phase",
        ):
            require(row[field] == expected[key][field], f"QA source changed: {field}")
        scores = score_answer(row["prediction"], row["answers"])
        for metric in ("f1", "exact_match"):
            require(
                math.isclose(row[metric], scores[metric], abs_tol=1e-12),
                "QA score mismatch",
            )
        require(row["finish_reason"] in ("stop", "length"), "QA request did not finish")
        require(
            row["usage"] and row["usage"]["prompt_tokens"] > row["context_tokens"],
            "Missing QA usage",
        )
    for metric in ("f1", "exact_match"):
        require(
            math.isclose(
                run["mean_" + metric],
                statistics.fmean(r[metric] for r in rows.values()),
                abs_tol=1e-12,
            ),
            "QA mean mismatch",
        )
    for metric in ("prompt_tokens", "completion_tokens"):
        require(
            run[metric] == sum(r["usage"][metric] for r in rows.values()),
            "QA token totals differ",
        )
    for duration in (run["wall_seconds"], *run["phase_seconds"].values()):
        require(math.isfinite(duration) and duration > 0, "Invalid QA timing")
    require(set(run["phase_seconds"]) == {"first", "followup"}, "Missing QA phase")
    return rows


def compare_qa(a, b, cases):
    ar, br = qa_rows(a, cases), qa_rows(b, cases)
    for field in ("cases_sha256", "seed", "model", "document_routing"):
        require(a[field] == b[field], f"QA configuration differs: {field}")
    require(
        all(
            ar[k]["usage"]["prompt_tokens"] == br[k]["usage"]["prompt_tokens"]
            for k in ar
        ),
        "QA prompt lengths differ",
    )
    result = {
        "seed": a["seed"],
        "questions": len(ar),
        "equal_predictions": sum(
            ar[k]["prediction"] == br[k]["prediction"] for k in ar
        ),
    }
    for name, run in (("baseline", a), ("candidate", b)):
        result[name] = {
            k: run[k]
            for k in (
                "wall_seconds",
                "mean_f1",
                "mean_exact_match",
                "mean_ttft_seconds",
                "mean_end_to_end_ttft_seconds",
                "prompt_tokens",
                "completion_tokens",
            )
        }
        result[name].update(
            {f"{k}_seconds": v for k, v in run["phase_seconds"].items()}
        )
        result[name]["truncated"] = sum(
            r["finish_reason"] == "length" for r in run["results"]
        )
    result["quality_gate_passed"] = (
        all(b["mean_" + m] >= a["mean_" + m] for m in ("f1", "exact_match"))
        and result["candidate"]["truncated"] <= result["baseline"]["truncated"]
    )
    result["wall_change_pct"] = (b["wall_seconds"] / a["wall_seconds"] - 1) * 100
    result["followup_change_pct"] = (
        b["phase_seconds"]["followup"] / a["phase_seconds"]["followup"] - 1
    ) * 100
    result["changed_predictions"] = [
        {
            "case_id": k[0],
            "source_id": k[1],
            "baseline_f1": ar[k]["f1"],
            "candidate_f1": br[k]["f1"],
        }
        for k in ar
        if ar[k]["prediction"] != br[k]["prediction"]
    ]
    return result


def summarize(root):
    plan = json.loads((root / "validation-plan.json").read_text())
    progress = json.loads((root / "progress.json").read_text())
    services, modes = plan["services"], plan["numeric_modes"]
    require(
        set(services) == {"native-0", "fork-0", "fork-1", "native-1"}
        and len(services) == 4,
        "Expected balanced independent services",
    )
    require(
        set(modes) == {"concurrent", "batched"} and len(modes) == 2,
        "Expected both arrival controls",
    )
    expected = {f"{s}-{m}" for s in services for m in [*modes, "qa"]}
    require(
        progress["phase"] == "complete"
        and set(progress["completed"]) == expected
        and len(progress["completed"]) == len(expected),
        "Experiment plan is incomplete",
    )
    runs = {
        f"{s}-{m}": json.loads((root / "runs" / f"{s}-{m}.json").read_text())
        for s in services
        for m in modes
    }
    for label, run in runs.items():
        config = run["config"]
        require(
            label.endswith("-" + config["request_mode"]),
            "Request mode differs from run label",
        )
        for field, declared in (
            ("seed", "numeric_seed"),
            ("prefix_tokens", "numeric_prefix_tokens"),
            ("branches", "numeric_branches_per_rank"),
            ("output_tokens", "numeric_output_tokens"),
            ("trials", "numeric_repeats_per_shape"),
        ):
            require(
                config[field] == plan[declared],
                "Numeric run differs from predeclared plan",
            )
    comparisons = {
        f"within/{label}": compare_numeric(run) for label, run in runs.items()
    }
    for mode in modes:
        for index in range(2):
            comparisons[f"cross_arm/{index}/{mode}"] = compare_numeric(
                runs[f"native-{index}-{mode}"], runs[f"fork-{index}-{mode}"]
            )
        for arm in ("native", "fork"):
            comparisons[f"restart/{arm}/{mode}"] = compare_numeric(
                runs[f"{arm}-0-{mode}"], runs[f"{arm}-1-{mode}"]
            )
    for service in services:
        comparisons[f"arrival/{service}"] = compare_numeric(
            runs[f"{service}-concurrent"], runs[f"{service}-batched"]
        )
    case_data = (root / "qa-cases.jsonl").read_bytes()
    case_digest = hashlib.sha256(case_data).hexdigest()
    require(
        case_digest == json.loads((root / "qa-ready.json").read_text())["cases_sha256"],
        "Frozen QA cases changed",
    )
    cases = [json.loads(line) for line in case_data.splitlines() if line.strip()]
    qa = []
    for index, seed in enumerate(plan["qa_seeds"]):
        pair = [
            json.loads((root / "runs" / f"{arm}-{index}-qa.json").read_text())
            for arm in ("native", "fork")
        ]
        require(
            all(r["seed"] == seed and r["cases_sha256"] == case_digest for r in pair),
            "QA run differs from frozen input plan",
        )
        qa.append(compare_qa(*pair, cases))
    require(len(qa) == 2, "Expected two QA seeds")
    aggregate = {
        name: {
            metric: statistics.fmean(p[name][metric] for p in qa)
            for metric in qa[0][name]
        }
        for name in ("baseline", "candidate")
    }
    aggregate["quality_gate_passed"] = all(p["quality_gate_passed"] for p in qa)
    for metric in ("wall", "followup"):
        aggregate[metric + "_change_pct"] = (
            aggregate["candidate"][metric + "_seconds"]
            / aggregate["baseline"][metric + "_seconds"]
            - 1
        ) * 100
    return {
        "scope": "Same-history probability comparisons; diagnostic latency excluded. Original-question QA gates apply to every seed, not only pooled scores. Memory and execution-path coverage require separate runtime audit.",
        "numerics": comparisons,
        "qa_pairs": qa,
        "qa_aggregate": aggregate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result["qa_aggregate"], indent=2))


if __name__ == "__main__":
    main()
