from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def token_f1(reference: str, candidate: str) -> float:
    reference_tokens = _tokens(reference)
    candidate_tokens = _tokens(candidate)
    if not reference_tokens and not candidate_tokens:
        return 1.0
    if not reference_tokens or not candidate_tokens:
        return 0.0
    reference_counts = _counts(reference_tokens)
    candidate_counts = _counts(candidate_tokens)
    common = sum(
        min(count, candidate_counts.get(token, 0))
        for token, count in reference_counts.items()
    )
    if not common:
        return 0.0
    precision = common / len(candidate_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def compare_raw_results(
    reference_batches: list[dict[str, Any]],
    candidate_batches: list[dict[str, Any]],
) -> dict[str, Any]:
    reference = _flatten_outputs(reference_batches)
    candidate = _flatten_outputs(candidate_batches)
    common_keys = sorted(reference.keys() & candidate.keys())
    rows = []
    for key in common_keys:
        reference_text = reference[key]
        candidate_text = candidate[key]
        normalized_reference = normalize_text(reference_text)
        normalized_candidate = normalize_text(candidate_text)
        rows.append(
            {
                "key": list(key),
                "scope": key[0],
                "exact_match": normalized_reference == normalized_candidate,
                "token_f1": token_f1(reference_text, candidate_text),
                "text_similarity": difflib.SequenceMatcher(
                    None, normalized_reference, normalized_candidate
                ).ratio(),
                "reference_text": reference_text,
                "candidate_text": candidate_text,
            }
        )

    exact_matches = [float(row["exact_match"]) for row in rows]
    token_f1s = [float(row["token_f1"]) for row in rows]
    similarities = [float(row["text_similarity"]) for row in rows]
    return {
        "comparison_type": "deterministic_output_agreement",
        "matched_outputs": len(rows),
        "missing_from_candidate": [
            list(key) for key in sorted(reference.keys() - candidate.keys())
        ],
        "extra_in_candidate": [
            list(key) for key in sorted(candidate.keys() - reference.keys())
        ],
        "normalized_exact_match_percent": _mean_percent(exact_matches),
        "mean_token_f1_percent": _mean_percent(token_f1s),
        "mean_text_similarity_percent": _mean_percent(similarities),
        "by_scope": {
            scope: _scope_summary([row for row in rows if row["scope"] == scope])
            for scope in ("common", "branch")
            if any(row["scope"] == scope for row in rows)
        },
        "rows": rows,
    }


def render_report(result: dict[str, Any], reference: Path, candidate: Path) -> str:
    lines = [
        "# Backend Output Agreement",
        "",
        f"Reference: `{reference}`",
        f"Candidate: `{candidate}`",
        "",
        "| Scope | Outputs | Exact match | Token F1 | Text similarity |",
        "|---|---:|---:|---:|---:|",
    ]
    for scope, summary in result["by_scope"].items():
        lines.append(
            f"| {scope} | {summary['outputs']} "
            f"| {summary['normalized_exact_match_percent']:.2f}% "
            f"| {summary['mean_token_f1_percent']:.2f}% "
            f"| {summary['mean_text_similarity_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"Matched outputs: {result['matched_outputs']}. Missing candidate outputs: "
            f"{len(result['missing_from_candidate'])}. Extra candidate outputs: "
            f"{len(result['extra_in_candidate'])}.",
            "",
            "> This is deterministic output agreement against the reference backend, "
            "not environment-level task accuracy. The bundled prompt snapshots do not "
            "contain a shared executable evaluator.",
            "",
        ]
    )
    return "\n".join(lines)


def _flatten_outputs(batches: list[dict[str, Any]]) -> dict[tuple[Any, ...], str]:
    outputs: dict[tuple[Any, ...], str] = {}
    for batch_index, batch in enumerate(batches):
        sample_start = int(batch.get("sample_start", batch_index))
        common = batch.get("common", {})
        for case in common.get("cases", []):
            key = ("common", sample_start, int(case["case_index"]))
            outputs[key] = str(case.get("text", ""))
        for branch in batch.get("branches", []):
            key = (
                "branch",
                sample_start,
                int(branch["case_index"]),
                int(branch["branch_id"]),
            )
            outputs[key] = str(branch.get("text", ""))
    return outputs


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", normalize_text(text), flags=re.UNICODE)


def _counts(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def _mean_percent(values: list[float]) -> float:
    return 100 * statistics.fmean(values) if values else 0.0


def _scope_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "outputs": len(rows),
        "normalized_exact_match_percent": _mean_percent(
            [float(row["exact_match"]) for row in rows]
        ),
        "mean_token_f1_percent": _mean_percent(
            [float(row["token_f1"]) for row in rows]
        ),
        "mean_text_similarity_percent": _mean_percent(
            [float(row["text_similarity"]) for row in rows]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = compare_raw_results(reference, candidate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "output_agreement.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "output_agreement.md").write_text(
        render_report(result, args.reference, args.candidate), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
