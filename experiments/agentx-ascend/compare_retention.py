"""Validate matched official workloads and summarize hybrid-retention A/B runs."""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path


def read_run(path):
    report = json.loads((path / "artifacts/profile_export_aiperf.json").read_text())
    manifest = json.loads((path / "run-manifest.json").read_text())
    summary = json.loads((path / "result-summary.json").read_text())
    config = copy.deepcopy(report["input_config"])
    config["endpoint"].pop("urls", None)
    config["artifacts"].pop("dir", None)
    latest = {}
    log = path / "server.log"
    diagnostic_path = path / "retention-diagnostics.json"
    if diagnostic_path.exists():
        # Keep the end-of-measurement snapshot if the service later handles
        # correctness probes or interactive traffic.
        latest = json.loads(diagnostic_path.read_text())
    elif log.exists():
        for line in log.read_text().splitlines():
            match = re.search(
                r"EngineCore_DP(\d+).*Ascend Mamba retention: (\{.*\})", line
            )
            if match:
                latest[match[1]] = json.loads(match[2])
        diagnostic_path.write_text(json.dumps(latest, indent=2) + "\n")
    else:
        raise FileNotFoundError(f"No retention diagnostics in {path}")
    return (
        config,
        manifest,
        {
            "retention": manifest["mamba_cache_retention"],
            "summary": summary,
            "latest_diagnostic_snapshots": latest,
        },
    )


def compare(
    reference, candidate, allow_code_change=False, allow_execution_change=False
):
    ref_config, ref_manifest, ref = read_run(reference)
    new_config, new_manifest, new = read_run(candidate)
    if ref_config != new_config:
        raise ValueError("Official input_config differs beyond URL/output paths")
    for key in (
        "model",
        "harness_commit",
        "dataset_revision",
        "dataset_sha256",
        "hardware",
        "tensor_parallel_size",
        "data_parallel_size",
        "max_model_len",
        "dtype",
        "gpu_memory_utilization",
        "session_routing",
        "software_versions",
        "max_num_batched_tokens",
        "max_num_seqs",
        "long_prefill_token_threshold",
    ):
        if ref_manifest[key] != new_manifest[key]:
            raise ValueError(f"Uncontrolled difference: {key}")
    for key in (
        "kv_cache_memory_bytes",
        "kv_capacity_tokens_per_rank",
        "launcher_sha256",
        "batch_diagnostics",
    ):
        if ref_manifest.get(key) != new_manifest.get(key):
            raise ValueError(f"Uncontrolled difference: {key}")
    if ref_manifest.get("profiler_dir") or new_manifest.get("profiler_dir"):
        raise ValueError(
            "Profiling runs cannot be used as official performance comparisons"
        )
    execution_changes = {
        key: {"reference": ref_manifest.get(key), "candidate": new_manifest.get(key)}
        for key in (
            "enforce_eager",
            "execution_mode",
            "compilation_config",
            "npugraph_ex",
            "cpu_binding",
        )
        if ref_manifest.get(key) != new_manifest.get(key)
    }
    if execution_changes and not allow_execution_change:
        raise ValueError(
            "Execution configuration differs (use --allow-execution-change)"
        )
    if (
        allow_execution_change
        and ref_manifest["mamba_cache_retention"]
        != new_manifest["mamba_cache_retention"]
    ):
        raise ValueError("Retention policy must match when comparing execution modes")
    changed_files = {
        name: {
            "reference": ref_manifest["plugin_sha256"].get(name),
            "candidate": digest,
        }
        for name, digest in new_manifest["plugin_sha256"].items()
        if ref_manifest["plugin_sha256"].get(name) != digest
    }
    for name, digest in ref_manifest["plugin_sha256"].items():
        if name not in new_manifest["plugin_sha256"]:
            changed_files[name] = {"reference": digest, "candidate": None}
    if changed_files and not allow_code_change:
        raise ValueError(
            "Uncontrolled difference: plugin_sha256 (use --allow-code-change for a code revision comparison)"
        )
    for run in (ref, new):
        summary = run["summary"]
        if (
            not summary["submission_valid"]
            or summary["was_cancelled"]
            or summary["request_errors"]
        ):
            raise ValueError(
                "Expected completed, valid official runs without request errors"
            )
    ref_summary, new_summary = ref["summary"], new["summary"]
    result = {
        "reference": ref,
        "candidate": new,
        "changed_plugin_files": changed_files,
        "official_workload_sha256": hashlib.sha256(
            json.dumps(ref_config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "relative_change_percent": {
            "output_tokens_per_second": 100
            * (
                new_summary["output_tokens_per_second"]
                / ref_summary["output_tokens_per_second"]
                - 1
            ),
            "mean_ttft": 100
            * (new_summary["ttft_ms"]["avg"] / ref_summary["ttft_ms"]["avg"] - 1),
            "p90_ttft": 100
            * (new_summary["ttft_ms"]["p90"] / ref_summary["ttft_ms"]["p90"] - 1),
        },
        "notes": [
            "One run per configuration; closed-loop arrivals and completed request mixtures may differ.",
            "Diagnostic snapshots are cumulative process samples including warmup; do not sum snapshots.",
            "Diagnostic query counts include retries and are not executed recomputation token counts.",
            "Preallocated NPU cache-pool size is unchanged; policy changes retained content and reuse.",
        ],
    }
    if execution_changes:
        result["changed_execution_settings"] = execution_changes
        result["notes"][-1] = (
            "KV cache budgets and reported per-rank capacities match; graph memory is additional."
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-execution-change",
        action="store_true",
        help="Compare execution modes while requiring the same retention and cache capacity",
    )
    parser.add_argument(
        "--allow-code-change",
        action="store_true",
        help="Report an explicit code revision comparison",
    )
    args = parser.parse_args()
    result = compare(
        args.reference,
        args.candidate,
        args.allow_code_change,
        args.allow_execution_change,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["relative_change_percent"], indent=2))
