"""Measure official sticky routing with bounded per-rank CPU KV backup.

This is a sequential pressure probe, not an AgentX or concurrency benchmark.
Only a dedicated backend may be used: resets affect all DP ranks.
"""

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path

from benchmark_agent_kv_tiering import (
    TOKEN_COUNTERS,
    Backend,
    delta,
    make_prompt,
    metrics,
    save_result,
    validate_trial,
)
from prometheus_client.parser import text_string_to_metric_families


class RoutedBackend(Backend):
    def __init__(self, url, control_url, model, output_tokens):
        super().__init__(url, model, output_tokens)
        self.control_url = control_url.rstrip("/")
        self.session_id = None

    def _request(self, path, body=None):
        inference = path == "/v1/completions"
        headers = {"Content-Type": "application/json"}
        if inference:
            if self.session_id is None:
                raise ValueError("inference requires a stable session ID")
            headers["X-Session-ID"] = self.session_id
        return urllib.request.urlopen(
            urllib.request.Request(
                (self.url if inference else self.control_url) + path,
                data=json.dumps(body).encode() if body is not None else None,
                headers=headers,
            ),
            timeout=300,
        )

    def metrics(self):
        with self._request("/metrics") as response:
            text = response.read().decode()
        ranks = {
            sample.labels["engine"]
            for family in text_string_to_metric_families(text)
            for sample in family.samples
            if sample.name == "vllm:num_preemptions_total" and "engine" in sample.labels
        }
        if len(ranks) != 2:
            raise RuntimeError("expected exactly two observed DP ranks")
        result = metrics(text)
        for rank in ranks:
            values = metrics(text, engine=rank)
            if not TOKEN_COUNTERS.issubset(values):
                raise RuntimeError("missing per-rank prompt-source counters")
            result.update(
                {f"rank/{rank}/{key}": value for key, value in values.items()}
            )
        return result


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def run(args):
    from transformers import AutoTokenizer

    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prompts = [
        make_prompt(tokenizer, args.seed, index, args.prompt_tokens)
        for index in range(args.sessions + args.pressure_requests)
    ]
    backend = RoutedBackend(
        args.base_url, args.control_url, args.model, args.output_tokens
    )
    result = {
        "valid": False,
        "scope": "Sequential dual-rank lifecycle pressure; no concurrent-capacity claim",
        "configuration": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "prompt_plan_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        "trials": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for trial_index in range(args.trials):
            # A backend without a connector cannot reset external storage.
            backend.reset(external=args.mode != "apc")
            time.sleep(args.settle_seconds)
            previous = backend.metrics()
            trial = {"index": trial_index}
            result["trials"].append(trial)
            started = time.monotonic()
            for phase, indices in (
                ("prime", range(args.sessions)),
                ("pressure", range(args.sessions, len(prompts))),
                ("resume", range(args.sessions)),
            ):
                if phase == "resume":
                    time.sleep(args.tool_gap_seconds)
                trial[phase] = []
                for index in indices:
                    # Identity must not depend on arm, pool size, phase, or trial.
                    backend.session_id = f"kv-lifecycle-{args.seed}-{index}"
                    row = backend.infer(
                        prompts[index],
                        max_offload_tokens=0
                        if phase == "pressure" and args.mode == "selective"
                        else None,
                    )
                    row["session"] = index
                    trial[phase].append(row)
                    save_result(args.output, result)
                after = backend.settled_metrics(
                    previous, len(indices) * args.prompt_tokens, args.settle_seconds
                )
                changes = delta(previous, after)
                trial[phase + "_counters"] = changes
                previous = after
            trial["elapsed_seconds"] = time.monotonic() - started
            validate_trial(trial, mode=args.mode, scenario="pressure")
            for phase in ("prime", "pressure", "resume"):
                counters = trial[phase + "_counters"]
                if counters.get("preemptions", 0):
                    raise RuntimeError(
                        "preemption occurred; reject this pressure point"
                    )
                ranks = {
                    key.split("/")[1] for key in counters if key.startswith("rank/")
                }
                if any(
                    sum(counters.get(f"rank/{r}/{k}", 0) for k in TOKEN_COUNTERS) == 0
                    for r in ranks
                ):
                    raise RuntimeError("workload did not exercise both ranks")
        latencies = [row["ttft_ms"] for t in result["trials"] for row in t["resume"]]
        result["summary"] = {
            "mean_resume_ttft_ms": statistics.fmean(latencies),
            "p95_resume_ttft_ms": percentile(latencies, 0.95),
            "elapsed_seconds": sum(t["elapsed_seconds"] for t in result["trials"]),
            "pressure_store_bytes": sum(
                t["pressure_counters"].get("store_bytes", 0) for t in result["trials"]
            ),
            "resume_load_bytes": sum(
                t["resume_counters"].get("load_bytes", 0) for t in result["trials"]
            ),
            "resume_recomputed_tokens": sum(
                t["resume_counters"]["tokens_local_compute"] for t in result["trials"]
            ),
        }
        result["valid"] = True
    except Exception as error:
        result["error"] = repr(error)
        raise
    finally:
        save_result(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-url", "control-url", "model"):
        parser.add_argument("--" + name, required=True)
    for name in ("tokenizer", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument(
        "--mode", required=True, choices=["apc", "offload", "selective"]
    )
    for name, default in (
        ("sessions", 8),
        ("pressure-requests", 16),
        ("prompt-tokens", 8193),
        ("output-tokens", 16),
        ("trials", 2),
        ("seed", 20261001),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--settle-seconds", type=float, default=1)
    parser.add_argument("--tool-gap-seconds", type=float, default=0)
    args = parser.parse_args()
    if (
        min(
            args.sessions,
            args.pressure_requests,
            args.prompt_tokens,
            args.output_tokens,
            args.trials,
        )
        < 1
    ):
        parser.error("request counts and token budgets must be positive")
    if any(
        not math.isfinite(v) or v < 0
        for v in (args.settle_seconds, args.tool_gap_seconds)
    ):
        parser.error("delays must be finite and nonnegative")
    print(json.dumps(run(args)["summary"], indent=2))


if __name__ == "__main__":
    main()
