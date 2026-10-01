#!/usr/bin/env python3
"""Probe exact KV restoration and terminal-branch cache pollution.

Run against a dedicated vLLM backend; cache resets affect the whole engine.
Save output on the experiment server. This is a controlled lifecycle workload,
not an official AgentX replay. GPU/CPU pool budgets belong to the server config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families

COUNTERS = {
    "store_bytes": "vllm:kv_offload_store_bytes_total",
    "load_bytes": "vllm:kv_offload_load_bytes_total",
    "preemptions": "vllm:num_preemptions_total",
}
TOKEN_COUNTERS = {
    "tokens_local_compute",
    "tokens_local_cache_hit",
    "tokens_external_kv_transfer",
}


def save_result(path: Path, result: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def metrics(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    legacy_transfers: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            for key, name in COUNTERS.items():
                if sample.name == name:
                    result[key] = result.get(key, 0.0) + sample.value
            if sample.name == "vllm:kv_offload_total_bytes_total":
                key = {
                    "GPU_to_CPU": "store_bytes",
                    "CPU_to_GPU": "load_bytes",
                }.get(sample.labels.get("transfer_type"))
                if key is not None:
                    legacy_transfers[key] = legacy_transfers.get(key, 0.0) + sample.value
            if sample.name == "vllm:prompt_tokens_by_source_total":
                key = "tokens_" + sample.labels["source"]
                result[key] = result.get(key, 0.0) + sample.value
    if any(result[key] != legacy_transfers[key]
           for key in legacy_transfers.keys() & result.keys()):
        raise RuntimeError("ambiguous mixed offload metric versions")
    result.update(legacy_transfers)
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise RuntimeError("invalid server counter value")
    return result


def delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    result = {key: after.get(key, 0) - before.get(key, 0) for key in after | before}
    if any(value < 0 for value in result.values()):
        raise RuntimeError("server counters reset during the experiment")
    return result


class Backend:
    def __init__(self, url: str, model: str, output_tokens: int):
        self.url = url.rstrip("/")
        self.model = model
        self.output_tokens = output_tokens

    def _request(self, path: str, body=None):
        return urllib.request.urlopen(
            urllib.request.Request(
                self.url + path,
                data=json.dumps(body).encode() if body is not None else None,
                headers={"Content-Type": "application/json"},
            ),
            timeout=300,
        )

    def metrics(self) -> dict[str, float]:
        with self._request("/metrics") as response:
            return metrics(response.read().decode())

    def settled_metrics(
        self, before: dict[str, float], prompt_tokens: int, settle_seconds: float
    ) -> dict[str, float]:
        """Wait for request accounting and a quiet counter window, with a timeout.

        This is a sampling guard, not an engine transfer-completion fence.
        """
        deadline = time.monotonic() + 30 + settle_seconds
        previous = None
        stable_since = time.monotonic()
        while True:
            current = self.metrics()
            if not TOKEN_COUNTERS.issubset(current):
                raise RuntimeError("missing prompt-source counters")
            changes = delta(before, current)
            observed = sum(changes[key] for key in TOKEN_COUNTERS)
            if observed > prompt_tokens:
                raise RuntimeError("unexpected requests or prompt-token accounting")
            now = time.monotonic()
            if current != previous:
                stable_since = now
                previous = current
            if observed == prompt_tokens and now - stable_since >= settle_seconds:
                return current
            if now >= deadline:
                raise RuntimeError(
                    "server metrics did not settle or account for requests"
                )
            time.sleep(0.2)

    def reset(self, *, external: bool):
        deadline = time.monotonic() + 30
        while True:
            path = "/reset_prefix_cache?reset_external=" + str(external).lower()
            with self._request(path, {}) as response:
                payload = json.load(response)
            if payload.get("success") is True:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("cache reset did not complete")
            time.sleep(0.1)

    def infer(
        self, prompt: list[int], *, max_offload_tokens: int | None = None
    ) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.output_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_offload_tokens is not None:
            payload["kv_transfer_params"] = {"max_offload_tokens": max_offload_tokens}
        started = time.perf_counter()
        first_token = None
        tokens, text = [], []
        usage = None
        done = False
        with self._request("/v1/completions", payload) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])
                for choice in chunk.get("choices", []):
                    ids = choice.get("token_ids") or []
                    content = choice.get("text") or ""
                    if first_token is None and (ids or content):
                        first_token = time.perf_counter()
                    tokens.extend(ids)
                    text.append(content)
                usage = chunk.get("usage") or usage
        ended = time.perf_counter()
        if not done or first_token is None or usage is None:
            raise RuntimeError("incomplete completion stream")
        if len(tokens) != self.output_tokens or usage["completion_tokens"] != len(
            tokens
        ):
            raise RuntimeError("missing token IDs or unexpected output length")
        if usage["prompt_tokens"] != len(prompt):
            raise RuntimeError("server did not use the requested token sequence")
        return {
            "ttft_ms": (first_token - started) * 1000,
            "latency_ms": (ended - started) * 1000,
            "token_ids": tokens,
            "text": "".join(text),
            "usage": usage,
        }


def make_prompt(tokenizer, seed: int, session: int, tokens: int) -> list[int]:
    # Session identity precedes all filler, preventing accidental cross-session
    # prefix hits. The same prompt is reused at resume in every comparison arm.
    code = hashlib.sha256(f"{seed}/{session}".encode()).hexdigest()[:8]
    header = tokenizer.encode(
        f"Session {seed}/{session}. Remember the access code {code}.\n",
        add_special_tokens=False,
    )
    filler = tokenizer.encode(
        "Reference record: retain the access code while reviewing these notes.\n",
        add_special_tokens=False,
    )
    question = tokenizer.encode(
        "\nWhat is the access code? Reply with only the code.\nAnswer:",
        add_special_tokens=False,
    )
    available = tokens - len(header) - len(question)
    if available < 1 or not filler:
        raise ValueError("prompt budget is too small")
    return header + (filler * (available // len(filler) + 1))[:available] + question


def validate_trial(trial: dict, *, mode: str, scenario: str) -> None:
    for cold, resumed in zip(trial["prime"], trial["resume"], strict=True):
        if cold["token_ids"] != resumed["token_ids"]:
            raise RuntimeError(
                "cold/resumed token mismatch; do not report a memory gain"
            )
    if mode == "apc":
        if any(
            trial[phase].get(direction, 0) > 0
            for phase in ("prime_counters", "pressure_counters", "resume_counters")
            for direction in ("store_bytes", "load_bytes")
        ):
            raise RuntimeError("APC baseline unexpectedly used external KV storage")
    else:
        if trial["prime_counters"].get("store_bytes", 0) <= 0:
            raise RuntimeError("offload arm did not actually store KV")
        if scenario == "roundtrip":
            if trial["resume_counters"].get("load_bytes", 0) <= 0:
                raise RuntimeError("roundtrip did not restore from external storage")
            if trial["resume_counters"].get("tokens_local_cache_hit", 0) != 0:
                raise RuntimeError("roundtrip still reused local KV")
        if (
            mode == "selective"
            and trial["pressure_counters"].get("store_bytes", 0) != 0
        ):
            raise RuntimeError("terminal requests still wrote KV to external storage")


def run(args) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prompts = [
        make_prompt(tokenizer, args.seed, index, args.prompt_tokens)
        for index in range(args.sessions + args.pressure_requests)
    ]
    backend = Backend(args.base_url, args.model, args.output_tokens)
    initial = backend.metrics()
    required = {"preemptions"} | TOKEN_COUNTERS
    # Transfer counters are registered lazily on the first transfer. Check
    # actual store/load deltas in validate_trial, after exercising the path.
    if not required.issubset(initial):
        raise RuntimeError(
            f"missing server metrics: {sorted(required - initial.keys())}"
        )
    result = {
        "scope": "Controlled sequential KV lifecycle; no model-quality or AgentX claim",
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "prompt_plan_sha256": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        "trials": [],
        "valid": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)

    def save():
        save_result(args.output, result)

    try:
        for index in range(args.trials):
            backend.reset(external=True)
            time.sleep(args.settle_seconds)
            before = backend.metrics()
            trial = {"index": index, "prime": [], "pressure": [], "resume": []}
            result["trials"].append(trial)
            for prompt in prompts[: args.sessions]:
                trial["prime"].append(backend.infer(prompt))
                save()
            if args.scenario == "roundtrip":
                # A successful reset also checks that device blocks are no
                # longer held by asynchronous backups before measuring them.
                backend.reset(external=False)
            primed = backend.settled_metrics(
                before, args.sessions * args.prompt_tokens, args.settle_seconds
            )
            trial["prime_counters"] = delta(before, primed)
            if args.scenario == "pressure":
                # These are known terminal requests in the application plan,
                # not predicted from future traces. They are never revisited.
                for prompt in prompts[args.sessions :]:
                    trial["pressure"].append(
                        backend.infer(
                            prompt,
                            max_offload_tokens=0 if args.mode == "selective" else None,
                        )
                    )
                    save()
            pressured = backend.settled_metrics(
                primed,
                args.pressure_requests * args.prompt_tokens
                if args.scenario == "pressure"
                else 0,
                args.settle_seconds,
            )
            trial["pressure_counters"] = delta(primed, pressured)
            for prompt in prompts[: args.sessions]:
                trial["resume"].append(backend.infer(prompt))
                save()
            resumed = backend.settled_metrics(
                pressured, args.sessions * args.prompt_tokens, args.settle_seconds
            )
            trial["resume_counters"] = delta(pressured, resumed)
            validate_trial(trial, mode=args.mode, scenario=args.scenario)
            save()
        result["summary"] = {
            "mean_resume_ttft_ms": statistics.fmean(
                row["ttft_ms"] for trial in result["trials"] for row in trial["resume"]
            ),
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
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["apc", "offload", "selective"], required=True
    )
    parser.add_argument("--scenario", choices=["roundtrip", "pressure"], required=True)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--pressure-requests", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if min(args.sessions, args.prompt_tokens, args.output_tokens, args.trials) < 1:
        parser.error(
            "sessions, prompt tokens, output tokens and trials must be positive"
        )
    if (
        args.pressure_requests < 0
        or not math.isfinite(args.settle_seconds)
        or args.settle_seconds < 0
    ):
        parser.error("pressure requests and settle seconds must be nonnegative")
    if args.scenario == "pressure" and not args.pressure_requests:
        parser.error("pressure scenario requires terminal requests")
    print(json.dumps(run(args)["summary"], indent=2))


if __name__ == "__main__":
    main()
