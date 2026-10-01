from __future__ import annotations

import copy
import io
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from offload_revisit_trace import build_revisit_trace


def test_build_revisit_trace_selects_longest_unconstrained_request() -> None:
    source = {
        "model": "model",
        "events": [
            {
                "kind": "llm",
                "usage": {"prompt_tokens": 100},
                "request": {"messages": [{"content": "short"}]},
            },
            {
                "kind": "llm",
                "usage": {"prompt_tokens": 200},
                "request": {
                    "messages": [{"content": "forced"}],
                    "tools": [{"type": "function"}],
                },
            },
            {
                "kind": "llm",
                "usage": {"prompt_tokens": 150},
                "request": {
                    "messages": [{"content": "long"}],
                    "max_tokens": 99,
                },
            },
        ],
    }

    trace = build_revisit_trace(source, pressure_requests=2)

    assert trace["metadata"]["request_order"] == ["A", "B", "C", "A"]
    assert trace["metadata"]["template_prompt_tokens"] == 150
    assert [event["request"]["max_tokens"] for event in trace["events"]] == [
        1,
        1,
        1,
        1,
    ]
    first = trace["events"][0]["request"]["messages"][0]["content"]
    last = trace["events"][-1]["request"]["messages"][0]["content"]
    assert first == last
    assert first.startswith("[OFFLOAD_PROBE_NAMESPACE:A]")


@pytest.fixture
def tiering_probe():
    return runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/benchmark_agent_kv_tiering.py")
    )


def test_tiering_legacy_transfer_counters(tiering_probe):
    parse = tiering_probe["metrics"]
    legacy = (
        'vllm:kv_offload_total_bytes_total{transfer_type="GPU_to_CPU"} 128\n'
        'vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"} 64\n'
    )
    assert parse(legacy) == {"store_bytes": 128, "load_bytes": 64}
    with pytest.raises(RuntimeError, match="ambiguous"):
        parse(legacy + "vllm:kv_offload_store_bytes_total 256\n")


def test_tiering_failed_save_preserves_last_complete_snapshot(
    tiering_probe, monkeypatch, tmp_path
):
    output = tmp_path / "result.json"
    previous = {"valid": False, "trials": [{"index": 0}]}
    tiering_probe["save_result"](output, previous)
    write = Path.write_text

    def interrupted(path, text, *args, **kwargs):
        write(path, text[:10], *args, **kwargs)
        raise OSError("interrupted write")

    monkeypatch.setattr(Path, "write_text", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        tiering_probe["save_result"](output, {"valid": True})
    assert json.loads(output.read_text()) == previous


def test_tiering_metrics_do_not_double_count_legacy_transfer_counters(tiering_probe):
    before = tiering_probe["metrics"](
        'vllm:kv_offload_store_bytes_total{engine="0"} 100\n'
        'vllm:kv_offload_total_bytes_total{transfer_type="GPU_to_CPU"} 100\n'
        'vllm:prompt_tokens_by_source_total{source="local_compute"} 80\n'
    )
    assert before == {"store_bytes": 100.0, "tokens_local_compute": 80.0}
    assert tiering_probe["delta"](before, before) == dict.fromkeys(before, 0.0)
    with pytest.raises(RuntimeError, match="counters reset"):
        tiering_probe["delta"](before, {"store_bytes": 0})


def test_tiering_probe_accepts_lazy_transfer_counters(
    tiering_probe, monkeypatch, tmp_path
):
    initial = dict.fromkeys(tiering_probe["TOKEN_COUNTERS"] | {"preemptions"}, 0)
    primed = initial | {"tokens_local_compute": 16, "store_bytes": 128}
    restored = primed | {"tokens_external_kv_transfer": 16, "load_bytes": 128}
    snapshots = iter([primed, primed, restored])
    backend = SimpleNamespace(
        metrics=lambda: initial,
        reset=lambda **kwargs: None,
        infer=lambda prompt: {"token_ids": [1, 2], "ttft_ms": 1},
        settled_metrics=lambda *args: next(snapshots),
    )
    run = tiering_probe["run"]
    monkeypatch.setitem(run.__globals__, "Backend", lambda *args: backend)
    monkeypatch.setitem(run.__globals__, "make_prompt", lambda *args: [0] * 16)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: None)
        ),
    )
    result = run(
        SimpleNamespace(
            tokenizer=tmp_path,
            base_url="http://unused",
            model="test",
            output_tokens=2,
            prompt_tokens=16,
            sessions=1,
            pressure_requests=0,
            seed=1,
            mode="offload",
            scenario="roundtrip",
            output=tmp_path / "roundtrip.json",
            trials=1,
            settle_seconds=0,
        )
    )
    assert result["valid"]
    assert result["summary"]["resume_load_bytes"] == 128


@pytest.mark.parametrize("value", ["NaN", "+Inf", "-1"])
def test_tiering_probe_rejects_invalid_counters(tiering_probe, value):
    with pytest.raises(RuntimeError, match="invalid server counter"):
        tiering_probe["metrics"](f"vllm:kv_offload_store_bytes_total {value}\n")


def test_tiering_measurement_waits_for_request_accounting(tiering_probe, monkeypatch):
    before = dict.fromkeys(tiering_probe["TOKEN_COUNTERS"], 0.0)
    completed = before | {"tokens_local_compute": 16.0, "store_bytes": 100.0}
    samples = iter([before, completed, completed, completed])
    backend = tiering_probe["Backend"]("http://unused", "model", 2)
    monkeypatch.setattr(backend, "metrics", lambda: next(samples))
    now = [0.0]
    monkeypatch.setattr(tiering_probe["time"], "monotonic", lambda: now[0])
    monkeypatch.setattr(
        tiering_probe["time"], "sleep", lambda delay: now.__setitem__(0, now[0] + delay)
    )
    assert backend.settled_metrics(before, 16, 0.3) == completed
    assert now[0] == pytest.approx(0.6)


@pytest.mark.parametrize("failure", ["missing", "extra", "stale"])
def test_tiering_measurement_rejects_unaccounted_requests(
    tiering_probe, monkeypatch, failure
):
    before = dict.fromkeys(tiering_probe["TOKEN_COUNTERS"], 0.0)
    current = before | {"tokens_local_compute": 17.0 if failure == "extra" else 0.0}
    if failure == "missing":
        current.pop("tokens_local_cache_hit")
    backend = tiering_probe["Backend"]("http://unused", "model", 2)
    monkeypatch.setattr(backend, "metrics", lambda: current)
    now = [0.0]
    monkeypatch.setattr(tiering_probe["time"], "monotonic", lambda: now[0])
    monkeypatch.setattr(
        tiering_probe["time"], "sleep", lambda delay: now.__setitem__(0, now[0] + delay)
    )
    with pytest.raises(RuntimeError, match="missing|unexpected|did not settle"):
        backend.settled_metrics(before, 16, 0.3)


@pytest.mark.parametrize("limit", [None, 0, 16, 8192])
def test_tiering_stream_times_generated_tokens_and_preserves_hint(
    tiering_probe, monkeypatch, limit
):
    chunks = [
        {"choices": [{"text": "", "token_ids": [], "prompt_token_ids": [1, 2]}]},
        {"choices": [{"text": "ok", "token_ids": [11, 12]}]},
        {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 2}},
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    backend = tiering_probe["Backend"]("http://unused", "model", 2)
    sent = []

    def response(path, payload):
        sent.append(payload)
        return io.BytesIO((body + "data: [DONE]\n\n").encode())

    monkeypatch.setattr(backend, "_request", response)
    ticks = iter([10.0, 11.0, 12.0])
    monkeypatch.setattr(tiering_probe["time"], "perf_counter", lambda: next(ticks))
    result = backend.infer([1, 2], max_offload_tokens=limit)
    assert result["ttft_ms"] == 1000
    assert result["latency_ms"] == 2000
    assert result["token_ids"] == [11, 12]
    assert sent[0].get("kv_transfer_params") == (
        {"max_offload_tokens": limit} if limit is not None else None
    )


@pytest.mark.parametrize(
    "change,mode,scenario,error",
    [
        ({"prime_counters": {"store_bytes": 0}}, "offload", "roundtrip", "store KV"),
        ({"resume_counters": {"load_bytes": 0}}, "offload", "roundtrip", "restore"),
        (
            {"resume_counters": {"load_bytes": 32, "tokens_local_cache_hit": 16}},
            "offload",
            "roundtrip",
            "local KV",
        ),
        ({"resume": [{"token_ids": [3]}]}, "offload", "roundtrip", "mismatch"),
        (
            {"pressure_counters": {"store_bytes": 16}},
            "selective",
            "pressure",
            "terminal",
        ),
        ({}, "apc", "roundtrip", "unexpectedly"),
    ],
)
def test_tiering_probe_rejects_false_success(
    tiering_probe, change, mode, scenario, error
):
    trial = {
        "prime": [{"token_ids": [1, 2]}],
        "resume": [{"token_ids": [1, 2]}],
        "prime_counters": {"store_bytes": 32},
        "pressure_counters": {"store_bytes": 0},
        "resume_counters": {"load_bytes": 32, "tokens_local_cache_hit": 0},
    }
    tiering_probe["validate_trial"](trial, mode="offload", scenario="roundtrip")
    changed = copy.deepcopy(trial) | change
    with pytest.raises(RuntimeError, match=error):
        tiering_probe["validate_trial"](changed, mode=mode, scenario=scenario)


@pytest.mark.parametrize(
    "change,error",
    [
        ({"store_bytes": 1}, "new backup"),
        ({"load_bytes": 0}, "both transfer"),
        ({"tokens_external_kv_transfer": 16}, "restore boundary"),
        ({"tokens_local_cache_hit": 28}, "local cache"),
        ({"tokens_local_compute": 2}, "restore boundary"),
    ],
)
def test_boundary_probe_requires_exact_restore_without_new_backup(
    monkeypatch, change, error
):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    probe = runpy.run_path(str(scripts / "check_agent_kv_boundaries.py"))
    cold = {"token_ids": [1, 2]}
    case = {
        "expected_external_tokens": 28,
        "resume": cold,
        "store_counters": {"store_bytes": 32},
        "resume_counters": {
            "tokens_local_compute": 1,
            "tokens_external_kv_transfer": 28,
            "tokens_local_cache_hit": 0,
            "store_bytes": 0,
            "load_bytes": 32,
        },
    }
    probe["validate_case"](case, cold, 29)
    case["resume_counters"].update(change)
    with pytest.raises(RuntimeError, match=error):
        probe["validate_case"](case, cold, 29)
