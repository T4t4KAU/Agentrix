import copy
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "dp_probe", scripts / "benchmark_dp_kv_lifecycle.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_control_operations_bypass_router_and_inference_keeps_identity(
    probe, monkeypatch
):
    requests = []
    monkeypatch.setattr(
        probe.urllib.request, "urlopen", lambda request, **kw: requests.append(request)
    )
    backend = probe.RoutedBackend("http://router", "http://backend", "model", 8)
    backend.session_id = "session-a"
    backend._request(
        "/v1/completions", {"kv_transfer_params": {"max_offload_tokens": 0}}
    )
    backend._request("/reset_prefix_cache?reset_external=true", {})
    backend._request("/metrics")
    assert [r.full_url for r in requests] == [
        "http://router/v1/completions",
        "http://backend/reset_prefix_cache?reset_external=true",
        "http://backend/metrics",
    ]
    assert requests[0].get_header("X-session-id") == "session-a"
    assert requests[1].get_header("X-session-id") is None
    assert json.loads(requests[0].data)["kv_transfer_params"]["max_offload_tokens"] == 0
    backend.session_id = None
    with pytest.raises(ValueError, match="stable session"):
        backend._request("/v1/completions", {})


@pytest.mark.parametrize("ranks", [1, 2, 3])
def test_metrics_require_exactly_two_ranks_and_preserve_per_rank_sources(
    probe, monkeypatch, ranks
):
    lines = []
    for rank in range(ranks):
        lines.append(f'vllm:num_preemptions_total{{engine="{rank}"}} 0')
        for source in ("local_compute", "local_cache_hit", "external_kv_transfer"):
            lines.append(
                f'vllm:prompt_tokens_by_source_total{{engine="{rank}",source="{source}"}} {rank + 1}'
            )
        lines.append(
            f'vllm:kv_offload_total_bytes_total{{engine="{rank}",transfer_type="CPU_to_GPU"}} {10 * (rank + 1)}'
        )
    backend = probe.RoutedBackend("http://router", "http://backend", "model", 8)
    monkeypatch.setattr(
        backend, "_request", lambda *a: io.BytesIO("\n".join(lines).encode())
    )
    if ranks != 2:
        with pytest.raises(RuntimeError, match="two observed"):
            backend.metrics()
    else:
        values = backend.metrics()
        assert values["tokens_local_compute"] == 3
        assert values["rank/0/tokens_local_compute"] == 1
        assert values["rank/1/tokens_local_compute"] == 2
        assert values["load_bytes"] == 30
        assert values["rank/1/load_bytes"] == 20


@pytest.mark.parametrize(
    "corruption", [None, "tokens", "plan", "incomplete", "no_restore"]
)
def test_report_rejects_invalid_comparisons(probe, tmp_path, corruption):
    from summarize_dp_kv_lifecycle import summarize

    cases = [
        {
            "label": "baseline",
            "modes": ["apc"],
            "argv": ["--kv-cache-memory-bytes", "1024"],
        },
        {
            "label": "candidate",
            "modes": ["selective"],
            "argv": [
                "--kv-cache-memory-bytes",
                "512",
                "--kv-transfer-config",
                json.dumps({"kv_connector_extra_config": {"cpu_bytes_to_use": 1024}}),
            ],
        },
    ]
    config = {"cases": cases, "seeds": [1], "trials": 1}
    names = ["baseline-apc-1", "candidate-selective-1"]
    progress = {"phase": "complete", "completed": names}
    result = {
        "valid": True,
        "prompt_plan_sha256": "same-input",
        "configuration": {
            "sessions": 1,
            "pressure_requests": 1,
            "prompt_tokens": 16,
            "output_tokens": 1,
            "tool_gap_seconds": 0,
            "settle_seconds": 1,
        },
        "trials": [
            {
                "prime": [{"session": 0, "token_ids": [7]}],
                "pressure": [{"session": 1, "token_ids": [8]}],
                "resume": [{"session": 0, "token_ids": [7], "ttft_ms": 10}],
                "elapsed_seconds": 1,
                "pressure_counters": {"store_bytes": 0},
                "resume_counters": {"load_bytes": 10, "tokens_local_compute": 1},
            }
        ],
    }
    candidate = copy.deepcopy(result)
    if corruption == "tokens":
        candidate["trials"][0]["pressure"][0]["token_ids"] = [9]
    elif corruption == "plan":
        candidate["prompt_plan_sha256"] = "different-input"
    elif corruption == "incomplete":
        progress["phase"] = "benchmark"
    elif corruption == "no_restore":
        candidate["trials"][0]["resume_counters"]["load_bytes"] = 0
    for filename, data in (
        ("launch-private.json", config),
        ("progress.json", progress),
        (names[0] + ".json", result),
        (names[1] + ".json", candidate),
    ):
        (tmp_path / filename).write_text(json.dumps(data))
    if corruption:
        with pytest.raises(ValueError):
            summarize(tmp_path)
    else:
        report = summarize(tmp_path)
        assert report["valid"]
        assert report["cross_run_output_comparisons"] == 3
        assert report["rows"][1]["kv_bytes_per_rank"] == 512


def test_board_samples_exclude_startup_and_process_memory(probe, tmp_path):
    from summarize_dp_kv_lifecycle import sampled_hbm

    states = [
        {"time": 1, "phase": "server_startup", "case": "small"},
        {"time": 3, "phase": "benchmark", "cell": "small-selective-1"},
        {"time": 6, "phase": "complete"},
    ]
    (tmp_path / "controller.log").write_text("\n".join(json.dumps(v) for v in states))
    body = (
        "| 0 | 0000:01:00.0 | 0 | 0 / 0 100 / 65536 |\n"
        "| 0 | 0000:02:00.0 | 0 | 0 / 0 101 / 65536 |\n"
        "| 4 0 | 12345 | worker | 99999 | 12345 |\n"
    )
    samples = [
        {"time": t, "returncode": 0, "stdout": body, "controller_descendants": [12345]}
        for t in (2, 4, 5, 7)
    ]
    (tmp_path / "npu-samples.jsonl").write_text(
        "\n".join(json.dumps(v) for v in samples)
    )
    assert sampled_hbm(tmp_path, ["small"]) == {
        "small": {"samples": 2, "observed_total_hbm_peak_mib": 201}
    }
    # A previously observed worker can lose its parent while being stopped.
    samples[2]["controller_descendants"] = []
    (tmp_path / "npu-samples.jsonl").write_text(
        "\n".join(json.dumps(v) for v in samples)
    )
    assert sampled_hbm(tmp_path, ["small"])["small"]["samples"] == 2
    samples[1]["controller_descendants"] = []
    samples[1]["stdout"] = body.replace("12345", "54321")
    (tmp_path / "npu-samples.jsonl").write_text(
        "\n".join(json.dumps(v) for v in samples)
    )
    with pytest.raises(ValueError, match="outside this experiment"):
        sampled_hbm(tmp_path, ["small"])


def test_apc_trial_does_not_require_an_external_cache(probe, monkeypatch, tmp_path):
    class DeviceOnlyBackend:
        def __init__(self, *args):
            self.tokens = 0

        def reset(self, *, external):
            if external:
                raise RuntimeError("no external connector installed")

        def metrics(self):
            return {
                "preemptions": 0,
                "tokens_local_compute": self.tokens,
                "tokens_local_cache_hit": 0,
                "tokens_external_kv_transfer": 0,
            }

        def infer(self, prompt, **kwargs):
            self.tokens += len(prompt)
            return {"token_ids": [7], "ttft_ms": 1}

        def settled_metrics(self, *args):
            return self.metrics()

    monkeypatch.setattr(probe, "RoutedBackend", DeviceOnlyBackend)
    monkeypatch.setattr(probe, "make_prompt", lambda *args: [1, 2])
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: None)
        ),
    )
    args = SimpleNamespace(
        output=tmp_path / "apc.json",
        tokenizer=tmp_path,
        seed=1,
        sessions=1,
        pressure_requests=1,
        prompt_tokens=2,
        base_url="http://router",
        control_url="http://backend",
        model="model",
        output_tokens=1,
        trials=1,
        settle_seconds=0,
        tool_gap_seconds=0,
        mode="apc",
    )
    assert probe.run(args)["valid"]
