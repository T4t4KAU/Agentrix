"""AgentX capacity observations and experimental policy comparisons on the server.

Policy sweeps explicitly opt in to default-off candidates. A successful run means
the measurement passed validation, not that the candidate improved performance.
Adaptive-prefill runs shorter than 900 seconds are non-submission screening only.
"""

import argparse
import fcntl
import hashlib
import itertools
import json
import os
import re
import shlex
import signal
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_last_sample = 0.0
_failures = 0
_stored = {}
_evicted = {}
_lookup_seen = {}


def install_agent_hints():
    """Forward existing runtime tree IDs, without changing replay or prompts."""
    from aiperf.endpoints.openai_chat import ChatEndpoint

    original = ChatEndpoint.format_payload
    if getattr(original, "_agentrix_session_hints", False):
        return

    def format_payload(self, request_info):
        payload = original(self, request_info)
        sid = request_info.x_correlation_id
        root = request_info.root_correlation_id or sid
        parent = request_info.parent_correlation_id
        if not sid or not root or (parent and not request_info.root_correlation_id):
            raise ValueError("AgentX request is missing its runtime session tree")
        payload["session_id"] = sid
        params = dict(payload.get("kv_transfer_params") or {})
        params["agentrix_session"] = {"root_session_id": root}
        if parent:
            params["agentrix_session"]["parent_session_id"] = parent
        payload["kv_transfer_params"] = params
        audit = Path(os.environ["AGENTRIX_SESSION_HINT_AUDIT"])
        with (audit / f"hints-{os.getpid()}.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "request_id": request_info.x_request_id,
                        "session_id": sid,
                        "root_session_id": root,
                        "parent_session_id": parent,
                    }
                )
                + "\n"
            )
        return payload

    format_payload._agentrix_session_hints = True
    ChatEndpoint.format_payload = format_payload


def validate_agent_hints(part):
    sent = {}
    for path in (part / "hints").glob("hints-*.jsonl"):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            sent[record["request_id"]] = record
    checked = 0
    children = 0
    for line in (part / "bench/profile_export.jsonl").read_text().splitlines():
        meta = json.loads(line)["metadata"]
        expected = meta.get("root_correlation_id") or meta["x_correlation_id"]
        actual = sent.get(meta["x_request_id"])
        if (
            actual is None
            or actual["session_id"] != meta["x_correlation_id"]
            or actual["root_session_id"] != expected
        ):
            raise RuntimeError("AgentX session hints do not match replay metadata")
        checked += 1
        children += meta.get("agent_depth", 0) > 0
    if not checked or not children:
        raise RuntimeError("Session comparison did not exercise Agent branches")
    (part / "hints-validated.json").write_text(
        json.dumps({"requests": checked, "child_requests": children}) + "\n"
    )


def install_policy_source(staged, root):
    """Install reviewed Python files only after the preceding run releases its lock."""
    manifest = json.loads((staged / "manifest.json").read_text())
    replacements = []
    for name, hashes in manifest["files"].items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or rel.parts[:2] != ("vllm", "vllm"):
            raise ValueError("Policy stage may only replace vLLM Python modules")
        target = root / rel
        data = (staged / "files" / rel).read_bytes()
        if hashlib.sha256(data).hexdigest() != hashes["after"]:
            raise RuntimeError(f"Staged source changed: {name}")
        current = hashlib.sha256(target.read_bytes()).hexdigest()
        if current not in (hashes["before"], hashes["after"]):
            raise RuntimeError(f"Server source diverged: {name}")
        compile(data, str(target), "exec")
        replacements.append((target, rel, data, current == hashes["after"]))
    for target, rel, data, installed in replacements:
        if installed:
            continue
        backup = staged / "original" / rel
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_bytes(target.read_bytes())
        temporary = target.with_suffix(".py.agentrix-stage")
        temporary.write_bytes(data)
        temporary.replace(target)
    return {name: entry["after"] for name, entry in manifest["files"].items()}


def cache_history(manager, request, cached):
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    pool = manager.block_pool
    group = manager.coordinator.single_type_managers[0]
    assert group.block_size == pool.hash_block_size
    hashes = request.block_hashes
    limit = min(len(hashes), (request.num_tokens - 1) // group.block_size)
    index = cached // group.block_size
    historical = 0
    for block_hash in itertools.islice(hashes, limit):
        if make_block_hash_with_group_id(block_hash, 0) not in _stored:
            break
        historical += group.block_size
    result = {"historical_prefix_tokens": historical}
    if index < limit:
        key = make_block_hash_with_group_id(hashes[index], 0)
        result.update(
            first_missing_hash=key.hex(),
            first_missing_last_stored_ns=_stored.get(key),
            first_missing_last_evicted_ns=_evicted.get(key),
            first_missing_present=pool.cached_block_hash_to_block.get_one_block(key)
            is not None,
        )
    return result


def install_history_observer():
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    insert = BlockPool._insert_block_hash
    evict = BlockPool._maybe_evict_cached_block
    lookup = KVCacheManager.get_computed_blocks

    def observed_insert(self, block_hash_with_group_id, block, num_tokens):
        result = insert(self, block_hash_with_group_id, block, num_tokens)
        _stored[block_hash_with_group_id] = time.time_ns()
        return result

    def observed_evict(self, block):
        keys = list(self.cached_block_hashes_by_block.get(block.block_id, ()))
        if block.block_hash is not None:
            keys.append(block.block_hash)
        result = evict(self, block)
        now = time.time_ns()
        for key in keys:
            if self.cached_block_hash_to_block.get_one_block(key) is None:
                _evicted[key] = now
        return result

    def observed_lookup(self, request):
        result = lookup(self, request)
        cached = result[1]
        if _lookup_seen.get(request.request_id) != cached:
            _lookup_seen[request.request_id] = cached
            record = {
                "time_ns": time.time_ns(),
                "id": request.request_id,
                "tokens": request.num_tokens,
                "cached": cached,
                **cache_history(self, request, cached),
            }
            path = Path(os.environ["AGENTRIX_CAPACITY_PROBE"]).with_name(
                "cache-history.jsonl"
            )
            with path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        return result

    BlockPool._insert_block_hash = observed_insert
    BlockPool._maybe_evict_cached_block = observed_evict
    KVCacheManager.get_computed_blocks = observed_lookup


def observe(scheduler, head, queue, budget):
    global _last_sample, _failures
    _failures += 1
    now = time.monotonic()
    if now - _last_sample < 0.25:
        return
    _last_sample = now
    start = time.perf_counter()
    manager = scheduler.kv_cache_manager
    coordinator = manager.coordinator
    assert scheduler.connector is None and scheduler.lora_config is None
    assert scheduler.num_lookahead_tokens == 0
    assert scheduler.scheduler_reserve_full_isl
    assert len(coordinator.single_type_managers) == 1
    assert type(coordinator.single_type_managers[0]).__name__ == "FullAttentionManager"
    free = manager.block_pool.get_num_free_blocks()
    candidates = []
    for request in itertools.islice(queue, 1, 9):
        item = {"id": request.request_id, "tokens": request.num_tokens}
        candidates.append(item)
        if (
            request.status.name != "WAITING"
            or request.num_computed_tokens != 0
            or request.has_encoder_inputs
            or request.num_stale_output_tokens
            or not manager.prefix_cache_lookup_enabled(request)
        ):
            item["skip"] = "outside_probe_scope"
            continue
        # The full-attention lookup and capacity estimator do not touch blocks,
        # change refcounts, reorder the free queue, or allocate KV slots.
        blocks, cached, _ = coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1
        )
        required = coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=min(request.num_tokens, manager.max_model_len),
            new_computed_blocks=blocks,
            num_encoder_tokens=0,
            total_computed_tokens=cached,
            num_local_computed_tokens=cached,
            num_tokens_main_model=min(request.num_tokens, manager.max_model_len),
            apply_admission_cap=True,
        )
        required += manager.watermark_blocks if scheduler.running else 0
        item.update(
            cached=cached,
            uncached=request.num_tokens - cached,
            required_blocks=required,
            fits=required <= free and budget > 0,
            **cache_history(manager, request, cached),
        )
    assert free == manager.block_pool.get_num_free_blocks()
    record = {
        "time_ns": time.time_ns(),
        "step": scheduler.current_step,
        "blocked_calls": _failures,
        "head": head.request_id,
        "head_tokens": head.num_tokens,
        "free_blocks": free,
        "running": len(scheduler.running),
        "queue_length": len(queue),
        "other_queue_length": len(scheduler.skipped_waiting)
        if queue is scheduler.waiting
        else len(scheduler.waiting),
        "token_budget": budget,
        "candidates": candidates,
        "probe_ms": (time.perf_counter() - start) * 1000,
    }
    with open(os.environ["AGENTRIX_CAPACITY_PROBE"], "a") as stream:
        stream.write(json.dumps(record) + "\n")


def stop_server(pid):
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.exists():
        return
    command = cmdline.read_bytes()
    if b"vllm" not in command or b"18000" not in command:
        raise RuntimeError("PID is not the owned benchmark server")
    os.kill(pid, signal.SIGTERM)
    for _ in range(90):
        stat = Path(f"/proc/{pid}/stat")
        if not stat.exists() or stat.read_text().split(") ", 1)[1].startswith("Z"):
            return
        time.sleep(1)
    raise RuntimeError("Benchmark server did not stop")


def cleanup_offload_cache(part):
    """Remove only this stopped server's unreferenced CPU offload mmap files."""
    log = part / "server.log"
    if not log.exists():
        return
    content = log.read_text(errors="replace")
    paths = {
        Path(name)
        for name in re.findall(
            r"Created mmap file (/dev/shm/vllm_offload_[0-9a-f-]+\.mmap)",
            content,
        )
        if Path(name).exists()
    }
    if not paths:
        return
    pid_file = part / "server.pid"
    pids = set(re.findall(r"\((?:EngineCore\S*|Worker\S*) pid=(\d+)\)", content))
    if pid_file.exists():
        pids.add(pid_file.read_text().strip())
    for pid in pids:
        stat = Path("/proc") / pid / "stat"
        if stat.exists() and not stat.read_text().split(") ", 1)[1].startswith("Z"):
            raise RuntimeError("Cannot remove offload cache of a running server")
    targets = {str(path) for path in paths}
    for proc in Path("/proc").iterdir():
        try:
            if not proc.name.isdigit() or proc.stat().st_uid != os.getuid():
                continue
            maps = (proc / "maps").read_text()
            if any(name in maps for name in targets):
                raise RuntimeError(f"Offload cache is still in use by PID {proc.name}")
            for fd in (proc / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue
                if target in targets:
                    raise RuntimeError(
                        f"Offload cache is still in use by PID {proc.name}"
                    )
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    for path in paths:
        if path.is_symlink() or path.stat().st_uid != os.getuid():
            raise RuntimeError(f"Unexpected offload cache owner: {path}")
        path.unlink()
        print("REMOVED stopped server cache", path, flush=True)


def run():
    root = Path("/mnt/sda1/hwx/Agentrix")
    baseline = root / "experiments/agentx-qwen3-8b-flash-attn"
    run_dir = baseline / ("capacity-probe-" + time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir()
    print("RUN", run_dir, flush=True)
    (baseline / "capacity-probe-latest").write_text(str(run_dir))
    source = root / "vllm/vllm/v1/core/sched/scheduler.py"
    original = source.read_text()
    anchor = "                if new_blocks is None:\n                    # The request cannot be scheduled."
    assert original.count(anchor) == 1
    hook = (
        "                if new_blocks is None:\n"
        "                    _capacity_observe(self, request, request_queue, "
        "min(token_budget, input_budget - draft_slots))\n"
        "                    # The request cannot be scheduled."
    )
    patched = original.replace(anchor, hook)
    patched += (
        "\nfrom runpy import run_path as _capacity_run_path\n"
        f"_capacity_probe = _capacity_run_path({str(Path(__file__).resolve())!r})\n"
        "_capacity_observe = _capacity_probe['observe']\n"
        "_capacity_probe['install_history_observer']()\n"
    )
    compile(patched, str(source), "exec")
    (run_dir / "scheduler.py.original").write_text(original)
    server = None
    try:
        (run_dir / "status").write_text("starting_server")
        stop_server(int((baseline / "server.pid").read_text()))
        source.write_text(patched)
        serve = (
            (baseline / "serve.sh")
            .read_text()
            .replace(
                "RUN=$ROOT/experiments/agentx-qwen3-8b-flash-attn",
                f"RUN={run_dir}",
            )
        )
        env = dict(os.environ, AGENTRIX_CAPACITY_PROBE=str(run_dir / "samples.jsonl"))
        with (run_dir / "server.log").open("w") as output:
            server = subprocess.Popen(
                ["bash", "-c", serve],
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        for _ in range(300):
            if server.poll() is not None:
                raise RuntimeError(f"Server exited: {server.returncode}")
            try:
                urllib.request.urlopen(
                    "http://127.0.0.1:18000/health", timeout=2
                ).close()
                break
            except OSError:
                time.sleep(2)
        else:
            raise RuntimeError("Server readiness timed out")
        (run_dir / "status").write_text("warmup_and_profiling")
        command = next(
            line.strip()
            for line in (baseline / "run-matrix.sh").read_text().splitlines()
            if line.strip().startswith(".venv/bin/aiperf profile ")
        ).split(' > "c${concurrency}.log"')[0]
        command = command.replace('"$concurrency"', "8")
        command = command.replace(
            "--benchmark-duration 3600", "--benchmark-duration 900"
        )
        command = command.replace('"$RUN/c${concurrency}"', str(run_dir / "bench"))
        with (run_dir / "bench.log").open("w") as output:
            result = subprocess.run(
                ["bash", "-c", f"cd {baseline}; source ./bench-env.sh; exec {command}"],
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=3600,
                check=False,
            )
        (run_dir / "bench.exit").write_text(str(result.returncode))
        if result.returncode:
            raise RuntimeError(f"Benchmark exited: {result.returncode}")
        (run_dir / "status").write_text("success")
    except BaseException:
        (run_dir / "status").write_text("failed")
        raise
    finally:
        try:
            if server is not None and server.poll() is None:
                stop_server(server.pid)
                server.wait(timeout=10)
        finally:
            if source.read_text() == patched:
                source.write_text(original)
                (run_dir / "source-restored").write_text("yes")
            elif source.read_text() != original:
                raise RuntimeError(
                    "Scheduler changed during probe; refusing to overwrite"
                )
    print("DONE", run_dir, flush=True)


def check_offload_roundtrip(part, mixed=False, reset_after=False):
    def post(path, body):
        request = urllib.request.Request(
            "http://127.0.0.1:18000" + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)

    def metrics():
        with urllib.request.urlopen(
            "http://127.0.0.1:18000/metrics", timeout=10
        ) as response:
            return response.read().decode()

    def counter(text, name):
        return sum(
            float(line.rsplit(" ", 1)[1])
            for line in text.splitlines()
            if line.split("{", 1)[0].split(" ", 1)[0] in (name, name + "_total")
        )

    payload = {
        "model": "Qwen3-8B",
        "prompt": (
            "A database stores records in pages and uses a cache for reuse. " * 512
        )
        + "\nQuestion: What does a database store?\nAnswer:",
        "temperature": 0,
        "seed": 20260707,
        "max_tokens": 32,
        "ignore_eos": True,
        "session_id": "agentrix-offload-smoke",
        "kv_transfer_params": {
            "agentrix_session": {"root_session_id": "agentrix-offload-smoke"}
        },
    }
    initial = metrics()
    first = post("/v1/completions", payload)
    store_name = "vllm:kv_offload_store_bytes"
    load_name = "vllm:kv_offload_load_bytes"
    for _ in range(30):
        stored = metrics()
        if counter(stored, store_name) > counter(initial, store_name):
            break
        time.sleep(1)
    else:
        raise RuntimeError("CPU offload did not store KV data during smoke check")
    gpu_hit = post("/v1/completions", payload)
    same_gpu_output = first["choices"][0]["text"] == gpu_hit["choices"][0]["text"]
    gpu_cached_tokens = gpu_hit["usage"]["prompt_tokens_details"]["cached_tokens"]
    after_gpu_hit = metrics()
    gpu_load_bytes = counter(after_gpu_hit, load_name) - counter(stored, load_name)
    if not same_gpu_output or gpu_cached_tokens <= 0 or gpu_load_bytes != 0:
        raise RuntimeError("GPU prefix reuse changed output or missed the cache")
    for _ in range(30):
        if post("/reset_prefix_cache?reset_external=false", {})["success"]:
            break
        time.sleep(1)
    else:
        raise RuntimeError("GPU prefix cache could not be cleared for smoke check")
    second = post("/v1/completions", payload)
    for _ in range(30):
        restored = metrics()
        loaded_bytes = counter(restored, load_name) - counter(stored, load_name)
        if loaded_bytes > 0:
            break
        time.sleep(1)
    else:
        raise RuntimeError("CPU offload did not restore KV data after GPU cache reset")
    same = first["choices"][0]["text"] == second["choices"][0]["text"]
    mixed_result = None
    if mixed:
        cold_payload = dict(
            payload,
            prompt="A separate cold session.\n" + payload["prompt"] * 8,
            max_tokens=128,
            session_id="agentrix-smoke-cold",
            kv_transfer_params={
                "agentrix_session": {"root_session_id": "agentrix-smoke-cold"}
            },
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            cold = pool.submit(post, "/v1/completions", cold_payload)
            # Let the cold prefill enter the engine before its cached neighbor.
            time.sleep(0.05)
            warm = pool.submit(post, "/v1/completions", payload).result(timeout=120)
            cold_result = cold.result(timeout=120)
        mixed_result = {
            "same_output": first["choices"][0]["text"] == warm["choices"][0]["text"],
            "warm": warm,
            "cold": cold_result,
        }
    (part / "roundtrip.json").write_text(
        json.dumps(
            {
                "first": first,
                "restored": second,
                "same_output": same,
                "same_gpu_output": same_gpu_output,
                "gpu_cached_tokens": gpu_cached_tokens,
                "gpu_hit_cpu_load_bytes": gpu_load_bytes,
                "loaded_bytes": loaded_bytes,
                "stored_bytes": counter(stored, store_name)
                - counter(initial, store_name),
                "mixed": mixed_result,
            },
            indent=2,
        )
        + "\n"
    )
    (part / "metrics-before.prom").write_text(initial)
    (part / "metrics-after.prom").write_text(restored)
    if not same:
        raise RuntimeError("Output changed after CPU KV cache roundtrip")
    if mixed_result is not None and (
        not mixed_result["same_output"]
        or mixed_result["warm"]["usage"]["prompt_tokens_details"]["cached_tokens"] <= 0
    ):
        raise RuntimeError("Mixed cold/warm smoke changed output or missed the cache")
    if reset_after:
        for _ in range(30):
            if post("/reset_prefix_cache?reset_external=true", {})["success"]:
                break
            time.sleep(1)
        else:
            raise RuntimeError("Could not clear GPU/CPU caches after smoke check")
        path = part / "roundtrip.json"
        result = json.loads(path.read_text())
        result["cache_reset_after_smoke"] = True
        path.write_text(json.dumps(result, indent=2) + "\n")


def validate_comparison_result(part, screening=False):
    profile = json.loads((part / "bench/profile_export_aiperf.json").read_text())
    if (
        (part / "bench.exit").read_text().strip() != "0"
        or profile["error_summary"]
        or profile["was_cancelled"]
    ):
        raise RuntimeError(f"{part.name} benchmark has errors or was cancelled")
    metadata = profile["metadata"]
    if screening:
        # AgentX requires >=900 seconds for submissions. Short screening keeps
        # the explicit invalid-submission stamp; only the duration lock may differ.
        logs = (part / "bench.log").read_text() + (
            part / "bench/logs/aiperf.log"
        ).read_text()
        violations = set(
            re.findall(r"Scenario violation \(override active\): ([^:]+):", logs)
        )
        if (
            metadata.get("submission_valid") is not False
            or set(metadata.get("submission_invalid_reasons", []))
            != {"unsafe_override"}
            or violations != {"--benchmark-duration"}
        ):
            raise RuntimeError(f"{part.name} screening violated more than duration")
    elif not metadata.get("submission_valid"):
        raise RuntimeError(f"{part.name} benchmark failed scenario validation")
    return profile


def finish_comparison_after(previous, arm):
    """Keep completed arms and retire later arms of an owned comparison."""
    baseline = previous.parent
    match = re.fullmatch(r"(prefill-sweep|offload-sweep)-\d{8}-\d{6}", previous.name)
    if match is None:
        raise ValueError("Only prefill and offload comparisons can be shortened")
    mode = match[1]
    if Path((baseline / f"{mode}-latest").read_text().strip()) != previous:
        raise RuntimeError("The selected comparison is no longer current")
    manifest_path = previous / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    order = manifest["order"]
    if arm not in order:
        raise ValueError("Requested finishing arm is absent from the run")
    selected = order[: order.index(arm) + 1]
    print("WAIT selected arm", previous / arm, flush=True)
    while True:
        status = previous / arm / "status"
        if status.exists() and status.read_text().strip() == "success":
            break
        if (previous / "status").read_text().strip() in ("failed", "success"):
            raise RuntimeError("Preceding run ended before the selected arm succeeded")
        time.sleep(0.25)
    for name in selected:
        part = previous / name
        if name in manifest.get("smoke_arms", []):
            smoke = json.loads((part / "roundtrip.json").read_text())
            if (
                (part / "status").read_text().strip() != "success"
                or not smoke["same_output"]
                or smoke["loaded_bytes"] <= 0
            ):
                raise RuntimeError(f"Cannot finish run: {name} smoke check failed")
            continue
        profile = json.loads((part / "bench/profile_export_aiperf.json").read_text())
        if (
            (part / "bench.exit").read_text().strip() != "0"
            or profile["error_summary"]
            or profile["was_cancelled"]
            or not profile["metadata"].get("submission_valid")
        ):
            raise RuntimeError(f"Cannot finish run: {name} has invalid results")

    controller_pid = previous / "controller.pid"
    if not controller_pid.exists():
        controller_pid = baseline / f"{mode}-controller.pid"
    pid = int(controller_pid.read_text())
    command = Path(f"/proc/{pid}/cmdline")
    if command.exists() and command.read_bytes():
        args = command.read_bytes().split(b"\0")
        if ("--" + mode).encode() not in args or not any(
            arg.endswith(b"/probe_capacity.py") for arg in args
        ):
            raise RuntimeError("PID is not the preceding comparison controller")
        os.kill(pid, signal.SIGTERM)
        for _ in range(180):
            stat = Path(f"/proc/{pid}/stat")
            if not stat.exists() or stat.read_text().split(") ", 1)[1].startswith("Z"):
                break
            time.sleep(1)
        else:
            raise RuntimeError("Preceding comparison controller did not stop")
    for name in order:
        pid_file = previous / name / "server.pid"
        if pid_file.exists():
            server_pid = int(pid_file.read_text())
            stat = Path(f"/proc/{server_pid}/stat")
            if stat.exists() and not stat.read_text().split(") ", 1)[1].startswith("Z"):
                stop_server(server_pid)
    skipped = order[len(selected) :]
    for name in skipped:
        part = previous / name
        part.mkdir(exist_ok=True)
        (part / "status").write_text("skipped_by_user")
    manifest["original_order"] = order
    manifest["order"] = selected
    manifest["skipped_by_user"] = skipped
    manifest["finish_after_arm"] = arm
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (previous / "status").write_text("success")
    print("FINISHED selected arms", selected, "SKIPPED", skipped, flush=True)


def verify_upstream_runtime(runtime):
    """Check the separately installed official wheel before each benchmark arm."""
    manifest = json.loads((runtime / "release-manifest.json").read_text())
    if (
        manifest["version"] != "0.28.0+cu129"
        or manifest["upstream_commit"] != "2cf0a6915ce544dc493a0990f2ea38d81601128a"
        or manifest["wheel_sha256"]
        != "8ec943b66a0c6b4351d0778e99d7bacfca5788dd8eedd49425092bacb61c4397"
    ):
        raise RuntimeError("Baseline must use the official v0.28.0 CUDA 12.9 wheel")
    for name, expected in manifest["file_sha256"].items():
        path = runtime / name
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"Official release file changed: {name}")
    return manifest


def run_comparison(
    duration,
    mode="capacity-ab",
    reverse=False,
    kv_cache_bytes=None,
    after_run=None,
    after_arm=None,
    policy_source=None,
    include_arc=False,
    upstream_env=None,
    resume_run=None,
    warmup_requests_per_lane=10,
):
    root = Path("/mnt/sda1/hwx/Agentrix")
    baseline = root / "experiments/agentx-qwen3-8b-flash-attn"
    source = root / "vllm/vllm/v1/core/sched/scheduler.py"
    upstream = verify_upstream_runtime(upstream_env) if upstream_env else None
    if upstream is not None:
        source = upstream_env / upstream["package_root"] / "v1/core/sched/scheduler.py"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    adapter_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    expected_kv_tokens = None
    if after_run is not None:
        previous = Path(after_run).resolve()
        if previous.parent != baseline or not (previous / "status").is_file():
            raise ValueError("--after-run must name an existing benchmark run")
        if after_arm is not None:
            finish_comparison_after(previous, after_arm)
        print("WAIT", previous, flush=True)
        while True:
            status = (previous / "status").read_text().strip()
            if status == "success":
                break
            if status == "failed":
                raise RuntimeError("Preceding benchmark failed; comparison not started")
            time.sleep(30)
        if upstream_env is not None:
            reference = json.loads((previous / "manifest.json").read_text())
            if (
                reference["kv_cache_memory_bytes"] != kv_cache_bytes
                or reference["duration_seconds"] != duration
                or reference["concurrency"] != 8
                or reference["seed"] != 20260707
            ):
                raise RuntimeError("Official baseline and reference settings differ")
            capacities = {
                int((previous / name / "kv-cache-tokens").read_text())
                for name in reference["order"]
            }
            if len(capacities) != 1:
                raise RuntimeError("Reference has inconsistent GPU KV cache capacity")
            expected_kv_tokens = capacities.pop()
    prefix = mode
    run_dir = (
        resume_run.resolve()
        if resume_run
        else baseline / (prefix + "-" + time.strftime("%Y%m%d-%H%M%S"))
    )
    if resume_run and (
        run_dir.parent != baseline
        or re.fullmatch(prefix + r"-\d{8}-\d{6}", run_dir.name) is None
        or not (run_dir / "manifest.json").is_file()
    ):
        raise ValueError("--resume-run must name an existing run of the same mode")
    arms = [("baseline", False, 0, 0), ("bypass", True, 0, 0)]
    if mode == "prefill-sweep":
        arms = [("baseline", False, 0, 0)] + [
            (f"cap{cap}", False, cap, 0) for cap in (512, 1024, 2048)
        ]
    elif mode == "offload-sweep":
        arms = [("smoke_cpu32", False, 0, 32), ("baseline", False, 0, 0)] + [
            (f"cpu{gib}", False, 0, gib) for gib in (32, 64)
        ]
    elif mode == "session-sweep":
        arms = [
            ("smoke_session", False, 0, 32),
            ("lru", False, 0, 32),
            ("session_lru", False, 0, 32),
        ]
        if include_arc:
            arms.insert(2, ("arc", False, 0, 32))
    elif mode == "gpu-session-sweep":
        arms = [
            ("smoke_gpu_session", False, 0, 32),
            ("lru", False, 0, 32),
            ("gpu_session", False, 0, 32),
        ]
    elif mode == "adaptive-prefill-sweep":
        arms = [
            ("lru", False, 0, 32),
            ("adaptive", False, 0, 32),
        ]
    elif mode == "upstream-sweep":
        arms = [
            ("smoke_upstream_cpu32", False, 0, 32),
            ("upstream_gpu", False, 0, 0),
            ("upstream_lru_cpu32", False, 0, 32),
        ]
    session_sweep = mode == "session-sweep"
    gpu_session_sweep = mode == "gpu-session-sweep"
    adaptive_sweep = mode == "adaptive-prefill-sweep"
    screening = adaptive_sweep and duration < 900
    agent_hints = (
        session_sweep or gpu_session_sweep or adaptive_sweep or mode == "upstream-sweep"
    )
    policies = {
        name: ("session_lru" if name == "smoke_session" else name)
        if session_sweep
        else "lru"
        for name, _, _, _ in arms
    }
    policy_config = {"retention_seconds": 120, "protected_fraction": 0.5}
    gpu_retention_config = {
        "protected_fraction": 0.25,
        "default_seconds": 15.0,
        "min_seconds": 2.0,
        "max_seconds": 30.0,
        "max_sessions": 256,
    }
    if reverse:
        arms = [arm for arm in arms if arm[0].startswith("smoke_")] + [
            arm for arm in reversed(arms) if not arm[0].startswith("smoke_")
        ]
    # Share the original matrix lock so two experiments cannot own GPU 1.
    with (baseline / "matrix.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if after_run else fcntl.LOCK_NB))
        run_dir.mkdir(exist_ok=bool(resume_run))
        if resume_run:
            pid = int((run_dir / "controller.pid").read_text())
            stat = Path(f"/proc/{pid}/stat")
            if stat.exists() and not stat.read_text().split(") ", 1)[1].startswith("Z"):
                raise RuntimeError("Comparison controller is still running")
        (run_dir / "controller.pid").write_text(str(os.getpid()))
        (baseline / (prefix + "-latest")).write_text(str(run_dir))
        policy_hashes = {}
        if policy_source is not None:
            stop_server(int((baseline / "server.pid").read_text()))
            try:
                policy_hashes = install_policy_source(policy_source, root)
                source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            except BaseException:
                (run_dir / "status").write_text("failed")
                raise
        manifest = {
            "duration_seconds": duration,
            "grace_seconds": 300,
            "concurrency": 8,
            "warmup_requests_per_lane": warmup_requests_per_lane,
            "inline_smoke": adaptive_sweep,
            "screening_only": screening,
            "seed": 20260707,
            "order": [name for name, _, _, _ in arms],
            "smoke_arms": [name for name, _, _, _ in arms if name.startswith("smoke_")],
            "scheduler_sha256": source_hash,
            "kv_cache_memory_bytes": kv_cache_bytes,
            "long_prefill_token_thresholds": {name: cap for name, _, cap, _ in arms},
            "capacity_bypass": {name: enabled for name, enabled, _, _ in arms},
            "cpu_offload_gib": {name: gib for name, _, _, gib in arms},
            "offload_policy": policies,
            "session_policy_config": policy_config if session_sweep else None,
            "gpu_session_retention": gpu_retention_config
            if gpu_session_sweep
            else None,
            "adaptive_prefill": {
                "mixed_prefill_tokens": 2048,
                "short_request_tokens": 1024,
                "max_full_prefix_probes": 2,
                "max_deferred_steps": 4,
                "lookahead": 8,
                "enabled": {name: name != "lru" for name, _, _, _ in arms},
            }
            if adaptive_sweep
            else None,
            "policy_source_sha256": policy_hashes,
            "agent_tree_hints": agent_hints,
            "upstream_release": {
                key: value for key, value in upstream.items() if key != "file_sha256"
            }
            if upstream is not None
            else None,
            "harness_adapter_sha256": adapter_hash,
            "offload_prompt_only": True,
            "offload_blocks_per_chunk": 1,
            "offload_store_threshold": 0,
            "after_run": str(after_run) if after_run else None,
            "after_arm": after_arm,
            "lookahead": 8,
            "max_overtakes_per_request": 4,
            "observation_hooks": False,
        }
        if resume_run:
            previous = json.loads((run_dir / "manifest.json").read_text())
            for key in manifest.keys() - {
                "harness_adapter_sha256",
                "after_run",
                "after_arm",
                "upstream_release",
            }:
                if previous.get(key) != manifest[key]:
                    raise RuntimeError(f"Cannot resume with changed settings: {key}")
            previous.setdefault("resume_history", []).append(
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "controller_sha256": adapter_hash,
                    "pid": os.getpid(),
                }
            )
            manifest = previous
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (run_dir / "status").write_text("resuming" if resume_run else "starting")
        print("RUN", run_dir, flush=True)
        serve_template = (baseline / "serve.sh").read_text()
        if upstream_env is not None:
            (run_dir / "release-manifest.json").write_text(
                json.dumps(upstream, indent=2) + "\n"
            )
            serve_template = (
                serve_template.replace(
                    "export PYTHONPATH=$ROOT/vllm",
                    "unset PYTHONPATH\nexport VLLM_CACHE_ROOT="
                    + shlex.quote(str(upstream_env / "cache")),
                )
                .replace('cd "$ROOT/vllm"', "cd " + shlex.quote(str(upstream_env)))
                .replace(
                    "exec .venv/bin/vllm serve ",
                    "exec " + shlex.quote(str(upstream_env / "bin/vllm")) + " serve ",
                )
                .replace("$ROOT/vllm/.venv/bin", str(upstream_env / "bin"))
            )
        matrix_template = (baseline / "run-matrix.sh").read_text()
        try:
            stop_server(int((baseline / "server.pid").read_text()))
            for name, enabled, cap, cpu_gib in arms:
                if (
                    upstream_env is not None
                    and verify_upstream_runtime(upstream_env) != upstream
                ):
                    raise RuntimeError("Official runtime changed between arms")
                if (
                    hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                    != adapter_hash
                ):
                    raise RuntimeError("Benchmark adapter changed during comparison")
                if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
                    raise RuntimeError("Scheduler source changed during comparison")
                for path, expected in policy_hashes.items():
                    if (
                        hashlib.sha256((root / path).read_bytes()).hexdigest()
                        != expected
                    ):
                        raise RuntimeError(f"Policy source changed: {path}")
                part = run_dir / name
                if resume_run and part.exists():
                    if (part / "status").read_text().strip() != "success":
                        raise RuntimeError(f"Cannot reuse incomplete arm: {name}")
                    if name.startswith("smoke_"):
                        smoke = json.loads((part / "roundtrip.json").read_text())
                        if not smoke["same_output"] or smoke["loaded_bytes"] <= 0:
                            raise RuntimeError("Cannot reuse failed smoke check")
                    else:
                        validate_comparison_result(part, screening)
                        if agent_hints:
                            validate_agent_hints(part)
                    tokens = int((part / "kv-cache-tokens").read_text())
                    if expected_kv_tokens is not None and expected_kv_tokens != tokens:
                        raise RuntimeError("Completed arms have different KV capacity")
                    expected_kv_tokens = tokens
                    cleanup_offload_cache(part)
                    print("REUSE completed arm", name, flush=True)
                    continue
                if cpu_gib:
                    available = (
                        int(
                            re.search(
                                r"MemAvailable:\s+(\d+)",
                                Path("/proc/meminfo").read_text(),
                            )[1]
                        )
                        * 1024
                    )
                    shm = os.statvfs("/dev/shm")
                    if available < (cpu_gib + 32) * 2**30:
                        raise RuntimeError(
                            "Insufficient available RAM for CPU KV cache"
                        )
                    if shm.f_bavail * shm.f_frsize < (cpu_gib + 1) * 2**30:
                        raise RuntimeError(
                            "Insufficient shared memory for CPU KV cache"
                        )
                part.mkdir()
                (run_dir / "status").write_text(f"{name}: starting_server")
                (part / "status").write_text("starting_server")
                serve = serve_template.replace(
                    "RUN=$ROOT/experiments/agentx-qwen3-8b-flash-attn",
                    f"RUN={part}",
                ).rstrip()
                additional_config = {"agentrix_capacity_bypass": enabled}
                if adaptive_sweep:
                    additional_config["agentrix_adaptive_prefill"] = name != "lru"
                if gpu_session_sweep and name != "lru":
                    additional_config["agentrix_gpu_session_retention"] = (
                        gpu_retention_config
                    )
                config = json.dumps(additional_config)
                serve += f" --long-prefill-token-threshold {cap}"
                if kv_cache_bytes is not None:
                    serve += f" --kv-cache-memory-bytes {kv_cache_bytes}"
                if cpu_gib:
                    transfer = {
                        "kv_connector": "OffloadingConnector",
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": {
                            "cpu_bytes_to_use": cpu_gib * 2**30,
                            "blocks_per_chunk": 1,
                            "eviction_policy": policies[name],
                            "store_threshold": 0,
                            "offload_prompt_only": True,
                        },
                    }
                    if policies[name] == "session_lru":
                        transfer["kv_connector_extra_config"]["cache_policy_config"] = (
                            policy_config
                        )
                    serve += " --kv-transfer-config " + shlex.quote(
                        json.dumps(transfer)
                    )
                if upstream_env is None:
                    serve += " --additional-config " + shlex.quote(config)
                serve += "\n"
                (part / "serve.sh").write_text(serve)
                env = dict(os.environ)
                env.pop("AGENTRIX_CAPACITY_PROBE", None)
                env["VLLM_SERVER_DEV_MODE"] = (
                    "1" if adaptive_sweep or name.startswith("smoke_") else "0"
                )
                server = None
                bench = None
                try:
                    with (part / "server.log").open("w") as output:
                        server = subprocess.Popen(
                            ["bash", "-c", serve],
                            env=env,
                            stdout=output,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    for _ in range(300):
                        if server.poll() is not None:
                            raise RuntimeError(f"Server exited: {server.returncode}")
                        try:
                            urllib.request.urlopen(
                                "http://127.0.0.1:18000/health", timeout=2
                            ).close()
                            break
                        except OSError:
                            time.sleep(2)
                    else:
                        raise RuntimeError("Server readiness timed out")
                    if (
                        cpu_gib
                        and "cudaHostRegister failed"
                        in (part / "server.log").read_text()
                    ):
                        raise RuntimeError("CPU KV cache could not be pinned for DMA")
                    capacity = re.search(
                        r"GPU KV cache size: ([\d,]+) tokens",
                        (part / "server.log").read_text(),
                    )
                    if capacity is None:
                        raise RuntimeError("Server did not report KV cache capacity")
                    kv_tokens = int(capacity[1].replace(",", ""))
                    (part / "kv-cache-tokens").write_text(str(kv_tokens))
                    if expected_kv_tokens is None:
                        expected_kv_tokens = kv_tokens
                    elif kv_tokens != expected_kv_tokens:
                        raise RuntimeError("KV cache capacity changed between arms")
                    print("READY", name, "kv_tokens", kv_tokens, flush=True)
                    if (
                        adaptive_sweep
                        and name != "lru"
                        and "Adaptive prefill enabled:"
                        not in (part / "server.log").read_text()
                    ):
                        raise RuntimeError("Adaptive prefill was not activated")
                    if (
                        gpu_session_sweep
                        and name != "lru"
                        and "GPU session retention enabled:"
                        not in (part / "server.log").read_text()
                    ):
                        raise RuntimeError("GPU session retention was not activated")
                    if upstream_env is not None:
                        server_log = (part / "server.log").read_text()
                        if any(
                            marker not in server_log
                            for marker in (
                                "Using V2 Model Runner",
                                "Using FlashAttention version 3",
                            )
                        ):
                            raise RuntimeError(
                                "Official baseline runner/backend differs"
                            )
                    if name.startswith("smoke_"):
                        (run_dir / "status").write_text(f"{name}: checking_roundtrip")
                        (part / "status").write_text("checking_roundtrip")
                        check_offload_roundtrip(part, mixed=adaptive_sweep)
                        (part / "status").write_text("success")
                        print("DONE", name, flush=True)
                        continue
                    if adaptive_sweep:
                        (run_dir / "status").write_text(f"{name}: checking_roundtrip")
                        (part / "status").write_text("checking_roundtrip")
                        check_offload_roundtrip(part, mixed=True, reset_after=True)
                        print("CHECKED", name, "GPU/CPU and mixed outputs", flush=True)
                    (run_dir / "status").write_text(f"{name}: warmup_and_profiling")
                    (part / "status").write_text("warmup_and_profiling")
                    command = next(
                        line.strip()
                        for line in matrix_template.splitlines()
                        if line.strip().startswith(".venv/bin/aiperf profile ")
                    ).split(' > "c${concurrency}.log"')[0]
                    command = (
                        command.replace('"$concurrency"', "8")
                        .replace(
                            "--benchmark-duration 3600",
                            f"--benchmark-duration {duration} --benchmark-grace-period 300",
                        )
                        .replace('"$RUN/c${concurrency}"', str(part / "bench"))
                        .replace(
                            "--warmup-requests-per-lane 10",
                            f"--warmup-requests-per-lane {warmup_requests_per_lane}",
                        )
                    )
                    if screening:
                        command += " --unsafe-override"
                    (part / "command.txt").write_text(command + "\n")
                    bench_setup = f"cd {baseline}; source ./bench-env.sh; "
                    if agent_hints:
                        hooks = part / "hints"
                        hooks.mkdir()
                        (hooks / "sitecustomize.py").write_text(
                            "import os, traceback\n"
                            "try:\n"
                            "    from probe_capacity import install_agent_hints\n"
                            "    install_agent_hints()\n"
                            "except Exception:\n"
                            "    traceback.print_exc()\n"
                            "    os._exit(70)\n"
                        )
                        bench_setup += (
                            "export PYTHONPATH="
                            + shlex.quote(
                                str(hooks)
                                + os.pathsep
                                + str(Path(__file__).resolve().parent)
                            )
                            + " AGENTRIX_SESSION_HINT_AUDIT="
                            + shlex.quote(str(hooks))
                            + "; "
                        )
                    with urllib.request.urlopen(
                        "http://127.0.0.1:18000/metrics", timeout=10
                    ) as response:
                        (part / "metrics-before.prom").write_bytes(response.read())
                    with (part / "bench.log").open("w") as output:
                        bench = subprocess.Popen(
                            [
                                "bash",
                                "-c",
                                bench_setup + f"exec {command}",
                            ],
                            stdout=output,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        bench.wait(timeout=duration + 3600)
                    (part / "bench.exit").write_text(str(bench.returncode))
                    with urllib.request.urlopen(
                        "http://127.0.0.1:18000/metrics", timeout=10
                    ) as response:
                        (part / "metrics-after.prom").write_bytes(response.read())
                    validate_comparison_result(part, screening)
                    phase = re.search(
                        r"Phase profiling \(profiling\) complete \| "
                        r"completed=(\d+), cancelled=(\d+), errors=(\d+)",
                        (part / "bench/logs/aiperf.log").read_text(),
                    )
                    if phase is None or int(phase[2]) or int(phase[3]):
                        raise RuntimeError(f"{name} profiling incomplete or has errors")
                    if agent_hints:
                        validate_agent_hints(part)
                    (part / "status").write_text("success")
                except BaseException:
                    (part / "status").write_text("failed")
                    raise
                finally:
                    try:
                        if bench is not None and bench.poll() is None:
                            os.killpg(bench.pid, signal.SIGTERM)
                            try:
                                bench.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                os.killpg(bench.pid, signal.SIGKILL)
                                bench.wait(timeout=10)
                    finally:
                        if server is not None and server.poll() is None:
                            stop_server(server.pid)
                            server.wait(timeout=10)
                        cleanup_offload_cache(part)
                print("DONE", name, flush=True)
            (run_dir / "status").write_text("success")
        except BaseException:
            (run_dir / "status").write_text("failed")
            raise
        print("DONE", run_dir, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--ab", action="store_true")
    mode.add_argument("--prefill-sweep", action="store_true")
    mode.add_argument("--offload-sweep", action="store_true")
    mode.add_argument("--session-sweep", action="store_true")
    mode.add_argument("--gpu-session-sweep", action="store_true")
    mode.add_argument("--adaptive-prefill-sweep", action="store_true")
    mode.add_argument("--upstream-sweep", action="store_true")
    parser.add_argument("--policy-source", type=Path)
    parser.add_argument("--upstream-env", type=Path)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--after-run", type=Path)
    parser.add_argument("--after-arm", choices=["baseline", "cap512", "cap1024"])
    parser.add_argument("--kv-cache-bytes", type=int)
    parser.add_argument(
        "--warmup-requests-per-lane",
        type=int,
        default=10,
        help="Warmup turns per lane; use 2 for adaptive-prefill screening",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=900,
        help="Measured seconds per arm; default 900, or >=300 for adaptive screening",
    )
    parser.add_argument(
        "--include-arc",
        action="store_true",
        help="Add ARC to the session-policy comparison after initial screening",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse measured arms after smoke checks",
    )
    args = parser.parse_args()
    comparison = (
        args.ab
        or args.prefill_sweep
        or args.offload_sweep
        or args.session_sweep
        or args.gpu_session_sweep
        or args.adaptive_prefill_sweep
        or args.upstream_sweep
    )
    if args.reverse and not (
        args.ab or args.gpu_session_sweep or args.adaptive_prefill_sweep
    ):
        parser.error(
            "--reverse requires --ab, --gpu-session-sweep or --adaptive-prefill-sweep"
        )
    if (
        args.prefill_sweep
        or args.offload_sweep
        or args.session_sweep
        or args.gpu_session_sweep
        or args.adaptive_prefill_sweep
        or args.upstream_sweep
    ) and args.kv_cache_bytes is None:
        parser.error("Sweeps require --kv-cache-bytes")
    if (
        args.session_sweep or args.gpu_session_sweep or args.adaptive_prefill_sweep
    ) != (args.policy_source is not None):
        parser.error("Policy sweeps and --policy-source must be supplied together")
    if args.upstream_sweep != (args.upstream_env is not None):
        parser.error("--upstream-sweep and --upstream-env must be supplied together")
    if args.include_arc and not args.session_sweep:
        parser.error("--include-arc requires --session-sweep")
    if args.after_run is not None and not comparison:
        parser.error("--after-run requires a comparison")
    if args.resume_run is not None and (not comparison or args.after_run is not None):
        parser.error("--resume-run requires a comparison without --after-run")
    if args.after_arm is not None and (
        args.after_run is None or not (args.offload_sweep or args.session_sweep)
    ):
        parser.error("--after-arm requires --after-run and an offload/session sweep")
    if args.kv_cache_bytes is not None and (args.kv_cache_bytes <= 0 or not comparison):
        parser.error("--kv-cache-bytes must be positive and requires a comparison")
    if args.warmup_requests_per_lane < 1 or (
        args.warmup_requests_per_lane != 10 and not args.adaptive_prefill_sweep
    ):
        parser.error(
            "Shorter warmup requires --adaptive-prefill-sweep and must be positive"
        )
    if comparison:
        min_duration = 300 if args.adaptive_prefill_sweep else 900
        if args.duration < min_duration:
            parser.error(f"Comparison requires at least {min_duration} seconds per arm")

        def terminate(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, terminate)
        mode = "capacity-ab"
        if args.prefill_sweep:
            mode = "prefill-sweep"
        elif args.offload_sweep:
            mode = "offload-sweep"
        elif args.session_sweep:
            mode = "session-sweep"
        elif args.gpu_session_sweep:
            mode = "gpu-session-sweep"
        elif args.adaptive_prefill_sweep:
            mode = "adaptive-prefill-sweep"
        elif args.upstream_sweep:
            mode = "upstream-sweep"
        run_comparison(
            args.duration,
            mode=mode,
            reverse=args.reverse,
            kv_cache_bytes=args.kv_cache_bytes,
            after_run=args.after_run,
            after_arm=args.after_arm,
            policy_source=args.policy_source,
            include_arc=args.include_arc,
            upstream_env=args.upstream_env,
            resume_run=args.resume_run,
            warmup_requests_per_lane=args.warmup_requests_per_lane,
        )
    else:
        run()
