"""AgentX capacity observations and scheduling comparisons on the server."""

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
from pathlib import Path

_last_sample = 0.0
_failures = 0
_stored = {}
_evicted = {}
_lookup_seen = {}


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


def check_offload_roundtrip(part):
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
    (part / "roundtrip.json").write_text(
        json.dumps(
            {
                "first": first,
                "restored": second,
                "same_output": same,
                "loaded_bytes": loaded_bytes,
                "stored_bytes": counter(stored, store_name)
                - counter(initial, store_name),
            },
            indent=2,
        )
        + "\n"
    )
    (part / "metrics-before.prom").write_text(initial)
    (part / "metrics-after.prom").write_text(restored)
    if not same:
        raise RuntimeError("Output changed after CPU KV cache roundtrip")


def finish_prefill_after(previous, arm):
    """Finish the selected arms and retire the original four-arm controller."""
    baseline = previous.parent
    if Path((baseline / "prefill-sweep-latest").read_text().strip()) != previous:
        raise RuntimeError("The selected prefill run is no longer current")
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
        profile = json.loads((part / "bench/profile_export_aiperf.json").read_text())
        if (
            (part / "bench.exit").read_text().strip() != "0"
            or profile["error_summary"]
            or profile["was_cancelled"]
            or not profile["metadata"].get("submission_valid")
        ):
            raise RuntimeError(f"Cannot finish run: {name} has invalid results")

    pid = int((baseline / "prefill-sweep-controller.pid").read_text())
    command = Path(f"/proc/{pid}/cmdline")
    if command.exists() and command.read_bytes():
        args = command.read_bytes().split(b"\0")
        if b"--prefill-sweep" not in args or not any(
            arg.endswith(b"/probe_capacity.py") for arg in args
        ):
            raise RuntimeError("PID is not the preceding prefill controller")
        os.kill(pid, signal.SIGTERM)
        for _ in range(180):
            stat = Path(f"/proc/{pid}/stat")
            if not stat.exists() or stat.read_text().split(") ", 1)[1].startswith("Z"):
                break
            time.sleep(1)
        else:
            raise RuntimeError("Preceding prefill controller did not stop")
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


def run_comparison(
    duration,
    mode="capacity-ab",
    reverse=False,
    kv_cache_bytes=None,
    after_run=None,
    after_arm=None,
):
    root = Path("/mnt/sda1/hwx/Agentrix")
    baseline = root / "experiments/agentx-qwen3-8b-flash-attn"
    source = root / "vllm/vllm/v1/core/sched/scheduler.py"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if after_run is not None:
        previous = Path(after_run).resolve()
        if previous.parent != baseline or not (previous / "status").is_file():
            raise ValueError("--after-run must name an existing benchmark run")
        if after_arm is not None:
            finish_prefill_after(previous, after_arm)
        print("WAIT", previous, flush=True)
        while True:
            status = (previous / "status").read_text().strip()
            if status == "success":
                break
            if status == "failed":
                raise RuntimeError("Preceding benchmark failed; comparison not started")
            time.sleep(30)
    prefix = mode
    run_dir = baseline / (prefix + "-" + time.strftime("%Y%m%d-%H%M%S"))
    arms = [("baseline", False, 0, 0), ("bypass", True, 0, 0)]
    if mode == "prefill-sweep":
        arms = [("baseline", False, 0, 0)] + [
            (f"cap{cap}", False, cap, 0) for cap in (512, 1024, 2048)
        ]
    elif mode == "offload-sweep":
        arms = [("smoke_cpu32", False, 0, 32), ("baseline", False, 0, 0)] + [
            (f"cpu{gib}", False, 0, gib) for gib in (32, 64)
        ]
    if reverse:
        arms.reverse()
    # Share the original matrix lock so two experiments cannot own GPU 1.
    with (baseline / "matrix.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if after_run else fcntl.LOCK_NB))
        run_dir.mkdir()
        (baseline / (prefix + "-latest")).write_text(str(run_dir))
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "duration_seconds": duration,
                    "grace_seconds": 300,
                    "concurrency": 8,
                    "seed": 20260707,
                    "order": [name for name, _, _, _ in arms],
                    "smoke_arms": [
                        name for name, _, _, _ in arms if name.startswith("smoke_")
                    ],
                    "scheduler_sha256": source_hash,
                    "kv_cache_memory_bytes": kv_cache_bytes,
                    "long_prefill_token_thresholds": {
                        name: cap for name, _, cap, _ in arms
                    },
                    "capacity_bypass": {name: enabled for name, enabled, _, _ in arms},
                    "cpu_offload_gib": {name: gib for name, _, _, gib in arms},
                    "offload_policy": "lru",
                    "offload_prompt_only": True,
                    "offload_blocks_per_chunk": 1,
                    "offload_store_threshold": 0,
                    "after_run": str(after_run) if after_run else None,
                    "after_arm": after_arm,
                    "lookahead": 8,
                    "max_overtakes_per_request": 4,
                    "observation_hooks": False,
                },
                indent=2,
            )
            + "\n"
        )
        print("RUN", run_dir, flush=True)
        serve_template = (baseline / "serve.sh").read_text()
        matrix_template = (baseline / "run-matrix.sh").read_text()
        expected_kv_tokens = None
        try:
            stop_server(int((baseline / "server.pid").read_text()))
            for name, enabled, cap, cpu_gib in arms:
                if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
                    raise RuntimeError("Scheduler source changed during comparison")
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
                part = run_dir / name
                part.mkdir()
                (run_dir / "status").write_text(f"{name}: starting_server")
                (part / "status").write_text("starting_server")
                serve = serve_template.replace(
                    "RUN=$ROOT/experiments/agentx-qwen3-8b-flash-attn",
                    f"RUN={part}",
                ).rstrip()
                config = json.dumps({"agentrix_capacity_bypass": enabled})
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
                            "eviction_policy": "lru",
                            "store_threshold": 0,
                            "offload_prompt_only": True,
                        },
                    }
                    serve += " --kv-transfer-config " + shlex.quote(
                        json.dumps(transfer)
                    )
                serve += " --additional-config " + shlex.quote(config) + "\n"
                (part / "serve.sh").write_text(serve)
                env = dict(os.environ)
                env.pop("AGENTRIX_CAPACITY_PROBE", None)
                env["VLLM_SERVER_DEV_MODE"] = "1" if name.startswith("smoke_") else "0"
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
                    if name.startswith("smoke_"):
                        (run_dir / "status").write_text(f"{name}: checking_roundtrip")
                        (part / "status").write_text("checking_roundtrip")
                        check_offload_roundtrip(part)
                        (part / "status").write_text("success")
                        print("DONE", name, flush=True)
                        continue
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
                    )
                    (part / "command.txt").write_text(command + "\n")
                    with urllib.request.urlopen(
                        "http://127.0.0.1:18000/metrics", timeout=10
                    ) as response:
                        (part / "metrics-before.prom").write_bytes(response.read())
                    with (part / "bench.log").open("w") as output:
                        bench = subprocess.Popen(
                            [
                                "bash",
                                "-c",
                                f"cd {baseline}; source ./bench-env.sh; exec {command}",
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
                    if bench.returncode:
                        raise RuntimeError(f"Benchmark exited: {bench.returncode}")
                    profile = json.loads(
                        (part / "bench/profile_export_aiperf.json").read_text()
                    )
                    if profile["error_summary"] or profile["was_cancelled"]:
                        raise RuntimeError(
                            f"{name} benchmark has errors or was cancelled"
                        )
                    if not profile["metadata"].get("submission_valid"):
                        raise RuntimeError(
                            f"{name} benchmark failed scenario validation"
                        )
                    phase = re.search(
                        r"Phase profiling \(profiling\) complete \| "
                        r"completed=(\d+), cancelled=(\d+), errors=(\d+)",
                        (part / "bench/logs/aiperf.log").read_text(),
                    )
                    if phase is None or int(phase[2]) or int(phase[3]):
                        raise RuntimeError(f"{name} profiling incomplete or has errors")
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
    parser.add_argument("--after-run", type=Path)
    parser.add_argument("--after-arm", choices=["baseline", "cap512", "cap1024"])
    parser.add_argument("--kv-cache-bytes", type=int)
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument(
        "--reverse", action="store_true", help="Run bypass before baseline"
    )
    args = parser.parse_args()
    comparison = args.ab or args.prefill_sweep or args.offload_sweep
    if args.reverse and not args.ab:
        parser.error("--reverse requires --ab")
    if (args.prefill_sweep or args.offload_sweep) and args.kv_cache_bytes is None:
        parser.error("Sweeps require --kv-cache-bytes")
    if args.after_run is not None and not comparison:
        parser.error("--after-run requires a comparison")
    if args.after_arm is not None and (
        args.after_run is None or not args.offload_sweep
    ):
        parser.error("--after-arm requires --offload-sweep and --after-run")
    if args.kv_cache_bytes is not None and (args.kv_cache_bytes <= 0 or not comparison):
        parser.error("--kv-cache-bytes must be positive and requires a comparison")
    if comparison:
        if args.duration < 900:
            parser.error("Comparison runs require at least 900 seconds per arm")

        def terminate(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, terminate)
        mode = "capacity-ab"
        if args.prefill_sweep:
            mode = "prefill-sweep"
        elif args.offload_sweep:
            mode = "offload-sweep"
        run_comparison(
            args.duration,
            mode=mode,
            reverse=args.reverse,
            kv_cache_bytes=args.kv_cache_bytes,
            after_run=args.after_run,
            after_arm=args.after_arm,
        )
    else:
        run()
