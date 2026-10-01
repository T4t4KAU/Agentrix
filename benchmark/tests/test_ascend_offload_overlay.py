"""Fail-closed source backport and isolation regressions."""

import importlib.util
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/prepare_ascend_offload_overlay.py"
SPEC = importlib.util.spec_from_file_location("offload_overlay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("source", ["missing", "anchor anchor"])
def test_patch_refuses_source_drift(source):
    with pytest.raises(ValueError, match="Unsupported source"):
        MODULE.replace_once(source, "anchor", "replacement")


def test_backport_isolated_and_preserves_zero_cap(tmp_path):
    source = tmp_path / "installed"
    scheduler = (
        source / "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
    )
    route = source / "entrypoints/serve/cache/api_router.py"
    scheduler.parent.mkdir(parents=True)
    route.parent.mkdir(parents=True)
    old = (
        "        self.config = SchedulerOffloadConfig.from_spec(spec)\n"
        "            max_hit_size_tokens -= 1\n"
        "    # number of hits in the GPU cache\n"
        "    def update_offload_keys(self) -> None:\n"
        "            num_offloadable_tokens = min(num_tokens_after_batch, req.num_tokens)\n"
    )
    core = source / "v1/core/sched/scheduler.py"
    core.parent.mkdir(parents=True)
    core.write_text(
        "def schedule(self, load_kv_async, num_new_tokens):\n"
        "    for _ in (0,):\n"
        "        for _ in (0,):\n"
        "            for _ in (0,):\n"
        "                if self.need_mamba_block_aligned_split:\n"
        "                    if num_new_tokens == 0:\n"
        '                        return "blocked"\n'
        '                return "load" if load_kv_async else "compute"\n'
    )
    scheduler.write_text(old)
    route.write_text(
        "    await engine_client(raw_request).reset_prefix_cache(\n"
        "        reset_running_requests, reset_external\n"
        "    )\n"
        "    return Response(status_code=200)"
    )
    (source / "__pycache__").mkdir()
    (source / "unchanged.py").write_text("sentinel = 1\n")
    target = tmp_path / "overlay"
    MODULE.prepare(source, target)
    patched = (target / scheduler.relative_to(source)).read_text()
    assert "if req_status.max_offload_tokens is not None:" in patched
    assert "type(raw) is int and raw >= 0" in patched
    assert "and not load_kv_async" in (target / core.relative_to(source)).read_text()
    assert "and not load_kv_async" not in core.read_text()
    schedule_scope = {}
    exec((target / core.relative_to(source)).read_text(), schedule_scope)  # noqa: S102
    schedule = schedule_scope["schedule"]
    aligned = SimpleNamespace(need_mamba_block_aligned_split=True)
    assert schedule(aligned, True, 0) == "load"
    assert schedule(aligned, False, 0) == "blocked"
    assert schedule(aligned, False, 1024) == "compute"

    boundary_code = dedent(
        patched[
            patched.index("            max_hit_size_tokens -= 1") : patched.index(
                "    max_offload_tokens:"
            )
        ]
    )
    # Execute the emitted lookup boundary code: a recurrent state cannot be
    # rewound by one token like an attention page.
    for length, alignment, expected in [
        (8192, 1024, 7168),
        (8193, 1024, 8192),
        (1024, 1024, 0),
        (8192, None, 8191),
    ]:
        scope = {
            "self": SimpleNamespace(_mamba_align_size=alignment),
            "max_hit_size_tokens": length,
        }
        exec(boundary_code, scope)  # noqa: S102 - locally emitted patch regression
        assert scope["max_hit_size_tokens"] == expected
    assert scheduler.read_text() == old
    assert (target / "unchanged.py").is_symlink()
    assert not (target / "__pycache__").exists()
    with pytest.raises(FileExistsError):
        MODULE.prepare(source, target)


def test_newer_runtime_is_not_patched(tmp_path):
    source = tmp_path / "installed"
    scheduler = (
        source / "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
    )
    scheduler.parent.mkdir(parents=True)
    scheduler.write_text("max_offload_tokens = None\n")
    target = tmp_path / "overlay"
    with pytest.raises(ValueError, match="already implements"):
        MODULE.prepare(source, target)
    assert not target.exists()
