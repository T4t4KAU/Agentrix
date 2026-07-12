import csv
import json
from pathlib import Path

from offload_comparison_report import collect_variant, render_report


FIELDNAMES = [
    "case_wall_time_ms",
    "branch_phase_wall_ms",
    "branch_total_output_tokens",
    "branch_output_tokens_per_s",
    "end_to_end_output_tokens_per_s",
    "baseline_unique_kv",
    "monowire_unique_kv",
    "kv_bytes_per_token",
    "kv_bytes_saved",
]


def _write_variant(root: Path, name: str, scale: int) -> None:
    backend_name = "flash_attn" if name.startswith("flash_") else "fork_attn"
    backend = root / name / backend_name
    backend.mkdir(parents=True)
    with (backend / "benchmark_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "case_wall_time_ms": 100 / scale,
                "branch_phase_wall_ms": 50 / scale,
                "branch_total_output_tokens": 20,
                "branch_output_tokens_per_s": 400 * scale,
                "end_to_end_output_tokens_per_s": 200 * scale,
                "baseline_unique_kv": 1000,
                "monowire_unique_kv": 600,
                "kv_bytes_per_token": 1024,
                "kv_bytes_saved": 409600,
            }
        )
    (backend / "server_profile.json").write_text(
        json.dumps(
            {
                "kv_offload_load_bytes": scale * 1024**3,
                "kv_offload_store_bytes": scale * 2 * 1024**3,
            }
        ),
        encoding="utf-8",
    )
    (backend / "vllm_server.log").write_text(
        f"Reqid: req-1, need to load: {scale * 256}\n"
        f"Retrieved {scale * 256} out of 256 required tokens.\n"
        f"Stored {scale * 512} out of total 512 tokens.\n",
        encoding="utf-8",
    )
    (root / name / "storage_summary.txt").write_text(
        f"disk_bytes={scale * 1024**3}\n", encoding="utf-8"
    )


def test_collect_variant_uses_native_prometheus_bytes(tmp_path: Path) -> None:
    _write_variant(tmp_path, "native_cpu", 2)
    result = collect_variant(tmp_path, "native_cpu")
    assert result["load_gib"] == 2
    assert result["store_gib"] == 4


def test_collect_variant_converts_lmcache_tokens(tmp_path: Path) -> None:
    _write_variant(tmp_path, "lmcache_cpu", 2)
    result = collect_variant(tmp_path, "lmcache_cpu")
    assert result["load_gib"] == 512 * 1024 / 1024**3
    assert result["store_gib"] == 1024 * 1024 / 1024**3
    assert result["disk_load_failures"] == 0


def test_collect_variant_keeps_load_failure_types_separate(tmp_path: Path) -> None:
    _write_variant(tmp_path, "lmcache_tiered", 1)
    log = tmp_path / "lmcache_tiered" / "fork_attn" / "vllm_server.log"
    with log.open("a", encoding="utf-8") as handle:
        handle.write("Memory allocation failed during disk load\n")
        handle.write("failed to load 256 tokens\n")
    result = collect_variant(tmp_path, "lmcache_tiered")
    assert result["load_failures"] == 1
    assert result["disk_load_failures"] == 1


def test_render_report_compares_all_variants(tmp_path: Path) -> None:
    for index, name in enumerate(
        (
            "no_offload",
            "native_cpu",
            "lmcache_default_cpu",
            "lmcache_cpu",
            "lmcache_tiered",
            "flash_no_offload",
            "flash_native_cpu",
        ),
        start=1,
    ):
        _write_variant(tmp_path, name, index)
    report = render_report(tmp_path)
    assert "Baseline: ForkAttention without KV offloading." in report
    assert "| Native CPU | 50.000 | 400.000 | +100.00%" in report
    assert "| LMCache default CPU (LRU) |" in report
    assert "| LMCache fork-aware CPU |" in report
    assert "| LMCache CPU+disk |" in report
    assert "| FlashAttention no offload |" in report
    assert "| FlashAttention + native CPU |" in report
    assert "## Pairwise Offload Impact" in report
    assert "| FlashAttention native CPU vs no offload | +16.67% | +16.67% |" in report
    assert "## Logical KV Cache Footprint" in report
    assert "| 1000 | 600 | 400 | 0.000 | 40.00% |" in report
