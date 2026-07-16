from __future__ import annotations

import csv
from pathlib import Path

from hotpot_offload_report import _prom_delta, _sample_summary, _warning_count


def test_prometheus_counter_delta(tmp_path: Path) -> None:
    (tmp_path / "metrics_before.prom").write_text(
        'vllm:kv_offload_store_bytes_total{engine="0"} 1024\n',
        encoding="utf-8",
    )
    (tmp_path / "metrics.prom").write_text(
        'vllm:kv_offload_store_bytes_total{engine="0"} 4096\n',
        encoding="utf-8",
    )

    assert _prom_delta(tmp_path, "vllm:kv_offload_store_bytes") == 3072


def test_warning_count_uses_measured_server_interval(tmp_path: Path) -> None:
    (tmp_path / "vllm_server.log").write_text(
        "cannot store blocks\n", encoding="utf-8"
    )
    (tmp_path / "measured_server.log").write_text(
        "cannot store blocks\nother\ncannot store blocks\n", encoding="utf-8"
    )

    assert _warning_count(tmp_path, "cannot store blocks") == 2


def test_warning_count_falls_back_to_full_log(tmp_path: Path) -> None:
    (tmp_path / "vllm_server.log").write_text(
        "cannot store blocks\nother\ncannot store blocks\n", encoding="utf-8"
    )

    assert _warning_count(tmp_path, "cannot store blocks") == 2


def test_sample_summary_scales_fraction_and_integrates_pcie(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory_samples.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "unix_s",
                "vllm:kv_offload_cpu_cache_usage_perc",
                "vllm:kv_offload_cpu_cache_occupancy_perc",
                "pcie_rx_kib_s",
                "pcie_tx_kib_s",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "unix_s": 10,
                "vllm:kv_offload_cpu_cache_usage_perc": 0.25,
                "vllm:kv_offload_cpu_cache_occupancy_perc": 0.5,
                "pcie_rx_kib_s": 100,
                "pcie_tx_kib_s": 200,
            }
        )
        writer.writerow(
            {
                "unix_s": 12,
                "vllm:kv_offload_cpu_cache_usage_perc": 0.5,
                "vllm:kv_offload_cpu_cache_occupancy_perc": 1.0,
                "pcie_rx_kib_s": 300,
                "pcie_tx_kib_s": 400,
            }
        )

    summary = _sample_summary(tmp_path)

    assert summary["cpu_cache_usage_peak_percent"] == 50
    assert summary["cpu_cache_occupancy_peak_percent"] == 100
    assert summary["pcie_rx_integral_bytes"] == 300 * 2 * 1024
    assert summary["pcie_tx_integral_bytes"] == 400 * 2 * 1024
