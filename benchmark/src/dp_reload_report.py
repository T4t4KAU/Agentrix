from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


def _read_variant(root: Path, name: str) -> dict[str, Any]:
    variant = root / name / "fork_attn"
    with (variant / "benchmark_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    profile = json.loads((variant / "server_profile.json").read_text(encoding="utf-8"))

    def total(key: str) -> float:
        return sum(float(row.get(key, 0)) for row in rows)

    output_tokens = total("branch_total_output_tokens")
    branch_ms = total("branch_phase_wall_ms")
    e2e_ms = total("case_wall_time_ms")
    return {
        "batches": len(rows),
        "branch_tps": output_tokens * 1000 / branch_ms if branch_ms else 0.0,
        "e2e_tps": output_tokens * 1000 / e2e_ms if e2e_ms else 0.0,
        "logical_kv_tokens_saved": total("kv_tokens_saved"),
        "logical_kv_gib_saved": total("kv_bytes_saved") / 1024**3,
        "preemptions": float(profile.get("num_preemptions", 0)),
        "reload": profile.get("fork_dp_reload", {}),
    }


def _delta(candidate: float, baseline: float) -> float:
    return 100 * (candidate / baseline - 1) if baseline else 0.0


def main() -> int:
    root = Path(sys.argv[1])
    baseline = _read_variant(root, "baseline")
    optimized = _read_variant(root, "optimized")
    reload = optimized["reload"]
    lines = [
        "# DP Reload Rebalance Comparison",
        "",
        "Both variants use the default LMCache LRU policy and an empty cache.",
        "The experimental DP reload rebalance flag is the only policy difference.",
        "",
        "| Variant | Branch tok/s | E2E tok/s | Preemptions | Logical KV saved (GiB) | GPU-local reload saved (GiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, data in (("Baseline", baseline), ("Optimized", optimized)):
        saved_reload = float(data["reload"].get("saved_reload_gib", 0))
        lines.append(
            f"| {name} | {data['branch_tps']:.2f} | {data['e2e_tps']:.2f} "
            f"| {data['preemptions']:.0f} | {data['logical_kv_gib_saved']:.3f} "
            f"| {saved_reload:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Delta",
            "",
            f"- Branch throughput: {_delta(optimized['branch_tps'], baseline['branch_tps']):+.2f}%",
            f"- End-to-end throughput: {_delta(optimized['e2e_tps'], baseline['e2e_tps']):+.2f}%",
            f"- Reload handoffs: {int(reload.get('committed', 0))} committed / "
            f"{int(reload.get('planned', 0))} planned",
            f"- GPU-local KV reload saved: {int(reload.get('saved_reload_tokens', 0))} tokens "
            f"({float(reload.get('saved_reload_gib', 0)):.3f} GiB)",
            "",
        ]
    )
    (root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
