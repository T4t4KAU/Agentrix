from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


def run_script(name: str, **environment: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(BENCHMARK_ROOT / "scripts" / name)],
        cwd=BENCHMARK_ROOT.parent,
        env={**os.environ, **environment},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("script", "environment"),
    [
        ("run_cacheblend_langgraph_rag.sh", {}),
        (
            "run_langgraph_fork_cacheblend_20_e2e.sh",
            {"VARIANTS": "cacheblend"},
        ),
    ],
)
def test_cacheblend_requires_explicit_opt_in(
    script: str, environment: dict[str, str]
) -> None:
    result = run_script(script, ENABLE_CACHEBLEND="0", **environment)

    assert result.returncode == 2
    assert "CacheBlend is disabled by default" in result.stderr


def test_cacheblend_rejects_plain_rag_format() -> None:
    result = run_script(
        "run_langgraph_fork_cacheblend_20_e2e.sh",
        ENABLE_CACHEBLEND="1",
        VARIANTS="cacheblend",
        RAG_FORMAT="plain",
    )

    assert result.returncode == 2
    assert "require RAG_FORMAT=cacheblend" in result.stderr
