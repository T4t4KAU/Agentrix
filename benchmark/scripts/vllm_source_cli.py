#!/usr/bin/env python3
"""Run the vLLM CLI directly from an Agentrix source checkout."""

# Standard
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VLLM_SOURCE = str(REPO_ROOT / "vllm")
sys.path.insert(0, VLLM_SOURCE)
os.environ["PYTHONPATH"] = (
    f"{VLLM_SOURCE}:{os.environ['PYTHONPATH']}"
    if os.environ.get("PYTHONPATH")
    else VLLM_SOURCE
)


def main() -> None:
    """Run the vLLM command-line entry point from the source checkout."""
    # First Party
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
