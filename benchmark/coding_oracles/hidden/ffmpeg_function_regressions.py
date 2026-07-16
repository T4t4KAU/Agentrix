from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


NEIGHBORS = {
    "ffmpeg_bprint_growth": "fate-avstring",
    "ffmpeg_dict_count": "fate-opt",
    "ffmpeg_fifo_readable": "fate-audio_fifo",
    "ffmpeg_parse_duration": "fate-eval",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task-id", choices=NEIGHBORS, required=True)
    args = parser.parse_args()
    result = subprocess.run(
        ("make", "-j16", NEIGHBORS[args.task_id]),
        cwd=args.workspace / "build", text=True, capture_output=True,
        timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    main()
