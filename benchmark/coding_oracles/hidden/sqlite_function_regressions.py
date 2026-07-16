from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


QUERIES = {
    "sqlite_length_text": ("SELECT length(''), length('abc'), length('猫a');", "0|3|2"),
    "sqlite_typeof_text": ("SELECT typeof(1), typeof(1.5), typeof('x'), typeof(x'01'), typeof(NULL);", "integer|real|text|blob|null"),
    "sqlite_round_negative_precision": ("SELECT round(1.25,-2), round(-1.25,-2), round(2.55,1);", "1.0|-1.0|2.5"),
    "sqlite_unicode_first_character": ("SELECT unicode('A'), unicode('猫'), unicode('') IS NULL;", "65|29483|1"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task-id", choices=QUERIES, required=True)
    args = parser.parse_args()
    sql, expected = QUERIES[args.task_id]
    result = subprocess.run(
        (str(args.workspace / "build" / "sqlite3"), "-batch", ":memory:", sql),
        text=True, capture_output=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected, result.stdout


if __name__ == "__main__":
    main()
