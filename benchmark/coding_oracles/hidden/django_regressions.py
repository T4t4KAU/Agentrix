from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


LABELS = {
    "django_non_atomic_view_copy": [
        "handlers.tests.TransactionsPerRequestTests",
    ],
    "django_multipart_same_name": [
        "test_client.tests.ClientTest.test_uploading_temp_file",
        "test_client.tests.ClientTest.test_uploading_named_temp_file",
    ],
    "django_session_cache_delete": [
        "sessions_tests.tests.CacheDBSessionTests.test_cache_async_delete_failure_non_fatal",
    ],
    "django_import_string_module": [
        "utils_tests.test_module_loading.ModuleImportTests.test_import_string_unloaded_submodule",
        "utils_tests.test_module_loading.ModuleImportTests.test_import_string_empty",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task-id", choices=LABELS, required=True)
    args = parser.parse_args()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.workspace)
    result = subprocess.run(
        (
            sys.executable,
            "tests/runtests.py",
            *LABELS[args.task_id],
            "--verbosity",
            "0",
        ),
        cwd=args.workspace,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    main()
