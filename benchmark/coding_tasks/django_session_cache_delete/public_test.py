import os
import subprocess
import sys
from pathlib import Path

workspace = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env["PYTHONPATH"] = str(workspace)
result = subprocess.run(
    (sys.executable, "tests/runtests.py", "sessions_tests.tests.CacheDBSessionTests.test_cache_delete_failure_non_fatal", "--verbosity", "0"),
    cwd=workspace, env=env, text=True, capture_output=True, timeout=90,
)
assert result.returncode == 0, result.stdout + result.stderr
