import os
import subprocess
import sys
from pathlib import Path

workspace = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env["PYTHONPATH"] = str(workspace)
result = subprocess.run(
    (sys.executable, "tests/runtests.py", "handlers.tests.TransactionsPerRequestTests.test_non_atomic_requests_does_not_mutate_original", "--verbosity", "0"),
    cwd=workspace, env=env, text=True, capture_output=True, timeout=90,
)
assert result.returncode == 0, result.stdout + result.stderr
