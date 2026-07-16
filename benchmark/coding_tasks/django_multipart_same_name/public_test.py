import os
import subprocess
import sys
from pathlib import Path

workspace = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env["PYTHONPATH"] = str(workspace)
result = subprocess.run(
    (sys.executable, "tests/runtests.py", "test_client.tests.ClientTest.test_uploading_file_and_field_with_same_name", "--verbosity", "0"),
    cwd=workspace, env=env, text=True, capture_output=True, timeout=90,
)
assert result.returncode == 0, result.stdout + result.stderr
