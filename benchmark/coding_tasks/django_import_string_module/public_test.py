import os
import subprocess
import sys
from pathlib import Path

workspace = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env["PYTHONPATH"] = str(workspace)
result = subprocess.run(
    (sys.executable, "tests/runtests.py", "utils_tests.test_module_loading.ModuleImportTests.test_import_string_unloaded_module", "utils_tests.test_module_loading.ModuleImportTests.test_import_string_objects_collision_handling", "--verbosity", "0"),
    cwd=workspace, env=env, text=True, capture_output=True, timeout=90,
)
assert result.returncode == 0, result.stdout + result.stderr
