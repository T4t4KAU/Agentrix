import subprocess
from pathlib import Path
workspace = Path(__file__).resolve().parents[1]
r = subprocess.run((str(workspace / "build" / "sqlite3"), "-batch", ":memory:", "SELECT length('Agentrix'), length('你好');"), text=True, capture_output=True, timeout=30)
assert r.returncode == 0 and r.stdout.strip() == "8|2", r.stdout + r.stderr
