import subprocess
from pathlib import Path
workspace = Path(__file__).resolve().parents[1]
r = subprocess.run((str(workspace / "build" / "sqlite3"), "-batch", ":memory:", "SELECT unicode('A'), unicode('中');"), text=True, capture_output=True, timeout=30)
assert r.returncode == 0 and r.stdout.strip() == "65|20013", r.stdout + r.stderr
