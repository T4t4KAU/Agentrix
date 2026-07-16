import subprocess
from pathlib import Path
w = Path(__file__).resolve().parents[1]
r = subprocess.run(("make", "fate-bprint"), cwd=w / "build", text=True, capture_output=True, timeout=120)
assert r.returncode == 0, r.stdout + r.stderr
