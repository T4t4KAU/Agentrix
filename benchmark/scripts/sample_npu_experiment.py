"""Keep raw NPU snapshots on the server while a specific controller is alive."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-memory", action="store_true")
    args = parser.parse_args()
    controller = psutil.Process(args.pid)
    with args.output.open("x") as log:
        while controller.is_running() and controller.status() != psutil.STATUS_ZOMBIE:
            result = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            try:
                owned = [p.pid for p in controller.children(recursive=True)]
            except psutil.NoSuchProcess:
                owned = []
            host_memory = None
            if args.host_memory:
                # PSS apportions shared pages; summed RSS double-counts them.
                # This covers the experiment process tree, not CPU KV alone.
                measured = []
                missing = []
                for pid in owned:
                    try:
                        memory = psutil.Process(pid).memory_full_info()
                        measured.append(
                            {"pid": pid, "pss": memory.pss, "uss": memory.uss}
                        )
                    except (psutil.Error, AttributeError):
                        missing.append(pid)
                host_memory = {"processes": measured, "missing_pids": missing}
            log.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "controller_descendants": owned,
                        "host_memory": host_memory,
                    }
                )
                + "\n"
            )
            log.flush()
            time.sleep(5)


if __name__ == "__main__":
    main()
