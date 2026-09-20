#!/usr/bin/env python3
"""Measure exact tool-snapshot revisions in isolated processes, without a model.

Compare an earlier prompt_compactor.py with today's streamed page sharing. Both
arms retain the same parent and branch versions and verify every byte by hash.
This measures application storage and host memory, not GPU KV or Agent quality.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CURRENT = ROOT / "application/src/agentrix_application/prompt_compactor.py"


def report_pages(count):
    # Unique deterministic ASCII pages prevent repeated padding from creating
    # an artificial within-snapshot deduplication advantage.
    for index in range(count):
        token = hashlib.sha256(f"tool-report-page-{index}".encode()).hexdigest()
        yield (f"report-page-{index:08d}\n" + token * 64)[:4096]


def edit(branch, chars):
    return (branch * 7919 + 101) % (chars - 32), f"branch-{branch:024d}\n"


def expected_hash(page_count, branch=None):
    offset, replacement = (
        edit(branch, page_count * 4096) if branch is not None else (-1, "")
    )
    hasher = hashlib.sha256()
    for number, page in enumerate(report_pages(page_count)):
        start = number * 4096
        lo, hi = max(offset, start), min(offset + len(replacement), start + len(page))
        if lo < hi:
            page = (
                page[: lo - start]
                + replacement[lo - offset : hi - offset]
                + page[hi - start :]
            )
        hasher.update(page.encode())
    return hasher.hexdigest()


def worker(args):
    spec = importlib.util.spec_from_file_location(
        "snapshot_store_under_test", args.source
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    page_count = args.size_mib * 256
    chars = page_count * 4096
    with tempfile.TemporaryDirectory(prefix="agentrix-snapshot-sharing-") as directory:
        path = Path(directory) / "snapshots.sqlite"
        store = module.PagedToolStore(path)
        store.open_session("parent")
        # Identical ingestion in both arms; the measured change is branch updates.
        original = "".join(report_pages(page_count))
        parent = store.put("parent", original)
        del original
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        versions = [("parent", parent)]
        for branch in range(args.branches):
            session = f"branch-{branch}"
            store.open_session(session, parent="parent")
            offset, replacement = edit(branch, chars)
            if args.worker == "private":
                body = "".join(
                    store.read(session, parent, offset=start, limit=16384)["content"]
                    for start in range(0, chars, 16384)
                )
                changed = (
                    body[:offset] + replacement + body[offset + len(replacement) :]
                )
                digest = store.put(session, changed)
                del body, changed
            else:
                digest = store.replace_range(
                    session,
                    parent,
                    offset=offset,
                    delete_chars=len(replacement),
                    content=replacement,
                )
            versions.append((session, digest))
        write_seconds = time.perf_counter() - started
        _, peak_python_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stats = store.stats()
        disk_bytes = path.stat().st_size
        # Verify complete historical results, including unchanged sibling data.
        verified = []
        for index, (session, digest) in enumerate(versions):
            actual = hashlib.sha256()
            for start in range(0, chars, 16384):
                actual.update(
                    store.read(session, digest, offset=start, limit=16384)[
                        "content"
                    ].encode()
                )
            expected = expected_hash(page_count, index - 1 if index else None)
            assert actual.hexdigest() == expected == digest
            verified.append(digest)
        for session, _ in versions:
            store.release_session(session)
        final = store.stats()
        assert final == dict(objects=0, stored_bytes=0, sessions=0, references=0)
        store.close()
        return dict(
            mode=args.worker,
            source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
            snapshot_bytes=chars,
            branches=args.branches,
            stats=stats,
            database_bytes=disk_bytes,
            peak_branch_python_bytes=peak_python_bytes,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            write_seconds=write_seconds,
            verified_sha256=verified,
            final=final,
            final_database_bytes=path.stat().st_size,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-store", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--size-mib", type=int, default=8)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--worker", choices=["private", "shared"], help=argparse.SUPPRESS
    )
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.size_mib, args.branches, args.repeats) < 1:
        parser.error("size, branches and repeats must be positive")
    if args.worker:
        print(json.dumps(worker(args)))
        return
    if not args.baseline_store or not args.output:
        parser.error("--baseline-store and --output are required")
    rows = []
    for repeat in range(args.repeats):
        order = ["private", "shared"] if repeat % 2 == 0 else ["shared", "private"]
        for mode in order:
            source = args.baseline_store if mode == "private" else CURRENT
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                mode,
                "--source",
                str(source),
                "--size-mib",
                str(args.size_mib),
                "--branches",
                str(args.branches),
            ]
            row = json.loads(subprocess.check_output(command, text=True))
            rows.append(row)
            print(
                mode,
                "stored_bytes",
                row["stats"]["stored_bytes"],
                "branch_python_peak",
                row["peak_branch_python_bytes"],
                flush=True,
            )
    assert all(row["verified_sha256"] == rows[0]["verified_sha256"] for row in rows)
    summary = {}
    for mode in ("private", "shared"):
        selected = [row for row in rows if row["mode"] == mode]
        summary[mode] = {
            key: statistics.median(row[key] for row in selected)
            for key in (
                "database_bytes",
                "peak_branch_python_bytes",
                "process_peak_rss_kib",
                "write_seconds",
            )
        } | {"stored_bytes": selected[0]["stats"]["stored_bytes"]}
    result = dict(
        scope=__doc__,
        hardware=platform.uname()._asdict(),
        python=sys.version,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=rows,
        summary=summary,
        all_versions_exact=True,
        all_storage_reclaimed=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
