#!/usr/bin/env python3
"""Measure tool-data storage and host memory in isolated processes, without a model.

Compare snapshot page sharing, stage reclamation or the repository file-read
tool. Each comparison checks exact retained data or returned output. These
measurements do not evaluate GPU KV occupancy or Agent quality.
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


def report_pages(count, stage=None):
    # Unique deterministic ASCII pages prevent repeated padding from creating
    # an artificial within-snapshot deduplication advantage.
    for index in range(count):
        key = f"tool-report-page-{index}"
        if stage is not None:
            key = f"stage-{stage}/{key}"
        token = hashlib.sha256(key.encode()).hexdigest()
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
    if args.worker in {"read_full", "read_stream"}:
        return read_worker(args, module.RepositoryTools)
    if args.worker in {"session", "stage"}:
        return lifecycle_worker(args, module.PagedToolStore)
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
            peak_stored_bytes=stats["stored_bytes"],
            database_bytes=disk_bytes,
            peak_branch_python_bytes=peak_python_bytes,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            write_seconds=write_seconds,
            verified_sha256=verified,
            final=final,
            final_database_bytes=path.stat().st_size,
        )


def lifecycle_worker(args, store_type):
    """Keep durable state while a delayed branch still consumes prior raw data.

    Raw reports are explicitly intermediate: after the verifier completes, the
    next stage needs only the durable checksum state. Both arms use page sharing
    and identical data/consumers; only the parent's release policy differs.
    """
    page_count = args.size_mib * 256
    chars = page_count * 4096
    with tempfile.TemporaryDirectory(prefix="agentrix-stage-lifetime-") as directory:
        path = Path(directory) / "snapshots.sqlite"
        store = store_type(path)
        store.open_session("workflow")
        config = store.put(
            "workflow", "Verify each complete report; retain the checksum chain."
        )
        state = store.put("workflow", json.dumps(dict(stage=-1, chain="initial")))
        pending = None
        expected_chain = "initial"
        verified, samples = [], []
        write_seconds = 0

        def sample(stage, phase):
            samples.append(
                dict(
                    stage=stage,
                    phase=phase,
                    **store.stats(),
                    database_bytes=path.stat().st_size,
                )
            )

        def consume(session, handle):
            hasher = hashlib.sha256()
            for offset in range(0, chars, 16384):
                hasher.update(
                    store.read(session, handle, offset=offset, limit=16384)[
                        "content"
                    ].encode()
                )
            assert hasher.hexdigest() == handle
            verified.append(handle)

        tracemalloc.start()
        for stage in range(args.stages):
            previous = json.loads(store.read("workflow", state)["content"])
            assert previous == dict(stage=stage - 1, chain=expected_chain)
            oracle = hashlib.sha256()
            for page in report_pages(page_count, stage):
                oracle.update(page.encode())
            began = time.perf_counter()
            report = store.put_stream("workflow", report_pages(page_count, stage))
            assert report == oracle.hexdigest()
            sample(stage, "produced")
            expected_chain = hashlib.sha256(
                (previous["chain"] + report).encode()
            ).hexdigest()
            state = store.put(
                "workflow", json.dumps(dict(stage=stage, chain=expected_chain))
            )
            reviewer = f"review-{stage}"
            store.open_session(reviewer, parent="workflow")
            # The verifier needs one report; its liveness is identical in both arms.
            store.checkpoint_session(reviewer, keep_results=[report])
            sample(stage, "forked")
            if args.worker == "stage":
                store.checkpoint_session("workflow", keep_results=[config, state])
            sample(stage, "checkpointed")
            write_seconds += time.perf_counter() - began
            if pending:
                # The preceding branch outlives its parent's checkpoint. It must
                # still read every byte, and only then release its ownership.
                consume(*pending)
                began = time.perf_counter()
                store.release_session(pending[0])
                write_seconds += time.perf_counter() - began
            sample(stage, "joined_previous")
            pending = (reviewer, report)
        if pending:
            consume(*pending)
            store.release_session(pending[0])
        assert (
            json.loads(store.read("workflow", state)["content"])["chain"]
            == expected_chain
        )
        assert store.read("workflow", config)["content"].startswith("Verify")
        _, peak_python_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stats = store.stats()
        store.release_session("workflow")
        final = store.stats()
        assert final == dict(objects=0, stored_bytes=0, sessions=0, references=0)
        store.close()
        return dict(
            mode=args.worker,
            stages=args.stages,
            snapshot_bytes=chars,
            source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
            stats=stats,
            peak_stored_bytes=max(s["stored_bytes"] for s in samples),
            database_bytes=max(s["database_bytes"] for s in samples),
            peak_branch_python_bytes=peak_python_bytes,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            write_seconds=write_seconds,
            verified_sha256=verified,
            final_chain=expected_chain,
            occupancy_samples=samples,
            final=final,
            final_database_bytes=path.stat().st_size,
        )


def read_worker(args, tools_type):
    """Measure the real repository read tool, keeping its output unchanged."""
    source = args.read_file.resolve()
    if args.page_reads:
        sys.path.insert(0, str(ROOT / "application/src"))
        from agentrix_application import PagedToolStore

    with tempfile.TemporaryDirectory(prefix="agentrix-read-result-") as directory:
        for iteration in range(2):
            store = (
                PagedToolStore(Path(directory) / f"results-{iteration}.sqlite")
                if args.page_reads
                else None
            )
            if store is not None:
                store.open_session("root")
            tools = tools_type(
                source.parent,
                {},
                **({"result_store": store} if args.page_reads else {}),
            )
            if iteration:
                tracemalloc.start()
            try:
                started = time.perf_counter()
                result = tools.read(source.name, args.start_line, args.end_line)
                if iteration:
                    _, peak = tracemalloc.get_traced_memory()
                else:
                    elapsed = time.perf_counter() - started
                    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    first = result
            finally:
                if iteration:
                    tracemalloc.stop()
            assert first["content_sha256"] == result["content_sha256"]
            if store is not None:
                if result["paged"]:
                    handle = json.loads(result["content"])
                    restored = hashlib.sha256()
                    for offset in range(0, handle["total_chars"], 16384):
                        restored.update(
                            store.read(
                                "root", handle["result_id"], offset=offset, limit=16384
                            )["content"].encode()
                        )
                    assert (
                        restored.hexdigest()
                        == result["content_sha256"]
                        == handle["result_id"]
                    )
                store.release_session("root")
                assert store.stats()["stored_bytes"] == 0
                store.close()
    return dict(
        mode=args.worker,
        input_file_bytes=source.stat().st_size,
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        read_seconds_unprofiled=elapsed,
        peak_read_python_bytes=peak,
        process_peak_rss_kib=peak_rss,
        content_sha256=result["content_sha256"],
        returned_sha256=result["returned_sha256"],
        original_bytes=result["original_bytes"],
        returned_bytes=result["returned_bytes"],
        truncated=result["truncated"],
        paged=result["paged"],
    )


def read_ablation(args):
    rows = []
    before = hashlib.sha256()
    with args.read_file.open("rb") as stream:
        while data := stream.read(1 << 20):
            before.update(data)
    for repeat in range(args.repeats):
        modes = ["read_full", "read_stream"]
        for mode in modes if repeat % 2 == 0 else reversed(modes):
            source = (
                args.baseline_tools
                if mode == "read_full"
                else ROOT / "benchmark/src/coding_agent_tools.py"
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                mode,
                "--source",
                str(source),
                "--read-file",
                str(args.read_file.resolve()),
                "--start-line",
                str(args.start_line),
                "--end-line",
                str(args.end_line),
            ]
            if args.page_reads:
                command.append("--page-reads")
            row = json.loads(subprocess.check_output(command, text=True))
            rows.append(row)
            print(
                mode,
                "rss_kib",
                row["process_peak_rss_kib"],
                "python_peak",
                row["peak_read_python_bytes"],
                flush=True,
            )
    after = hashlib.sha256()
    with args.read_file.open("rb") as stream:
        while data := stream.read(1 << 20):
            after.update(data)
    assert before.digest() == after.digest(), "input changed during comparison"
    for key in (
        "content_sha256",
        "returned_sha256",
        "original_bytes",
        "returned_bytes",
        "truncated",
        "paged",
    ):
        assert all(row[key] == rows[0][key] for row in rows), key
    summary = {
        mode: {
            key: statistics.median(row[key] for row in rows if row["mode"] == mode)
            for key in (
                "read_seconds_unprofiled",
                "peak_read_python_bytes",
                "process_peak_rss_kib",
            )
        }
        for mode in ("read_full", "read_stream")
    }
    payload = dict(
        scope="Actual RepositoryTools.read on the same immutable file and line range; no model inference.",
        measurement="RSS and time cover the first, unprofiled read; Python allocations cover a second read with tracemalloc.",
        hardware=platform.uname()._asdict(),
        python=sys.version,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=before.hexdigest(),
        input_file=str(args.read_file.resolve()),
        start_line=args.start_line,
        end_line=args.end_line,
        page_reads=args.page_reads,
        rows=rows,
        summary=summary,
        exact_output_match=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-store", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--size-mib", type=int, default=8)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--read-file",
        type=Path,
        help="Compare the real file-read tool on this existing file.",
    )
    parser.add_argument(
        "--baseline-tools",
        type=Path,
        help="Earlier coding_agent_tools.py for a file-read comparison.",
    )
    parser.add_argument("--start-line", type=int, default=1)
    parser.add_argument("--end-line", type=int, default=32)
    parser.add_argument(
        "--page-reads",
        action="store_true",
        help="Enable tool-result paging and verify complete snapshot recovery.",
    )
    parser.add_argument(
        "--stages",
        type=int,
        default=0,
        help="Compare session-end versus stage-aware reclamation with this many stages.",
    )
    parser.add_argument(
        "--worker",
        choices=["private", "shared", "session", "stage", "read_full", "read_stream"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.size_mib, args.branches, args.repeats) < 1 or args.stages < 0:
        parser.error("size, branches and repeats must be positive")
    if args.worker:
        print(json.dumps(worker(args)))
        return
    if args.read_file:
        if not args.baseline_tools or not args.output or args.stages:
            parser.error(
                "file-read comparison needs --baseline-tools and --output, without --stages"
            )
        read_ablation(args)
        return
    if not args.output or (not args.stages and not args.baseline_store):
        parser.error(
            "--output is required; page-sharing comparison also needs --baseline-store"
        )
    rows = []
    modes = ["session", "stage"] if args.stages else ["private", "shared"]
    for repeat in range(args.repeats):
        order = modes if repeat % 2 == 0 else list(reversed(modes))
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
                "--stages",
                str(args.stages),
            ]
            row = json.loads(subprocess.check_output(command, text=True))
            rows.append(row)
            print(
                mode,
                "peak_stored_bytes",
                row["peak_stored_bytes"],
                "branch_python_peak",
                row["peak_branch_python_bytes"],
                flush=True,
            )
    assert all(row["verified_sha256"] == rows[0]["verified_sha256"] for row in rows)
    summary = {}
    for mode in modes:
        selected = [row for row in rows if row["mode"] == mode]
        summary[mode] = {
            key: statistics.median(row[key] for row in selected)
            for key in (
                "database_bytes",
                "peak_branch_python_bytes",
                "process_peak_rss_kib",
                "write_seconds",
                "peak_stored_bytes",
            )
        } | {"stored_bytes": selected[0]["stats"]["stored_bytes"]}
    result = dict(
        scope=(
            "Scripted tool-data lifecycle: both arms use page sharing; intermediate reports "
            "expire after their delayed verifier finishes, durable state remains. No model inference."
            if args.stages
            else __doc__
        ),
        hardware=platform.uname()._asdict(),
        python=sys.version,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=rows,
        summary=summary,
        all_required_reads_exact=True,
        all_storage_reclaimed=True,
    )
    if not args.stages:
        result["all_versions_exact"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
