#!/usr/bin/env python3
"""Compare official FlashInfer Cascade with the repository's CUDA ForkAttention.

Emit JSON lines to stdout; redirect them to the experiment server. This is a
fixed-plan operator benchmark, not an AgentX or end-to-end serving benchmark.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import flashinfer
import numpy as np
import torch

from vllm.v1.attention.backends.fork_attn import (
    ForkAttentionImpl,
    _build_fork_plan,
    _ForkCUDAGraphWorkspace,
    _get_plan_cudagraph_requirements,
)


def emit(kind, **data):
    print(json.dumps({"kind": kind, **data}), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fork-library",
        type=Path,
        required=True,
        help="Current-source library from build_fork_attention_benchmark.py.",
    )
    parser.add_argument(
        "--prefixes", type=int, nargs="+", default=[1024, 8192, 32768, 65536]
    )
    parser.add_argument("--tails", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--branches", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--calls-per-graph", type=int, default=50)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Emit kernel names separately from timings.",
    )
    args = parser.parse_args()
    positive = [
        *args.tails,
        *args.branches,
        args.groups,
        args.page_size,
        args.heads,
        args.kv_heads,
        args.head_dim,
        args.repeats,
        args.calls_per_graph,
    ]
    if min(positive) <= 0 or min(args.prefixes) < 0:
        parser.error(
            "dimensions and repeat counts must be positive; prefixes may be zero"
        )
    if any(p % args.page_size for p in args.prefixes):
        parser.error("prefixes must be complete pages")
    if args.heads % args.kv_heads or any(b % args.groups for b in args.branches):
        parser.error("head and group counts must divide evenly")
    return args


def telemetry():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,temperature.gpu,"
            "clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def provenance(args):
    root = Path(__file__).resolve().parents[2]
    files = [
        Path(__file__),
        Path(__file__).with_name("build_fork_attention_benchmark.py"),
        args.fork_library,
        root / "vllm/vllm/v1/attention/backends/fork_attn.py",
        root / "vllm/vllm/v1/attention/ops/fork_attention.py",
        root / "vllm/vllm/_C_stable_libtorch.abi3.so",
        Path(flashinfer.__file__).parent / "cascade.py",
    ]
    props = torch.cuda.get_device_properties(0)
    return dict(
        args={**vars(args), "fork_library": str(args.fork_library)},
        torch=torch.__version__,
        cuda=torch.version.cuda,
        flashinfer=flashinfer.__version__,
        triton=importlib.metadata.version("triton"),
        vllm_distribution=importlib.metadata.version("vllm"),
        parent_commit=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        vllm_commit=subprocess.check_output(
            ["git", "-C", str(root / "vllm"), "rev-parse", "HEAD"], text=True
        ).strip(),
        hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        gpu=props.name,
        sm_count=props.multi_processor_count,
        l2_bytes=props.L2_cache_size if hasattr(props, "L2_cache_size") else None,
        telemetry=telemetry(),
        atol=0.002,
        rtol=0.02,
        timing_scope="Fixed-plan CUDA graph, including every attention/merge kernel; excludes planning, H2D and KV writes.",
        cache_condition="Repeated reads of the same physical KV; no forced L2 eviction.",
        fork_admission="Force operator execution, bypass serving prefix/query admission thresholds.",
    )


def make_inputs(args, branches, prefix, tail, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    page = args.page_size
    prefix_pages, tail_pages = prefix // page, math.ceil(tail / page)
    total_pages = args.groups * prefix_pages + branches * tail_pages
    # Every implementation consumes these same fragmented physical pages.
    order = rng.permutation(total_pages)
    shared = [
        order[g * prefix_pages : (g + 1) * prefix_pages].tolist()
        for g in range(args.groups)
    ]
    suffixes = [
        order[
            args.groups * prefix_pages + b * tail_pages : args.groups * prefix_pages
            + (b + 1) * tail_pages
        ].tolist()
        for b in range(branches)
    ]
    rows = [
        shared[b // (branches // args.groups)] + suffixes[b] for b in range(branches)
    ]
    dtype = getattr(torch, args.dtype)
    q = torch.randn(branches, args.heads, args.head_dim, device="cuda", dtype=dtype)
    # vLLM's actual interleaved [page, kv_head, token, K+V] cache layout.
    cache = torch.randn(
        total_pages, args.kv_heads, page, 2 * args.head_dim, device="cuda", dtype=dtype
    )
    k, v = cache.transpose(1, 2).split(args.head_dim, dim=-1)
    return q, cache, k, v, rows, shared, suffixes


def reference(q, k, v, rows, length, ratio):
    outputs = []
    # FP32 grouped attention avoids materializing repeated GQA K/V heads.
    for index, row in enumerate(rows):
        keys = k[row].reshape(-1, k.shape[2], k.shape[3])[:length].float()
        values = v[row].reshape(-1, v.shape[2], v.shape[3])[:length].float()
        query = q[index].float().reshape(k.shape[2], ratio, q.shape[-1])
        scores = torch.einsum("hgd,lhd->hgl", query, keys) * q.shape[-1] ** -0.5
        out = torch.einsum("hgl,lhd->hgd", scores.softmax(-1), values)
        outputs.append(out.reshape_as(q[index]))
    return torch.stack(outputs)


def gpu_int(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def make_flashinfer(args, q, k, v, rows, shared, suffixes, prefix, tail):
    branches = len(rows)
    page = args.page_size
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    qo, indptr, indices, last = [], [], [], []
    if prefix:
        qo.append(gpu_int(range(0, branches + 1, branches // args.groups)))
        indptr.append(
            gpu_int(range(0, args.groups * len(shared[0]) + 1, len(shared[0])))
        )
        indices.append(gpu_int(sum(shared, [])))
        last.append(gpu_int([page] * args.groups))
    qo.append(gpu_int(range(branches + 1)))
    indptr.append(gpu_int(range(0, branches * len(suffixes[0]) + 1, len(suffixes[0]))))
    indices.append(gpu_int(sum(suffixes, [])))
    last.append(gpu_int([(tail - 1) % page + 1] * branches))
    cascade = flashinfer.MultiLevelCascadeAttentionWrapper(
        len(qo),
        workspace,
        "NHD",
        use_cuda_graph=True,
        qo_indptr_buf_arr=qo,
        paged_kv_indptr_buf_arr=indptr,
        paged_kv_indices_buf_arr=indices,
        paged_kv_last_page_len_buf_arr=last,
    )
    torch.cuda.synchronize()
    start = time.perf_counter()
    cascade.plan(
        qo,
        indptr,
        indices,
        last,
        args.heads,
        args.kv_heads,
        args.head_dim,
        page,
        causal=True,
        q_data_type=args.dtype,
        kv_data_type=args.dtype,
    )
    torch.cuda.synchronize()
    cascade_setup_ms = (time.perf_counter() - start) * 1000
    # A non-cascade reference uses the same FlashInfer prefill kernel family.
    full_indptr = gpu_int(range(0, branches * len(rows[0]) + 1, len(rows[0])))
    full_indices = gpu_int(sum(rows, []))
    full_last = gpu_int([(prefix + tail - 1) % page + 1] * branches)
    full_qo = gpu_int(range(branches + 1))
    native = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        qo_indptr_buf=full_qo,
        paged_kv_indptr_buf=full_indptr,
        paged_kv_indices_buf=full_indices,
        paged_kv_last_page_len_buf=full_last,
    )
    native.plan(
        full_qo,
        full_indptr,
        full_indices,
        full_last,
        args.heads,
        args.kv_heads,
        args.head_dim,
        page,
        causal=True,
        q_data_type=args.dtype,
        kv_data_type=args.dtype,
    )
    info = dict(
        cascade_initial_plan_ms_including_jit=cascade_setup_ms,
        cascade_backend=[w._backend for w in cascade._batch_prefill_wrappers],
        native_backend=native._backend,
    )
    return {
        "flashinfer_paged": lambda: native.run(q, (k, v)),
        "flashinfer_cascade": lambda: cascade.run(q, (k, v)),
    }, info


def make_fork(args, q, cache, rows, length):
    branches = len(rows)
    start = time.perf_counter()
    plan = _build_fork_plan(
        query_start_locs=list(range(branches + 1)),
        seq_lens=[length] * branches,
        block_rows=rows,
        num_actual_tokens=branches,
        block_size=args.page_size,
        head_ratio=args.heads // args.kv_heads,
        require_shared=False,
    )
    plan_us = (time.perf_counter() - start) * 1e6
    assert plan is not None
    functions, info = {}, {"fork_initial_cpu_plan_us": plan_us}
    node_uses_triton = (
        args.head_dim == 128
        and args.heads == 4 * args.kv_heads
        and torch.cuda.get_device_capability(q.device) == (12, 0)
    )
    for partition, chunk in (("node", None), ("flatten", 1024)):
        ctas, splits = _get_plan_cudagraph_requirements(
            plan,
            head_ratio=args.heads // args.kv_heads,
            head_dim=args.head_dim,
            block_size=args.page_size,
            flatten_chunk_tokens=chunk,
        )
        # Match the backend's graph buckets instead of inventing a tuned kernel.
        ctas = 1 << (max(2, ctas) - 1).bit_length()
        workspace = _ForkCUDAGraphWorkspace(
            num_heads_q=args.heads,
            num_heads_kv=args.kv_heads,
            head_dim=args.head_dim,
            block_size=args.page_size,
            max_model_len=length,
            max_queries=branches,
            max_ctas=ctas,
            max_splits=splits,
            device=q.device,
            pin_memory=True,
            flatten_chunk_tokens=chunk,
        )
        fields = workspace.pack(
            plan, query_capacity=branches, cta_capacity=ctas, split_capacity=splits
        )
        metadata = (
            SimpleNamespace(fork_flat_metadata=None, **fields)
            if "fork_flat_metadata" not in fields
            else SimpleNamespace(**fields)
        )
        impl = SimpleNamespace(
            head_size=args.head_dim,
            num_heads=args.heads,
            num_kv_heads=args.kv_heads,
            scale=args.head_dim**-0.5,
        )
        output = torch.empty_like(q)

        def run(
            impl=impl,
            metadata=metadata,
            output=output,
            workspace=workspace,
            partition=partition,
        ):
            if partition == "flatten" or node_uses_triton:
                ForkAttentionImpl._forward_fork(
                    impl, q, cache, metadata, output, branches
                )
            else:
                keys, values = cache.transpose(1, 2).split(args.head_dim, dim=-1)
                torch.ops.agentrix_fork_bench.fork_attention(
                    output.unsqueeze(1),
                    metadata.fork_softmax_lse,
                    metadata.fork_split_out,
                    metadata.fork_split_lse,
                    q.unsqueeze(1),
                    keys,
                    values,
                    metadata.fork_num_split_per_seq,
                    metadata.fork_query_tables,
                    metadata.fork_block_tables,
                    metadata.fork_num_seqs_per_ctas,
                    metadata.fork_cta_ranks,
                    metadata.fork_kv_in_ctas,
                    metadata.fork_mnw,
                    metadata.fork_max_split_per_seq,
                    impl.scale,
                )
            return output

        functions["fork_" + partition] = run
        info["fork_" + partition] = dict(
            cta_capacity=ctas,
            split_capacity=splits,
            flatten_chunk_tokens=chunk,
            kernel="triton"
            if partition == "flatten" or node_uses_triton
            else "cuda_cpp",
        )
    return functions, info


def capture(function, count):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(count):
            output = function()
    return graph, output


def validate(actual, expected):
    torch.testing.assert_close(actual.float(), expected, atol=0.002, rtol=0.02)
    delta = actual.float() - expected
    return dict(
        max_abs=delta.abs().max().item(), rmse=delta.square().mean().sqrt().item()
    )


def measure_case(args, branches, prefix, tail, seed):
    shape = dict(
        branches=branches, prefix=prefix, tail=tail, seed=seed, groups=args.groups
    )
    emit("case_start", **shape)
    q, cache, k, v, rows, shared, suffixes = make_inputs(
        args, branches, prefix, tail, seed
    )
    ref = reference(q, k, v, rows, prefix + tail, args.heads // args.kv_heads)
    functions, fi_info = make_flashinfer(
        args, q, k, v, rows, shared, suffixes, prefix, tail
    )
    forks, fork_info = make_fork(args, q, cache, rows, prefix + tail)
    functions.update(forks)
    correctness = {name: validate(fn(), ref) for name, fn in functions.items()}
    graphs = {name: capture(fn, args.calls_per_graph) for name, fn in functions.items()}
    for name, (graph, out) in graphs.items():
        graph.replay()
        torch.cuda.synchronize()
        correctness[name]["graph"] = validate(out, ref)
    # Check that captured graphs consume current Q rather than frozen outputs.
    q.neg_()
    changed_ref = reference(q, k, v, rows, prefix + tail, args.heads // args.kv_heads)
    for name, (graph, out) in graphs.items():
        graph.replay()
        torch.cuda.synchronize()
        correctness[name]["changed_query"] = validate(out, changed_ref)
    q.neg_()
    samples = {name: [] for name in functions}
    before = telemetry()
    names = list(functions)
    for repeat in range(args.repeats):
        # Rotate order so every implementation occupies every position equally.
        order = names[repeat % len(names) :] + names[: repeat % len(names)]
        for name in order:
            graph = graphs[name][0]
            graph.replay()
            torch.cuda.synchronize()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / args.calls_per_graph)
    profile = {}
    if args.profile:
        for name, fn in functions.items():
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                fn()
                torch.cuda.synchronize()
            profile[name] = [
                e.name
                for e in prof.events()
                if e.device_type == torch.autograd.DeviceType.CUDA
            ]
    emit(
        "result",
        **shape,
        correctness=correctness,
        samples_us=samples,
        median_us={name: statistics.median(s) for name, s in samples.items()},
        setup={**fi_info, **fork_info},
        profile=profile,
        telemetry_before=before,
        telemetry_after=telemetry(),
        page_table_sha256=hashlib.sha256(
            np.array(rows, dtype=np.int32).tobytes()
        ).hexdigest(),
        kv_strides=list(k.stride()),
    )


def main():
    args = parse_args()
    torch.cuda.set_device(0)
    torch.ops.load_library(str(args.fork_library.resolve()))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    emit("manifest", **provenance(args))
    cases = list(
        itertools.product(args.branches, args.prefixes, args.tails, args.seeds)
    )
    # Randomize shape order reproducibly; each shape retains its declared seed.
    np.random.default_rng(20260927).shuffle(cases)
    with torch.inference_mode():
        for branches, prefix, tail, seed in cases:
            measure_case(args, branches, prefix, tail, seed)
            gc.collect()
            torch.cuda.empty_cache()
    emit("complete", cases=len(cases))


if __name__ == "__main__":
    main()
