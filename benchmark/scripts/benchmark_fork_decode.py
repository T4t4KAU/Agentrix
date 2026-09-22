#!/usr/bin/env python3
"""Measure production Fork plans and complete CUDA graph attention calls."""

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from vllm.v1.attention.backends import fork_attn as current
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.ops.fork_attention import fork_attention as triton_fork_attention

from vllm import _custom_ops as ops


def load_backend(path, name="fork_reference"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def graph_time(run, trace=None):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(100):
            run()
    samples = []
    for _ in range(9):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 100)
    result = {"median_us": statistics.median(samples), "samples_us": samples}
    if trace is not None:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            graph.replay()
            torch.cuda.synchronize()
        profiler.export_chrome_trace(str(trace))
        kernels = {}
        for event in json.loads(trace.read_text())["traceEvents"]:
            if event.get("cat") == "kernel":
                kernels.setdefault(event["name"], []).append(event["dur"])
        result["kernels"] = {
            name: {"calls": len(values), "mean_us": statistics.mean(values)}
            for name, values in kernels.items()
        }
    return result


def paired_graph_times(runners):
    graphs = {}
    for label, run in runners.items():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(100):
                run()
        graphs[label] = graph
    samples = {label: [] for label in runners}
    for repeat in range(9):
        order = list(graphs) if repeat % 2 == 0 else list(reversed(graphs))
        for label in order:
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            graphs[label].replay()
            end.record()
            end.synchronize()
            samples[label].append(start.elapsed_time(end) * 1000 / 100)
    return {
        label: {"median_us": statistics.median(values), "samples_us": values}
        for label, values in samples.items()
    }


def measure(
    prefix,
    branches,
    modules,
    trace_dir=None,
    *,
    suffix=129,
    kernel="auto",
    metadata_mode="graph",
    flatten_chunk_tokens=None,
    reference_flatten_chunk_tokens=None,
    flatten_block_n=128,
    flatten_warps=4,
    tree=False,
    ragged=False,
):
    block_size, heads, kv_heads, dim = 16, 32, 8, 128
    attention = (
        triton_fork_attention
        if kernel == "triton"
        or (kernel == "auto" and torch.cuda.get_device_capability() == (12, 0))
        else ops.fork_attention
    )
    prefix_blocks = prefix // block_size
    private_blocks = math.ceil((suffix + 256) / block_size)
    quad_pages = math.ceil(branches / 4) * 4 if tree else 0
    pair_pages = math.ceil(branches / 2) * 4 if tree else 0
    private_base = prefix_blocks + quad_pages + pair_pages
    context = prefix + (128 if tree else 0)
    lengths_cpu = [
        context + (1 + i * (suffix - 1) // (branches - 1) if ragged else suffix)
        for i in range(branches)
    ]
    rows = np.array(
        [
            list(range(prefix_blocks))
            + (
                list(
                    range(
                        prefix_blocks + (i // 4) * 4, prefix_blocks + (i // 4 + 1) * 4
                    )
                )
                + list(
                    range(
                        prefix_blocks + quad_pages + (i // 2) * 4,
                        prefix_blocks + quad_pages + (i // 2 + 1) * 4,
                    )
                )
                if tree
                else []
            )
            + list(
                range(
                    private_base + i * private_blocks,
                    private_base + (i + 1) * private_blocks,
                )
            )
            for i in range(branches)
        ],
        dtype=np.int32,
    )
    kwargs = {
        "query_start_locs": list(range(branches + 1)),
        "seq_lens": lengths_cpu,
        "block_rows": rows,
        "num_actual_tokens": branches,
        "block_size": block_size,
        "head_ratio": heads // kv_heads,
        "require_shared": True,
    }
    result = {
        "prefix": prefix,
        "branches": branches,
        "suffix": suffix,
        "tree": tree,
        "ragged": ragged,
        "seq_lens": lengths_cpu,
    }
    for label, module in modules.items():
        cache = module._ForkPlanCache() if hasattr(module, "_ForkPlanCache") else None

        def build(cache=cache, module=module, **options):
            if cache is None:
                return module._build_fork_plan(**options)
            return cache.build(
                seq_lens=options["seq_lens"],
                block_rows=options["block_rows"],
                block_row_indices=np.arange(branches, dtype=np.int32),
                block_size=block_size,
                head_ratio=heads // kv_heads,
            )

        for _ in range(8):
            build(**kwargs)
        samples = []
        for _ in range(3):
            for step in range(128):
                args = dict(kwargs, seq_lens=[length + step for length in lengths_cpu])
                start = time.perf_counter_ns()
                plan = build(**args)
                samples.append((time.perf_counter_ns() - start) / 1000)
                assert plan is not None
        result[label + "_plan"] = {
            "median_us": statistics.median(samples),
            "mean_us": statistics.mean(samples),
            "p95_us": float(np.percentile(samples, 95)),
            "samples_us": samples,
        }

    torch.manual_seed(20260905)
    q = torch.randn(branches, 1, heads, dim, device="cuda", dtype=torch.bfloat16)
    # Match the production interleaved K/V cache strides.
    kv = torch.randn(
        private_base + branches * private_blocks,
        2,
        block_size,
        kv_heads,
        dim,
        device="cuda",
        dtype=q.dtype,
    )
    k, v = kv.unbind(1)
    table = torch.tensor(rows, device="cuda")
    lengths = torch.tensor(lengths_cpu, dtype=torch.int32, device="cuda")
    starts = torch.arange(branches + 1, dtype=torch.int32, device="cuda")
    flash_out = torch.empty_like(q.view(branches, heads, dim))

    def flash():
        flash_attn_varlen_func(
            q=q.view_as(flash_out),
            k=k,
            v=v,
            out=flash_out,
            cu_seqlens_q=starts,
            max_seqlen_q=1,
            seqused_k=lengths,
            max_seqlen_k=context + suffix,
            softmax_scale=dim**-0.5,
            causal=True,
            block_table=table,
            num_splits=0,
        )

    flash()
    result["flash_gpu"] = graph_time(flash)
    runners, workspaces = {}, []
    for label, module in modules.items():
        plan = module._build_fork_plan(**kwargs)
        flat_chunk = (
            flatten_chunk_tokens
            if label == "current"
            else reference_flatten_chunk_tokens
        )
        flat_options = {"flatten_chunk_tokens": flat_chunk} if flat_chunk else {}
        ctas, splits = module._get_plan_cudagraph_requirements(
            plan,
            head_ratio=heads // kv_heads,
            head_dim=dim,
            block_size=block_size,
            **flat_options,
        )
        ctas = 2 ** math.ceil(math.log2(max(2, ctas)))
        capture_splits = min(
            32,
            2
            ** math.ceil(
                math.log2(
                    math.ceil(2048 / module._prefix_chunk_blocks(block_size, 2048)) + 2
                )
            ),
        )
        capacity_fn = module._fork_cudagraph_cta_capacity
        capture_ctas = (
            capacity_fn(branches, capture_splits)
            if len(inspect.signature(capacity_fn).parameters) > 1
            else capacity_fn(branches)
        )
        graph_eligible = capture_ctas >= ctas and capture_splits >= splits
        ctas, splits = max(ctas, capture_ctas), max(splits, capture_splits)
        workspace = module._ForkCUDAGraphWorkspace(
            num_heads_q=heads,
            num_heads_kv=kv_heads,
            head_dim=dim,
            block_size=block_size,
            max_model_len=32768,
            max_queries=branches,
            max_ctas=ctas,
            max_splits=splits,
            device=torch.device("cuda"),
            pin_memory=True,
            **flat_options,
        )
        packed = workspace.pack(
            plan, query_capacity=branches, cta_capacity=ctas, split_capacity=splits
        )
        if metadata_mode == "eager" and flat_chunk:
            flat = module._flatten_fork_plan(plan, block_size, flat_chunk, 8)
            flat_ctas = 1 << (len(flat.segments) - 1).bit_length()
            flat_splits = 1 << (max(flat.num_splits_per_query) - 1).bit_length()
            workspace = module._ForkFlattenWorkspace(
                num_heads_q=heads,
                num_heads_kv=kv_heads,
                head_dim=dim,
                block_size=block_size,
                chunk_tokens=flat.chunk_tokens,
                target_chunk_tokens=flat_chunk,
                max_queries=branches,
                max_ctas=flat_ctas,
                max_splits=flat_splits,
                device=q.device,
                pin_memory=True,
            )
            packed = workspace.pack(
                plan,
                query_capacity=branches,
                cta_capacity=flat_ctas,
                split_capacity=flat_splits,
            )
        elif metadata_mode == "eager":
            workspace = module._ForkBufferPool(
                num_heads_q=heads,
                num_heads_kv=kv_heads,
                head_dim=dim,
                device=q.device,
                pin_memory=True,
            )
            packed = module._pack_fork_plan(
                plan,
                num_heads_q=heads,
                num_heads_kv=kv_heads,
                head_dim=dim,
                page_block_size=block_size,
                device=q.device,
                buffer_pool=workspace,
            )
        workspaces.append(workspace)
        output = torch.empty_like(q)

        flat_kernel_options = (
            {"block_n": flatten_block_n, "num_warps": flatten_warps}
            if label == "current"
            else {}
        )

        def fork(
            output=output,
            packed=packed,
            module=module,
            flat_kernel_options=flat_kernel_options,
        ):
            if packed.get("fork_flat_metadata") is not None:
                module.fork_flatten_attention(
                    output,
                    packed["fork_softmax_lse"],
                    packed["fork_split_out"],
                    packed["fork_split_lse"],
                    q,
                    k,
                    v,
                    packed["fork_num_split_per_seq"],
                    packed["fork_flat_metadata"],
                    packed["fork_flat_chunk_tokens"],
                    packed["fork_max_split_per_seq"],
                    dim**-0.5,
                    **flat_kernel_options,
                )
                return
            attention(
                output,
                packed["fork_softmax_lse"],
                packed["fork_split_out"],
                packed["fork_split_lse"],
                q,
                k,
                v,
                packed["fork_num_split_per_seq"],
                packed["fork_query_tables"],
                packed["fork_block_tables"],
                packed["fork_num_seqs_per_ctas"],
                packed["fork_cta_ranks"],
                packed["fork_kv_in_ctas"],
                packed["fork_mnw"],
                packed["fork_max_split_per_seq"],
                dim**-0.5,
            )

        fork()
        torch.testing.assert_close(
            output.view_as(flash_out), flash_out, atol=2e-2, rtol=2e-2
        )
        trace = (
            trace_dir / f"{label}_p{prefix}_b{branches}_s{suffix}.trace.json"
            if trace_dir is not None
            else None
        )
        runners[label] = fork
        result[label + "_gpu"] = graph_time(fork, trace) if trace else {}
        result[label + "_gpu"].update(
            cta_capacity=ctas,
            split_capacity=splits,
            max_abs_error=float((output.view_as(flash_out) - flash_out).abs().max()),
            segments=int((packed["fork_flat_metadata"][:, 16] > 0).sum())
            if flat_chunk
            else sum(
                int((counts > 0).sum()) for counts in packed["fork_num_seqs_per_ctas"]
            ),
            active_splits=int(packed["fork_num_split_per_seq"].sum()),
            metadata_mode=metadata_mode,
            split_workspace_bytes=packed["fork_split_out"].untyped_storage().nbytes()
            + packed["fork_split_lse"].untyped_storage().nbytes(),
            serving_graph_eligible=graph_eligible,
            serving_cta_capacity=capture_ctas,
            serving_split_capacity=capture_splits,
        )
    for label, timing in paired_graph_times(runners).items():
        result[label + "_gpu"].update(timing)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-backend", type=Path)
    parser.add_argument("--reference-ops", type=Path)
    parser.add_argument("--reference-flatten-chunk-tokens", type=int)
    parser.add_argument("--candidate-backend", type=Path)
    parser.add_argument("--candidate-ops", type=Path)
    parser.add_argument("--flatten-chunk-tokens", type=int)
    parser.add_argument("--flatten-block-n", type=int, choices=[64, 128], default=128)
    parser.add_argument("--flatten-warps", type=int, choices=[2, 4], default=4)
    parser.add_argument(
        "--tree", action="store_true", help="Add shared 64-token quartet and pair nodes"
    )
    parser.add_argument(
        "--ragged",
        action="store_true",
        help="Spread branch tails from one token to --suffix-tokens, inclusive",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-tokens", type=int)
    parser.add_argument("--target-chunks", type=int)
    parser.add_argument("--m16-tile-tokens", type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--prefix-tokens", type=int, nargs="+", default=[16384])
    parser.add_argument("--branches", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--suffix-tokens", type=int, nargs="+", default=[17, 129, 511])
    parser.add_argument("--kernel", choices=["auto", "cuda", "triton"], default="auto")
    parser.add_argument("--metadata-mode", choices=["graph", "eager"], default="graph")
    args = parser.parse_args()
    if args.candidate_ops:
        load_backend(args.candidate_ops, "vllm.v1.attention.ops.fork_attention")
    if args.flatten_chunk_tokens is not None and args.flatten_chunk_tokens < 128:
        parser.error("Flatten chunks must contain at least 128 tokens")
    candidate = (
        load_backend(args.candidate_backend, "fork_candidate")
        if args.candidate_backend
        else current
    )
    if any(p < 0 or p % 16 for p in args.prefix_tokens):
        parser.error("prefix lengths must be nonnegative multiples of 16")
    if min(args.branches) < 2 or min(args.suffix_tokens) < 1:
        parser.error("at least two branches and a nonempty suffix are required")
    if (
        max(args.prefix_tokens)
        + max(args.suffix_tokens)
        + 256
        + (128 if args.tree else 0)
        > 32768
    ):
        parser.error("prefix, suffix and 256 growth tokens must fit in 32768 tokens")
    if (args.chunk_tokens is None) != (args.target_chunks is None):
        parser.error("--chunk-tokens and --target-chunks must be used together")
    if args.chunk_tokens is not None:
        if min(args.chunk_tokens, args.target_chunks) <= 0:
            parser.error("chunk dimensions must be positive")
        candidate._prefix_chunk_blocks = lambda block_size, max_blocks: max(
            math.ceil(args.chunk_tokens / block_size),
            math.ceil(max_blocks / args.target_chunks),
        )
    if args.m16_tile_tokens is not None:
        original_mnw = candidate._get_mnw

        def tuned_mnw(queries, ratio, tokens, dim, block_size):
            return original_mnw(
                queries,
                ratio,
                args.m16_tile_tokens
                if queries * candidate._kernel_head_ratio(ratio) <= 16
                else tokens,
                dim,
                block_size,
            )

        candidate._get_mnw = tuned_mnw
    modules = {"current": candidate}
    if args.reference_backend:
        modules = {"before": load_backend(args.reference_backend), **modules}
        if args.reference_ops:
            reference_ops = load_backend(args.reference_ops, "fork_reference_ops")
            modules[
                "before"
            ].fork_flatten_attention = reference_ops.fork_flatten_attention
    elif args.reference_ops or args.reference_flatten_chunk_tokens:
        parser.error("reference Flatten options require --reference-backend")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        result = {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "dtype": "bfloat16",
            "heads": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "kernel": args.kernel,
            "metadata_mode": args.metadata_mode,
            "plan_timing_scope": (
                "forest construction only; excludes flattening and metadata packing"
            ),
            "tuning": {k: v for k, v in vars(args).items() if isinstance(v, int)},
            "backend_sha256": {
                k: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                for k, m in modules.items()
            },
            "cases": [],
        }
        cases = (
            (prefix, branches, suffix)
            for prefix in args.prefix_tokens
            for branches in args.branches
            for suffix in args.suffix_tokens
        )
        for prefix, branches, suffix in cases:
            trace_dir = args.output.parent / args.output.stem if args.profile else None
            if trace_dir is not None:
                trace_dir.mkdir(exist_ok=True)
            case = measure(
                prefix,
                branches,
                modules,
                trace_dir,
                suffix=suffix,
                kernel=args.kernel,
                metadata_mode=args.metadata_mode,
                flatten_chunk_tokens=args.flatten_chunk_tokens,
                reference_flatten_chunk_tokens=args.reference_flatten_chunk_tokens,
                flatten_block_n=args.flatten_block_n,
                flatten_warps=args.flatten_warps,
                tree=args.tree,
                ragged=args.ragged,
            )
            result["cases"].append(case)
            print(
                json.dumps(
                    {
                        k: (
                            {a: b for a, b in v.items() if not a.startswith("samples")}
                            if isinstance(v, dict)
                            else v
                        )
                        for k, v in case.items()
                    }
                ),
                flush=True,
            )
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
