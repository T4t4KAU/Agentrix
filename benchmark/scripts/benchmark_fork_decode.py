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

from vllm import _custom_ops as ops


def load_backend(path):
    spec = importlib.util.spec_from_file_location("fork_reference", path)
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


def measure(prefix, branches, modules, trace_dir=None):
    block_size, heads, kv_heads, dim, suffix = 16, 32, 8, 128, 129
    prefix_blocks = prefix // block_size
    private_blocks = math.ceil((suffix + 256) / block_size)
    rows = np.array(
        [
            list(range(prefix_blocks))
            + list(
                range(
                    prefix_blocks + i * private_blocks,
                    prefix_blocks + (i + 1) * private_blocks,
                )
            )
            for i in range(branches)
        ],
        dtype=np.int32,
    )
    kwargs = {
        "query_start_locs": list(range(branches + 1)),
        "seq_lens": [prefix + suffix] * branches,
        "block_rows": rows,
        "num_actual_tokens": branches,
        "block_size": block_size,
        "head_ratio": heads // kv_heads,
        "require_shared": True,
    }
    result = {"prefix": prefix, "branches": branches, "suffix": suffix}
    for label, module in modules.items():
        build = (
            module._ForkPlanCache().build
            if hasattr(module, "_ForkPlanCache")
            else module._build_fork_plan
        )
        for _ in range(8):
            build(**kwargs)
        samples = []
        for _ in range(3):
            for step in range(128):
                args = dict(kwargs, seq_lens=[prefix + suffix + step] * branches)
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
        prefix_blocks + branches * private_blocks,
        2,
        block_size,
        kv_heads,
        dim,
        device="cuda",
        dtype=q.dtype,
    )
    k, v = kv.unbind(1)
    table = torch.tensor(rows, device="cuda")
    lengths = torch.full((branches,), prefix + suffix, dtype=torch.int32, device="cuda")
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
            max_seqlen_k=prefix + suffix,
            softmax_scale=dim**-0.5,
            causal=True,
            block_table=table,
            num_splits=0,
        )

    flash()
    result["flash_gpu"] = graph_time(flash)
    for label, module in modules.items():
        plan = module._build_fork_plan(**kwargs)
        ctas, splits = module._get_plan_cudagraph_requirements(
            plan,
            head_ratio=heads // kv_heads,
            head_dim=dim,
            block_size=block_size,
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
        )
        packed = workspace.pack(
            plan, query_capacity=branches, cta_capacity=ctas, split_capacity=splits
        )
        output = torch.empty_like(q)

        def fork(output=output, packed=packed):
            ops.fork_attention(
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
            trace_dir / f"{label}_p{prefix}_b{branches}.trace.json"
            if trace_dir is not None
            else None
        )
        result[label + "_gpu"] = graph_time(fork, trace)
        result[label + "_gpu"].update(
            cta_capacity=ctas,
            split_capacity=splits,
            max_abs_error=float((output.view_as(flash_out) - flash_out).abs().max()),
            segments=len(plan.segments),
            serving_graph_eligible=graph_eligible,
            serving_cta_capacity=capture_ctas,
            serving_split_capacity=capture_splits,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-backend", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-tokens", type=int)
    parser.add_argument("--target-chunks", type=int)
    parser.add_argument("--m16-tile-tokens", type=int)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if (args.chunk_tokens is None) != (args.target_chunks is None):
        parser.error("--chunk-tokens and --target-chunks must be used together")
    if args.chunk_tokens is not None:
        if min(args.chunk_tokens, args.target_chunks) <= 0:
            parser.error("chunk dimensions must be positive")
        current._prefix_chunk_blocks = lambda block_size, max_blocks: max(
            math.ceil(args.chunk_tokens / block_size),
            math.ceil(max_blocks / args.target_chunks),
        )
    if args.m16_tile_tokens is not None:
        original_mnw = current._get_mnw

        def tuned_mnw(queries, ratio, tokens, dim, block_size):
            return original_mnw(
                queries,
                ratio,
                args.m16_tile_tokens
                if queries * current._kernel_head_ratio(ratio) <= 16
                else tokens,
                dim,
                block_size,
            )

        current._get_mnw = tuned_mnw
    modules = {"current": current}
    if args.reference_backend:
        modules = {"before": load_backend(args.reference_backend), **modules}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        result = {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "dtype": "bfloat16",
            "heads": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "tuning": {k: v for k, v in vars(args).items() if isinstance(v, int)},
            "backend_sha256": {
                k: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                for k, m in modules.items()
            },
            "cases": [],
        }
        for prefix, branches in [(2048, 16), (8192, 16), (16384, 8), (16384, 16)]:
            trace_dir = args.output.parent / args.output.stem if args.profile else None
            if trace_dir is not None:
                trace_dir.mkdir(exist_ok=True)
            case = measure(prefix, branches, modules, trace_dir)
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
