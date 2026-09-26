"""Fixed-plan NPU ForkAttention microbenchmark; not an AgentX score.

Run with the Ascend environment and these operator files deployed to its plugin.
All output, profiler traces and compiler caches belong on the experiment server.
"""

import argparse
import hashlib
import json
import math
import statistics
import time
from importlib.metadata import version
from itertools import permutations
from pathlib import Path

import torch
import torch_npu
import numpy as np

import vllm_ascend
from vllm_ascend.attention.fork_plan import MAX_PREFIX_SPLITS, build_fork_plan
from vllm_ascend.ops.fork_attention import MAX_COHORT_QUERIES, ForkAttention


def csv_ints(value):
    values = [int(part) for part in value.split(",")]
    if not values or any(part <= 0 for part in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def capture(fn):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = fn()
    return graph, output


def time_graph(graph, iterations):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def profile_graphs(graphs, directory):
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(directory),
        record_shapes=True,
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=1, active=len(graphs), repeat=1
        ),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.MemoryAccess,
            l2_cache=True,
        ),
    ) as profiler:
        profiler.step()
        for name, graph in graphs:
            with torch.profiler.record_function(name):
                for _ in range(5):
                    graph.replay()
                torch.npu.synchronize()
            profiler.step()


def make_case(batch, prefix, tail, block, device):
    if prefix % block:
        raise ValueError("prefix must consist of complete cache blocks")
    prefix_pages, tail_pages = prefix // block, math.ceil(tail / block)
    num_pages = prefix_pages + batch * tail_pages
    permutation = torch.randperm(num_pages).tolist()
    rows = [
        permutation[:prefix_pages]
        + permutation[
            prefix_pages + i * tail_pages : prefix_pages + (i + 1) * tail_pages
        ]
        for i in range(batch)
    ]
    # Qwen3.5-9B's actual TP1 full-attention geometry.
    q = torch.randn(batch, 16, 256, device=device, dtype=torch.bfloat16)
    k = torch.randn(num_pages, block, 4, 256, device=device, dtype=q.dtype)
    v = torch.randn_like(k)
    table = torch.tensor(rows, dtype=torch.int32, device=device)
    return q, k, v, table, np.asarray(rows, dtype=np.int32), [prefix + tail] * batch


def baseline_call(q, k, v, table, lengths, block):
    output = torch.empty_like(q)
    lse = torch.empty(1, dtype=q.dtype, device=q.device)
    mask = torch.ones((2048, 2048), dtype=torch.bool, device=q.device).triu(1)
    key, value = k.flatten(2), v.flatten(2)
    kwargs = dict(
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[2],
        input_layout="TND",
        block_size=block,
        scale=q.shape[-1] ** -0.5,
        sparse_mode=3,
        atten_mask=mask,
        block_table=table,
        actual_seq_lengths=list(range(1, q.shape[0] + 1)),
        actual_seq_lengths_kv=lengths,
    )
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
        q, key, value, **kwargs
    )

    def run():
        torch_npu.npu_fused_infer_attention_score.out(
            q, key, value, **kwargs, workspace=workspace, out=[output, lse]
        )
        return output

    return run


def run_case(args, batch, prefix, tail):
    q, k, v, table, rows, lengths = make_case(
        batch, prefix, tail, args.block_size, f"npu:{args.device}"
    )
    baseline = baseline_call(q, k, v, table, lengths, args.block_size)
    reference = baseline().clone()
    baseline_graph, baseline_output = capture(baseline)
    candidates = []
    for splits in args.splits:
        plan_start = time.perf_counter_ns()
        plan = build_fork_plan(
            rows,
            lengths,
            args.block_size,
            prefix_splits=splits,
            min_shared_tokens=args.block_size,
        )
        plan_us = (time.perf_counter_ns() - plan_start) / 1000
        op = ForkAttention(q, k, v, plan)
        result = op(q)
        torch.testing.assert_close(
            result.float(), reference.float(), atol=2e-3, rtol=2e-2
        )
        diff = float((result.float() - reference.float()).abs().max())
        graph, output = capture(lambda op=op: op(q))
        control_plan = build_fork_plan(
            rows,
            lengths,
            args.block_size,
            prefix_splits=splits,
            min_shared_tokens=args.block_size,
            share_prefix_queries=False,
        )
        control = ForkAttention(q, k, v, control_plan)
        torch.testing.assert_close(
            control(q).float(), reference.float(), atol=2e-3, rtol=2e-2
        )
        control_graph, control_output = capture(lambda control=control: control(q))
        samples = {"baseline_us": [], "fork_us": [], "split_only_us": []}
        orders = list(
            permutations(
                [
                    ("baseline_us", baseline_graph),
                    ("fork_us", graph),
                    ("split_only_us", control_graph),
                ]
            )
        )
        for repeat in range(args.repeats):
            ordered = orders[repeat % len(orders)]
            for name, current_graph in ordered:
                samples[name].append(time_graph(current_graph, args.iterations))
        base_us, fork_us = (
            statistics.median(samples[name]) for name in ("baseline_us", "fork_us")
        )
        row = dict(
            splits=plan.prefix_splits,
            baseline_us=base_us,
            fork_us=fork_us,
            speedup=base_us / fork_us,
            split_only_us=statistics.median(samples["split_only_us"]),
            max_abs_diff=diff,
            samples=samples,
            plan_us=plan_us,
            explicit_workspace_bytes=sum(
                t.numel() * t.element_size()
                for t in (
                    op.packed_query,
                    op.partial_output,
                    op.partial_lse,
                    op.block_table,
                    op.workspace,
                )
            ),
        )
        candidates.append(row)
        print(
            json.dumps(dict(branches=batch, prefix=prefix, tail=tail, **row)),
            flush=True,
        )
        if args.profile_dir:
            profile_graphs(orders[0], args.profile_dir)
        del orders, ordered, graph, output, op, control_graph, control_output, control
    del baseline_graph, baseline_output
    return dict(branches=batch, prefix=prefix, tail=tail, candidates=candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--branches", type=csv_ints, default=[2, 3, 4, 8])
    parser.add_argument("--prefixes", type=csv_ints, default=[8192, 32768, 65536])
    parser.add_argument("--tails", type=csv_ints, default=[128, 1024])
    parser.add_argument("--splits", type=csv_ints, default=[1, 4, 8, 16, 32])
    # The hybrid manager uses 1024-token blocks, but FIA receives 128-token
    # kernel pages after the runner's physical-to-logical block conversion.
    parser.add_argument("--block-size", type=int, choices=[128], default=128)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument(
        "--profile-dir", help="server directory; requires one shape and split count"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new server artifact path")
    if (
        min(args.block_size, args.repeats, args.iterations) <= 0
        or min(args.branches) < 2
        or max(args.branches) > MAX_COHORT_QUERIES
        or max(args.splits) > MAX_PREFIX_SPLITS
    ):
        parser.error(
            "requires positive repetitions, 2 to 8 branches and 1 to 32 splits"
        )
    if any(prefix % args.block_size for prefix in args.prefixes):
        parser.error("prefixes must consist of complete cache blocks")
    if args.profile_dir and any(
        len(items) != 1
        for items in (args.branches, args.prefixes, args.tails, args.splits)
    ):
        parser.error(
            "profiling requires exactly one branch count, prefix, tail and split count"
        )
    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    source = Path(vllm_ascend.__file__).parent
    hashes = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in [
            "attention/fork_plan.py",
            "ops/fork_attention.py",
            "ops/triton/fork_attention.py",
        ]
    }
    report = dict(
        kind="operator_microbenchmark_not_agentx",
        configuration={**vars(args), "output": str(args.output)},
        versions={
            name: version(name)
            for name in ("torch", "torch-npu", "triton-ascend", "vllm", "vllm-ascend")
        },
        plugin_source=str(source),
        source_sha256=hashes,
        baseline="paged FIA TND sparse_mode=3, matching the selected full-attention path",
        timing="NPU events around repeated fixed-plan graph replay; excludes host planning and graph task updates",
        cases=[],
        complete=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in args.branches:
        for prefix in args.prefixes:
            for tail in args.tails:
                report["cases"].append(run_case(args, batch, prefix, tail))
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
