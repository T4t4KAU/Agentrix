#!/usr/bin/env python3
"""Profile one matched Flash/Fork 16-branch decode operator invocation."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from collections.abc import Callable

import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.backends.fork_attn import (
    _get_adaptive_prefix_chunk_blocks,
    _get_mnw,
    _get_prefix_chunk_blocks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attention-backend", choices=("FLASH_ATTN", "FORK_ATTN"), required=True
    )
    parser.add_argument("--prefix-tokens", type=int, default=8192)
    parser.add_argument("--private-suffix-tokens", type=int, default=128)
    parser.add_argument(
        "--prefix-chunk-tokens",
        type=int,
        default=0,
        help="Fixed chunk size, or zero to use the production adaptive policy.",
    )
    parser.add_argument("--branches", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=5)
    return parser.parse_args()


def pack_boxes(
    boxes: list[tuple[list[int], list[int], int, int]],
    *,
    num_seqs: int,
    hratio: int,
    block_size: int,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[int],
    int,
]:
    device = torch.device("cuda")
    num_split_per_seq = [0] * num_seqs
    grouped = defaultdict(list)
    for box in boxes:
        q_ids, _, rank, kv_len = box
        for q_id in q_ids:
            num_split_per_seq[q_id] = max(num_split_per_seq[q_id], rank + 1)
        grouped[_get_mnw(len(q_ids), hratio, kv_len, block_size)].append(box)

    query_tables = []
    block_tables = []
    num_seqs_per_ctas = []
    cta_ranks = []
    kv_in_ctas = []
    mnw = []
    for tile, group in sorted(grouped.items(), reverse=True):
        max_queries = max(len(item[0]) for item in group)
        max_blocks = max(len(item[1]) for item in group)
        query_tables.append(
            torch.tensor(
                [item[0] + [0] * (max_queries - len(item[0])) for item in group],
                dtype=torch.int32,
                device=device,
            )
        )
        block_tables.append(
            torch.tensor(
                [item[1] + [0] * (max_blocks - len(item[1])) for item in group],
                dtype=torch.int32,
                device=device,
            )
        )
        num_seqs_per_ctas.append(
            torch.tensor(
                [len(item[0]) for item in group],
                dtype=torch.int32,
                device=device,
            )
        )
        cta_ranks.append(
            torch.tensor([item[2] for item in group], dtype=torch.int32, device=device)
        )
        kv_in_ctas.append(
            torch.tensor([item[3] for item in group], dtype=torch.int32, device=device)
        )
        mnw.extend(tile)

    splits = torch.tensor(num_split_per_seq, dtype=torch.int32, device=device)
    return (
        splits,
        query_tables,
        block_tables,
        num_seqs_per_ctas,
        cta_ranks,
        kv_in_ctas,
        mnw,
        max(num_split_per_seq),
    )


def make_inputs(
    args: argparse.Namespace,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[tuple[list[int], list[int], int, int]],
]:
    if args.prefix_tokens % args.block_size:
        raise ValueError("prefix tokens must be block-aligned")
    if args.private_suffix_tokens % args.block_size:
        raise ValueError("private suffix tokens must be block-aligned")
    if args.prefix_chunk_tokens < 0:
        raise ValueError("prefix chunk tokens must be non-negative")
    if args.prefix_chunk_tokens % args.block_size:
        raise ValueError("prefix chunk tokens must be block-aligned")

    torch.manual_seed(2026)
    device = torch.device("cuda")
    dtype = torch.float16
    prefix_blocks = args.prefix_tokens // args.block_size
    suffix_blocks = args.private_suffix_tokens // args.block_size
    if args.prefix_chunk_tokens:
        chunk_blocks = args.prefix_chunk_tokens // args.block_size
    else:
        max_complete_blocks = prefix_blocks + suffix_blocks
        base_chunk_blocks = _get_prefix_chunk_blocks(
            args.block_size, max_complete_blocks
        )
        base_chunks = (prefix_blocks + base_chunk_blocks - 1) // base_chunk_blocks
        target_plan_ctas = (
            2 * torch.cuda.get_device_properties(device).multi_processor_count
            + args.num_kv_heads
            - 1
        ) // args.num_kv_heads
        missing_ctas = max(0, target_plan_ctas - (base_chunks + args.branches))
        target_chunks = min(31, base_chunks + missing_ctas)
        chunk_blocks = _get_adaptive_prefix_chunk_blocks(
            block_size=args.block_size,
            prefix_blocks=prefix_blocks,
            base_chunk_blocks=base_chunk_blocks,
            num_reqs=args.branches,
            num_kv_heads=args.num_kv_heads,
            num_sms=torch.cuda.get_device_properties(device).multi_processor_count,
            prefix_cohorts=1,
            max_prefix_chunks=target_chunks,
        )
    total_blocks = prefix_blocks + args.branches * suffix_blocks

    q = torch.randn(
        args.branches,
        1,
        args.num_heads,
        args.head_dim,
        dtype=dtype,
        device=device,
    )
    k_cache = torch.randn(
        total_blocks,
        args.block_size,
        args.num_kv_heads,
        args.head_dim,
        dtype=dtype,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)

    shared_blocks = list(range(prefix_blocks))
    table_rows = []
    boxes = []
    rank = 0
    for start in range(0, prefix_blocks, chunk_blocks):
        blocks = shared_blocks[start : start + chunk_blocks]
        boxes.append(
            (list(range(args.branches)), blocks, rank, len(blocks) * args.block_size)
        )
        rank += 1
    for branch in range(args.branches):
        start = prefix_blocks + branch * suffix_blocks
        suffix = list(range(start, start + suffix_blocks))
        table_rows.append(shared_blocks + suffix)
        boxes.append(([branch], suffix, rank, args.private_suffix_tokens))

    block_table = torch.tensor(table_rows, dtype=torch.int32, device=device)
    seq_lens = torch.full(
        (args.branches,),
        args.prefix_tokens + args.private_suffix_tokens,
        dtype=torch.int32,
        device=device,
    )
    return q, k_cache, v_cache, block_table, seq_lens, boxes


def make_flash(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[Callable[[], None], torch.Tensor]:
    batch, _, num_heads, head_dim = q.shape
    output = torch.empty((batch, num_heads, head_dim), dtype=q.dtype, device=q.device)
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device)

    def run() -> None:
        flash_attn_varlen_func(
            q=q.view(batch, num_heads, head_dim),
            k=k_cache,
            v=v_cache,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=seq_lens,
            max_seqlen_k=int(seq_lens[0].item()),
            softmax_scale=1.0 / math.sqrt(head_dim),
            causal=True,
            block_table=block_table,
            num_splits=0,
        )

    return run, output


def make_fork(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    boxes: list[tuple[list[int], list[int], int, int]],
    block_size: int,
) -> tuple[Callable[[], None], torch.Tensor]:
    batch, _, num_heads, head_dim = q.shape
    hratio = num_heads // k_cache.shape[2]
    packed = pack_boxes(boxes, num_seqs=batch, hratio=hratio, block_size=block_size)
    splits, q_tables, tables, seqs, ranks, kv_lens, mnw, max_split = packed
    output = torch.empty_like(q)
    softmax_lse = torch.empty(
        (batch, num_heads, 1), dtype=torch.float32, device=q.device
    )
    split_out = torch.empty(
        (batch, num_heads, max_split, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    split_lse = torch.empty(
        (batch, num_heads, max_split), dtype=torch.float32, device=q.device
    )

    def run() -> None:
        ops.fork_attention(
            output,
            softmax_lse,
            split_out,
            split_lse,
            q,
            k_cache,
            v_cache,
            splits,
            q_tables,
            tables,
            seqs,
            ranks,
            kv_lens,
            mnw,
            max_split,
            1.0 / math.sqrt(head_dim),
        )

    return run, output


def main() -> None:
    args = parse_args()
    q, k_cache, v_cache, block_table, seq_lens, boxes = make_inputs(args)
    flash, flash_output = make_flash(q, k_cache, v_cache, block_table, seq_lens)
    fork, fork_output = make_fork(q, k_cache, v_cache, boxes, args.block_size)

    flash()
    fork()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        fork_output, flash_output.view_as(q), atol=2e-2, rtol=2e-2
    )

    target = flash if args.attention_backend == "FLASH_ATTN" else fork
    for _ in range(args.warmups):
        target()
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    target()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(
        "OPERATOR_PROFILE_RESULT "
        f"backend={args.attention_backend} branches={args.branches} "
        f"prefix={args.prefix_tokens} suffix={args.private_suffix_tokens} "
        f"shared_chunks={sum(len(item[0]) > 1 for item in boxes)} "
        f"max_splits={max(item[2] for item in boxes) + 1} "
        f"output_tokens={args.branches}",
        flush=True,
    )


if __name__ == "__main__":
    main()
