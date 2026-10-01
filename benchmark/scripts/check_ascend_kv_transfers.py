#!/usr/bin/env python3
"""Verify Ascend's installed block-copy helpers using small private buffers.

Checks DMA correctness, not model-state restoration or serving performance.
Run in the matching Ascend virtual environment and save results on the server.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path


def run(args):
    import torch
    import torch_npu  # noqa: F401
    from vllm_ascend.simple_kv_offload.npu_mem_ops import (
        DIRECTION_D2H,
        DIRECTION_H2D,
        build_params,
        copy_blocks,
    )
    from vllm_ascend.utils import enable_custom_op

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"scope": __doc__, "valid": False, "cases": []}

    def wait(event):
        deadline = time.monotonic() + 30
        while not event.query():
            if time.monotonic() >= deadline:
                raise TimeoutError("NPU transfer completion event timed out")
            time.sleep(0.001)

    try:
        torch.npu.set_device(args.device)
        if not enable_custom_op():
            raise RuntimeError("installed Ascend custom operators are unavailable")
        result["versions"] = {
            p: importlib.metadata.version(p)
            for p in ("torch", "torch-npu", "vllm", "vllm-ascend")
        }
        blocks = 16
        layouts = [(256, torch.bfloat16), (512, torch.float32), (1024, torch.uint8)]
        device, host, original = {}, {}, {}
        for i, (width, dtype) in enumerate(layouts):
            key = f"page{i}"
            device[key] = torch.empty((blocks, width), dtype=dtype, device="npu")
            host[key] = torch.empty((blocks, width), dtype=dtype, pin_memory=True)
            original[key] = (
                torch.arange(blocks * width).reshape(blocks, width) % 97
            ).to(dtype)
        result["device_payload_bytes"] = sum(
            t.numel() * t.element_size() for t in device.values()
        )
        store = build_params(device, host, DIRECTION_D2H)
        load = build_params(host, device, DIRECTION_H2D)
        producer, to_host, to_device = (torch.npu.Stream() for _ in range(3))
        for cycle in range(args.cycles):
            for name, sources in [
                ("permuted", [5, 1, 3]),
                ("shared_source", [1, 1, 5]),
                ("single_page", [3]),
                ("empty", []),
            ]:
                cpu_ids = [2, 4, 6][: len(sources)]
                npu_ids = [8, 10, 12][: len(sources)]
                expected_host, golden = {}, {}
                with torch.npu.stream(producer):
                    for key in device:
                        golden[key] = original[key] + cycle
                        device[key].copy_(golden[key])
                        host[key].fill_(201)
                        expected_host[key] = host[key].clone()
                        for src, dst in zip(sources, cpu_ids, strict=True):
                            expected_host[key][dst].copy_(golden[key][src])
                to_host.wait_stream(producer)
                stored = torch.npu.Event()
                with torch.npu.stream(to_host):
                    copy_blocks(sources, cpu_ids, store)
                    stored.record(to_host)
                wait(stored)
                for key in host:
                    if not torch.equal(host[key], expected_host[key]):
                        raise RuntimeError(f"D2H mismatch: {name}/{cycle}/{key}")
                    device[key].fill_(203)
                # The load destination is no longer being written by compute.
                torch.npu.current_stream().synchronize()
                loaded = torch.npu.Event()
                with torch.npu.stream(to_device):
                    copy_blocks(cpu_ids, npu_ids, load)
                    loaded.record(to_device)
                wait(loaded)
                for key in device:
                    expected = torch.full_like(original[key], 203)
                    for src, dst in zip(cpu_ids, npu_ids, strict=True):
                        expected[dst].copy_(expected_host[key][src])
                    if not torch.equal(device[key].cpu(), expected):
                        raise RuntimeError(f"H2D mismatch: {name}/{cycle}/{key}")
                result["cases"].append({"name": name, "cycle": cycle, "valid": True})
        result["valid"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        temporary = args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device < 0 or not 1 <= args.cycles <= 16:
        parser.error("device must be nonnegative; cycles must be in 1..16")
    result = run(args)
    print(json.dumps({"valid": result["valid"], "cases": len(result["cases"])}))


if __name__ == "__main__":
    main()
