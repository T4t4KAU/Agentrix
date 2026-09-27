#!/usr/bin/env python3
"""Build the current CUDA ForkAttention sources as an isolated benchmark op.

Uses the same template instantiations as vLLM CMake, without rebuilding unrelated
operators or replacing the user's installed extension. Build products live in
the normal local compiler cache; redirect the build log to the experiment server.
"""

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def main():
    root = Path(__file__).resolve().parents[2] / "vllm"
    src = root / "csrc/libtorch_stable/attention/fork"
    files = sorted(src.iterdir()) + [
        root / "csrc/libtorch_stable/ops.h",
        root / "csrc/libtorch_stable/torch_utils.h",
    ]
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()
    }
    major, minor = torch.cuda.get_device_capability()
    identity = json.dumps([hashes, torch.__version__, major, minor], sort_keys=True)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    build = Path.home() / ".cache/agentrix-build" / ("fork-" + digest)
    build.mkdir(parents=True, exist_ok=True)
    bindings = build / "bindings.cpp"
    bindings.write_text("""#include "ops.h"
#include <torch/csrc/stable/library.h>
STABLE_TORCH_LIBRARY_FRAGMENT(agentrix_fork_bench, ops) {
  ops.def("fork_attention(Tensor! out, Tensor! softmax_lse, Tensor! split_out, "
          "Tensor! split_lse, Tensor q, Tensor k_cache, Tensor v_cache, "
          "Tensor num_split_per_seq, Tensor[] query_tables, Tensor[] block_tables, "
          "Tensor[] num_seqs_per_ctas, Tensor[] cta_ranks, Tensor[] kv_in_ctas, "
          "int[] mnw, int max_split_per_seq, float softmax_scale) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(agentrix_fork_bench, CUDA, ops) {
  ops.impl("fork_attention", TORCH_BOX(&vllm::fork_attention::fork_attention));
}
""")
    sources = [bindings, src / "fork_attention.cu"]
    template = (src / "fork_fwd_instantiation.cu.in").read_text()
    for dim in (64, 128, 256):
        sources.append(src / f"fork_fwd_hdim{dim}.cu")
        for dtype, cpp_type in (
            ("fp16", "cute::half_t"),
            ("bf16", "cutlass::bfloat16_t"),
        ):
            for ratio in (1, 2, 4):
                generated = build / f"fork_{dim}_{dtype}_r{ratio}.cu"
                generated.write_text(
                    template.replace("@FORK_CPP_DTYPE@", cpp_type)
                    .replace("@FORK_HEAD_DIM@", str(dim))
                    .replace("@FORK_HEAD_RATIO@", str(ratio))
                )
                sources.append(generated)
    flags = [
        "-O3",
        "-std=c++17",
        "-DUSE_CUDA",
        "-DVLLM_ENABLE_FORK_ATTENTION",
        "-DFORK_NAMESPACE=agentrix_fork_bench_kernel",
    ]
    cuda_flags = flags + [
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}",
    ]
    print(
        json.dumps(
            {
                "kind": "build_start",
                "directory": str(build),
                "hashes": hashes,
                "cuda_flags": cuda_flags,
                "torch": torch.__version__,
            }
        ),
        flush=True,
    )
    library = load(
        name="agentrix_fork_benchmark",
        sources=[str(p) for p in sources],
        extra_include_paths=[
            str(src),
            str(root / "csrc/libtorch_stable"),
            str(root / "csrc"),
            str(root / ".deps/cutlass-src/include"),
        ],
        extra_cflags=flags,
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=["-Wl,-Bsymbolic"],
        build_directory=str(build),
        is_python_module=False,
        verbose=True,
    )
    print(
        json.dumps(
            {
                "kind": "build_complete",
                "library": str(library),
                "sha256": hashlib.sha256(Path(library).read_bytes()).hexdigest(),
                "source_hashes": hashes,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
