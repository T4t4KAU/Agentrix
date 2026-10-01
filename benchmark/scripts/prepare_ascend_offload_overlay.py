#!/usr/bin/env python3
"""Backport upstream APIs and async-load scheduling into an isolated canonical-handler runtime.

The installed vLLM package is never modified. This deliberately accepts only
the older source shapes tested by the Ascend experiment, failing on drift.
Newer runtimes already provide these APIs and must not use this helper.
"""

import argparse
from pathlib import Path


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("Unsupported source revision; expected one patch anchor")
    return source.replace(old, new, 1)


def prepare(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(target)
    patches = {}
    scheduler = Path("distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py")
    text = (source / scheduler).read_text()
    if "max_offload_tokens" in text:
        raise ValueError("This runtime already implements the admission hint")
    text = replace_once(
        text,
        "    # number of hits in the GPU cache\n",
        "    max_offload_tokens: int | None = None\n"
        "    # number of hits in the GPU cache\n",
    )
    text = replace_once(
        text,
        "    def update_offload_keys(self) -> None:\n",
        "        params = self.req.kv_transfer_params\n"
        '        raw = params.get("max_offload_tokens") if params else None\n'
        "        if type(raw) is int and raw >= 0:\n"
        "            self.max_offload_tokens = raw\n"
        "        elif raw is not None:\n"
        '            logger.warning("max_offload_tokens must be a non-negative int, got %r; ignoring", raw)\n'
        "\n"
        "    def update_offload_keys(self) -> None:\n",
    )
    text = replace_once(
        text,
        "            num_offloadable_tokens = min(num_tokens_after_batch, req.num_tokens)\n",
        "            num_offloadable_tokens = min(num_tokens_after_batch, req.num_tokens)\n"
        "            if req_status.max_offload_tokens is not None:\n"
        "                num_offloadable_tokens = min(\n"
        "                    num_offloadable_tokens, req_status.max_offload_tokens\n"
        "                )\n",
    )
    text = replace_once(
        text,
        "        self.config = SchedulerOffloadConfig.from_spec(spec)\n",
        "        self.config = SchedulerOffloadConfig.from_spec(spec)\n"
        "        self._mamba_align_size = None\n"
        "        for idx, block_size in enumerate(spec.gpu_block_size):\n"
        "            kv_spec = spec.kv_cache_config.kv_cache_groups[idx].kv_cache_spec\n"
        '            if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode in ("align", "all"):\n'
        "                size = block_size * spec.block_size_factor\n"
        "                assert self._mamba_align_size is None or self._mamba_align_size == size\n"
        "                self._mamba_align_size = size\n",
    )
    text = replace_once(
        text,
        "            max_hit_size_tokens -= 1\n",
        "            max_hit_size_tokens -= 1\n"
        "            if self._mamba_align_size is not None:\n"
        "                max_hit_size_tokens = (\n"
        "                    max_hit_size_tokens // self._mamba_align_size\n"
        "                    * self._mamba_align_size\n"
        "                )\n",
    )
    patches[scheduler] = text
    core = Path("v1/core/sched/scheduler.py")
    # Async restore schedules zero compute tokens. Alignment must not prevent
    # allocation and submission of the external transfer in that case.
    patches[core] = replace_once(
        (source / core).read_text(),
        "                if self.need_mamba_block_aligned_split:\n",
        "                if self.need_mamba_block_aligned_split and not load_kv_async:\n",
    )
    route = Path("entrypoints/serve/cache/api_router.py")
    patches[route] = replace_once(
        (source / route).read_text(),
        "    await engine_client(raw_request).reset_prefix_cache(\n"
        "        reset_running_requests, reset_external\n"
        "    )\n"
        "    return Response(status_code=200)",
        "    success = await engine_client(raw_request).reset_prefix_cache(\n"
        "        reset_running_requests, reset_external\n"
        "    )\n"
        '    return {"success": success}',
    )

    # Only ancestors of patched modules need real directories. Other package
    # entries remain symlinks to the installed, pinned source and extensions.
    def populate(src: Path, dst: Path, relative: Path) -> None:
        dst.mkdir()
        for child in src.iterdir():
            if child.name == "__pycache__":
                continue
            rel = relative / child.name
            out = dst / child.name
            if rel in patches:
                out.write_text(patches[rel])
            elif child.is_dir() and any(rel in p.parents for p in patches):
                populate(child, out, rel)
            else:
                out.symlink_to(child.resolve())

    populate(source.resolve(), target, Path())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.target)
