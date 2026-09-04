#!/usr/bin/env python3
"""Generate deterministic, local image workloads for the MM-DP experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


SIZES = {
    "small": 448,
    "medium": 1024,
    "large": 2048,
    "xlarge": 3072,
}


def make_image(side: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:side, :side]
    base = np.empty((side, side, 3), dtype=np.uint8)
    base[..., 0] = (xx * 255 // max(side - 1, 1) + seed * 17) % 256
    base[..., 1] = (yy * 255 // max(side - 1, 1) + seed * 29) % 256
    base[..., 2] = ((xx + yy) * 127 // max(side - 1, 1) + seed * 43) % 256
    noise = rng.integers(0, 24, size=base.shape, dtype=np.uint8)
    return Image.fromarray(np.add(base, noise, dtype=np.uint8), mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for size_name, side in SIZES.items():
        for variant in range(args.variants):
            path = args.output / f"{size_name}_{side}_v{variant}.jpg"
            make_image(side, seed=variant + side).save(
                path, format="JPEG", quality=92, subsampling=0
            )
            print(path)


if __name__ == "__main__":
    main()
