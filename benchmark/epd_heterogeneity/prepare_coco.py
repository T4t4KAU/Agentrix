from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps


@dataclass(frozen=True)
class ImageBucket:
    name: str
    width: int
    height: int

    @property
    def expected_visual_tokens(self) -> int:
        patch_size = 14
        merge_size = 2
        return (self.width // patch_size) * (self.height // patch_size) // (
            merge_size**2
        )


BUCKETS = (
    ImageBucket("low", 448, 448),
    ImageBucket("medium", 896, 896),
    ImageBucket("1080p_class", 1792, 1008),
    ImageBucket("4k_class", 3584, 2016),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    return parser.parse_args()


def select_members(
    archive: zipfile.ZipFile, samples: int, seed: int
) -> list[str]:
    members = sorted(
        name
        for name in archive.namelist()
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
        and "/val2017/" in f"/{name}"
    )
    if not members:
        members = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
    if samples > len(members):
        raise ValueError(f"requested {samples} images, archive has {len(members)}")
    random.Random(seed).shuffle(members)
    return members[:samples]


def letterbox(image: Image.Image, width: int, height: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    contained = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (127, 127, 127))
    left = (width - contained.width) // 2
    top = (height - contained.height) // 2
    canvas.paste(contained, (left, top))
    return canvas


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "images.jsonl"
    with zipfile.ZipFile(args.archive) as archive:
        members = select_members(archive, args.samples, args.seed)
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for sample_index, member in enumerate(members):
                source_bytes = archive.read(member)
                source_sha256 = hashlib.sha256(source_bytes).hexdigest()
                with Image.open(io.BytesIO(source_bytes)) as source:
                    source_size = source.size
                    for bucket in BUCKETS:
                        bucket_dir = args.output / bucket.name
                        bucket_dir.mkdir(exist_ok=True)
                        destination = bucket_dir / f"{sample_index:04d}.jpg"
                        variant = letterbox(source, bucket.width, bucket.height)
                        variant.save(
                            destination,
                            format="JPEG",
                            quality=args.jpeg_quality,
                            optimize=True,
                        )
                        record = {
                            "sample_index": sample_index,
                            "source_member": member,
                            "source_sha256": source_sha256,
                            "source_width": source_size[0],
                            "source_height": source_size[1],
                            "path": str(destination.resolve()),
                            **asdict(bucket),
                            "expected_visual_tokens": bucket.expected_visual_tokens,
                        }
                        manifest.write(json.dumps(record, sort_keys=True) + "\n")
    metadata = {
        "archive": str(args.archive.resolve()),
        "samples": args.samples,
        "seed": args.seed,
        "jpeg_quality": args.jpeg_quality,
        "buckets": [
            {**asdict(bucket), "expected_visual_tokens": bucket.expected_visual_tokens}
            for bucket in BUCKETS
        ],
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.samples * len(BUCKETS)} variants to {args.output}")


if __name__ == "__main__":
    main()
