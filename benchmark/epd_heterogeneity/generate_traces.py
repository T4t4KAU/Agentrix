from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--sessions", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sparse-max-tokens", type=int, default=256)
    return parser.parse_args()


def load_images(path: Path, bucket: str) -> list[dict[str, object]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [record for record in records if record["name"] == bucket]


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def sparse_trace(
    images: list[dict[str, object]],
    sessions: int,
    steps: int,
    probability: float,
    rng: random.Random,
    max_tokens: int,
) -> list[dict[str, object]]:
    records = []
    image_index = 0
    for workflow_id in range(sessions):
        offset_ms = rng.randint(0, 500)
        for step_id in range(steps):
            has_image = rng.random() < probability
            image = images[image_index % len(images)] if has_image else None
            image_index += int(has_image)
            tool_gap_ms = 1000 + rng.randint(0, 500)
            records.append(
                {
                    "workflow_id": workflow_id,
                    "step_id": step_id,
                    "arrival_offset_ms": offset_ms,
                    "step_type": "visual" if has_image else "text",
                    "image_path": image["path"] if image else None,
                    "expected_visual_tokens": (
                        image["expected_visual_tokens"] if image else 0
                    ),
                    "text_input_tokens": 512,
                    "max_tokens": max_tokens,
                    "tool_gap_ms": tool_gap_ms,
                }
            )
            offset_ms += tool_gap_ms
    return sorted(records, key=lambda item: int(item["arrival_offset_ms"]))


def burst_trace(
    images: list[dict[str, object]], burst_size: int
) -> list[dict[str, object]]:
    records = []
    for workflow_id in range(burst_size):
        image = images[workflow_id % len(images)]
        records.append(
            {
                "workflow_id": workflow_id,
                "step_id": 0,
                "arrival_offset_ms": 2000,
                "step_type": "visual_burst",
                "image_path": image["path"],
                "expected_visual_tokens": image["expected_visual_tokens"],
                "text_input_tokens": 128,
                "max_tokens": 128,
            }
        )
    return records


def encoder_profile_trace(
    images: list[dict[str, object]], concurrency: int, image_offset: int
) -> list[dict[str, object]]:
    records = []
    for workflow_id in range(concurrency):
        image = images[(image_offset + workflow_id) % len(images)]
        records.append(
            {
                "workflow_id": workflow_id,
                "step_id": 0,
                "arrival_offset_ms": 1000,
                "step_type": "encoder_profile",
                "image_path": image["path"],
                "expected_visual_tokens": image["expected_visual_tokens"],
                "text_input_tokens": 32,
                "max_tokens": 1,
            }
        )
    return records


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    medium_images = load_images(args.image_manifest, "medium")
    large_images = load_images(args.image_manifest, "1080p_class")
    rng = random.Random(args.seed)
    for probability in (0.05, 0.10, 0.25, 0.50, 1.00):
        records = sparse_trace(
            medium_images,
            args.sessions,
            args.steps,
            probability,
            rng,
            args.sparse_max_tokens,
        )
        write_jsonl(args.output / f"sparse_p{probability:.2f}.jsonl", records)
    for burst_size in (4, 8, 16, 32):
        write_jsonl(
            args.output / f"burst_{burst_size}.jsonl",
            burst_trace(large_images, burst_size),
        )
    for bucket in ("low", "medium", "1080p_class", "4k_class"):
        bucket_images = load_images(args.image_manifest, bucket)
        for matrix_index, concurrency in enumerate((1, 4, 16)):
            write_jsonl(
                args.output / f"a1_encoder_{bucket}_c{concurrency}.jsonl",
                encoder_profile_trace(
                    bucket_images,
                    concurrency,
                    image_offset=matrix_index * 16,
                ),
            )
    print(f"wrote synthetic traces to {args.output}")


if __name__ == "__main__":
    main()
