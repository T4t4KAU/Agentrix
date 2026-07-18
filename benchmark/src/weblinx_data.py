from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET = "McGill-NLP/WebLINX"
DEFAULT_RAW_DATASET = "McGill-NLP/WebLINX-full"
SUPPORTED_INTENTS = {"change", "click", "submit", "text_input"}
_CANDIDATE_RE = re.compile(r"^\(uid\s*=\s*([^)]+)\)\s*(.*)$")
_ACTION_UID_RE = re.compile(r'uid="([^"]+)"')


def parse_candidates(value: str, limit: int | None = None) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in value.splitlines():
        match = _CANDIDATE_RE.match(line.strip())
        if match is None:
            continue
        uid, document = match.groups()
        uid = uid.strip()
        if not uid or uid in seen:
            continue
        seen.add(uid)
        candidates.append(
            {
                "uid": uid,
                "rank": len(candidates) + 1,
                "document": document.strip(),
            }
        )
        if limit is not None and len(candidates) >= limit:
            break
    return candidates


def action_intent(action: str) -> str:
    return action.partition("(")[0].strip()


def action_uid(action: str) -> str | None:
    match = _ACTION_UID_RE.search(action)
    return match.group(1) if match else None


def build_shared_text(row: dict[str, Any]) -> str:
    return "\n\n".join(
        (
            f"Viewport: {row.get('viewport') or 'unknown'}",
            "Conversation context:\n" + str(row.get("utterances") or "none"),
            "Previous browser actions:\n" + str(row.get("action_history") or "none"),
            "Pruned webpage DOM:\n" + str(row.get("clean_html") or "none"),
        )
    )


def select_candidate_rows(
    rows: Iterable[dict[str, Any]],
    *,
    case_count: int,
    branch_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if case_count <= 0 or branch_count <= 0:
        raise ValueError("case_count and branch_count must be positive")

    eligible = _eligible_candidate_rows(rows, branch_count=branch_count, seed=seed)
    selected = eligible[:case_count]
    if len(selected) != case_count:
        raise ValueError(
            f"found only {len(selected)} eligible distinct demonstrations; "
            f"need {case_count}"
        )
    return selected


def _eligible_candidate_rows(
    rows: Iterable[dict[str, Any]], *, branch_count: int, seed: int
) -> list[dict[str, Any]]:

    eligible: list[dict[str, Any]] = []
    for row in rows:
        action = str(row.get("action") or "")
        if action_intent(action) not in SUPPORTED_INTENTS:
            continue
        candidates = parse_candidates(str(row.get("candidates") or ""))
        target_uid = action_uid(action)
        if len(candidates) < branch_count or target_uid is None:
            continue
        if target_uid not in {
            candidate["uid"] for candidate in candidates[:branch_count]
        }:
            continue
        shared_text = build_shared_text(row)
        if "password" in shared_text.casefold():
            continue
        eligible.append(dict(row))

    random.Random(seed).shuffle(eligible)
    distinct: list[dict[str, Any]] = []
    demos: set[str] = set()
    for row in eligible:
        demo = str(row["demo"])
        if demo in demos:
            continue
        demos.add(demo)
        distinct.append(row)
    return distinct


def screenshot_name_from_replay(replay: dict[str, Any], turn_index: int) -> str:
    turns = replay.get("data")
    if not isinstance(turns, list) or not 0 <= turn_index < len(turns):
        raise ValueError(f"turn index {turn_index} is missing from replay")
    state = turns[turn_index].get("state") or {}
    if state.get("screenshot_status") != "good":
        raise ValueError(f"turn {turn_index} does not have a good screenshot")
    screenshot = state.get("screenshot")
    if not isinstance(screenshot, str) or not screenshot:
        raise ValueError(f"turn {turn_index} does not reference a screenshot")
    return screenshot


def normalize_screenshot(source: Path, destination: Path) -> str:
    from PIL import Image, ImageOps

    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        normalized = ImageOps.pad(
            image.convert("RGB"),
            (1280, 720),
            method=Image.Resampling.LANCZOS,
            color=(255, 255, 255),
        )
        normalized.save(destination, format="PNG", optimize=True)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def build_manifest(
    *,
    output_dir: Path,
    split: str,
    case_count: int,
    branch_count: int,
    seed: int,
    dataset: str = DEFAULT_DATASET,
    raw_dataset: str = DEFAULT_RAW_DATASET,
    revision: str | None = None,
) -> Path:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    if case_count <= 0 or branch_count <= 0:
        raise ValueError("case_count and branch_count must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_dataset(dataset, "chat", split=split, revision=revision)
    selected = _eligible_candidate_rows(rows, branch_count=branch_count, seed=seed)

    cases: list[dict[str, Any]] = []
    for row in selected:
        demo = str(row["demo"])
        turn_index = int(row["turn"])
        replay_path = Path(
            hf_hub_download(
                raw_dataset,
                f"demonstrations/{demo}/replay.json",
                repo_type="dataset",
                revision=revision,
            )
        )
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        try:
            screenshot_name = screenshot_name_from_replay(replay, turn_index)
        except ValueError:
            continue
        screenshot_path = Path(
            hf_hub_download(
                raw_dataset,
                f"demonstrations/{demo}/screenshots/{screenshot_name}",
                repo_type="dataset",
                revision=revision,
            )
        )
        case_index = len(cases)
        relative_image = Path("assets") / f"case_{case_index:02d}.png"
        image_sha256 = normalize_screenshot(
            screenshot_path, output_dir / relative_image
        )
        candidates = parse_candidates(str(row["candidates"]), limit=branch_count)
        target_uid = action_uid(str(row["action"]))
        branches = [
            {
                **candidate,
                "is_ground_truth": candidate["uid"] == target_uid,
                "private_text": (
                    f"Candidate rank: {candidate['rank']}\n"
                    f"Candidate UID: {candidate['uid']}\n"
                    f"Candidate element: {candidate['document']}\n\n"
                    "Evaluate this candidate as the next browser action. Return only "
                    "one action in WebLINX function-call format."
                ),
            }
            for candidate in candidates
        ]
        cases.append(
            {
                "case_index": case_index,
                "case_id": f"{demo}-{turn_index}",
                "demo_name": demo,
                "turn_index": turn_index,
                "intent": action_intent(str(row["action"])),
                "target_action": row["action"],
                "viewport": row.get("viewport"),
                "screenshot": relative_image.as_posix(),
                "image_sha256": image_sha256,
                "shared_text": build_shared_text(row),
                "branches": branches,
            }
        )
        if len(cases) == case_count:
            break

    if len(cases) != case_count:
        raise ValueError(
            f"found only {len(cases)} eligible cases with good screenshots; "
            f"need {case_count}"
        )

    manifest = {
        "schema_version": 1,
        "source_dataset": dataset,
        "raw_dataset": raw_dataset,
        "revision": revision,
        "split": split,
        "seed": seed,
        "case_count": case_count,
        "branch_count": branch_count,
        "image_size": [1280, 720],
        "cases": cases,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a small WebLINX image subset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--case-count", type=int, default=8)
    parser.add_argument("--branch-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--raw-dataset", default=DEFAULT_RAW_DATASET)
    parser.add_argument("--revision")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_manifest(
        output_dir=args.output_dir,
        split=args.split,
        case_count=args.case_count,
        branch_count=args.branch_count,
        seed=args.seed,
        dataset=args.dataset,
        raw_dataset=args.raw_dataset,
        revision=args.revision,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
