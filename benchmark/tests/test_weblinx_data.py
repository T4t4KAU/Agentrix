import json
from pathlib import Path

from PIL import Image

from weblinx_data import (
    action_uid,
    normalize_screenshot,
    parse_candidates,
    screenshot_name_from_replay,
    select_candidate_rows,
)


def _row(demo: str, turn: int, target_index: int = 0) -> dict[str, object]:
    candidates = "\n".join(
        f"(uid = uid-{index}) [[tag]] button [[text]] candidate {index}"
        for index in range(8)
    )
    return {
        "demo": demo,
        "turn": turn,
        "action": f'click(uid="uid-{target_index}")',
        "action_history": "load(url=example.com)",
        "utterances": "Open the example page.",
        "candidates": candidates,
        "clean_html": "(html(body(button Example)))",
        "viewport": "720h x 1280w",
    }


def test_parse_candidates_preserves_rank_and_deduplicates_uid() -> None:
    candidates = parse_candidates(
        "\n".join(
            (
                "(uid = first) [[tag]] button",
                "not a candidate",
                "(uid = first) duplicate",
                "(uid = second) [[tag]] input",
            )
        )
    )

    assert candidates == [
        {"uid": "first", "rank": 1, "document": "[[tag]] button"},
        {"uid": "second", "rank": 2, "document": "[[tag]] input"},
    ]


def test_select_candidate_rows_uses_distinct_demonstrations() -> None:
    selected = select_candidate_rows(
        [_row("demo-a", 1), _row("demo-a", 2), _row("demo-b", 3)],
        case_count=2,
        branch_count=8,
        seed=7,
    )

    assert {row["demo"] for row in selected} == {"demo-a", "demo-b"}
    assert all(action_uid(str(row["action"])) == "uid-0" for row in selected)


def test_screenshot_name_requires_good_screenshot() -> None:
    replay = {
        "data": [
            {"state": {"screenshot": "bad.png", "screenshot_status": "broken"}},
            {"state": {"screenshot": "good.png", "screenshot_status": "good"}},
        ]
    }

    assert screenshot_name_from_replay(replay, 1) == "good.png"


def test_normalize_screenshot_has_fixed_dimensions_and_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    destination = tmp_path / "assets" / "normalized.png"
    Image.new("RGB", (200, 100), color=(10, 20, 30)).save(source)

    digest = normalize_screenshot(source, destination)

    with Image.open(destination) as image:
        assert image.size == (1280, 720)
    assert len(digest) == 64
    assert json.loads(json.dumps({"sha256": digest}))["sha256"] == digest
