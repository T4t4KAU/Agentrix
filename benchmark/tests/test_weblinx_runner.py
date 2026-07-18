import base64
from pathlib import Path

import pytest

from weblinx_runner import (
    branch_image_case_index,
    image_data_url,
    load_manifest,
    shared_messages,
)


def test_branch_image_mapping_supports_same_and_control_modes() -> None:
    assert branch_image_case_index(3, 5, 8, "same") == 3
    assert branch_image_case_index(3, 5, 8, "different") == 0
    with pytest.raises(ValueError, match="unsupported image mode"):
        branch_image_case_index(0, 0, 8, "invalid")


def test_shared_messages_put_image_before_long_state() -> None:
    messages = shared_messages("long shared state", "data:image/png;base64,abc")

    content = messages[1]["content"]
    assert content[1]["type"] == "image_url"
    assert content[2] == {"type": "text", "text": "long shared state"}


def test_image_data_url_round_trips(tmp_path: Path) -> None:
    image = tmp_path / "screen.png"
    image.write_bytes(b"png bytes")

    url = image_data_url(image)

    assert base64.b64decode(url.partition(",")[2]) == b"png bytes"


def test_marked_image_variants_have_unique_payloads(tmp_path: Path) -> None:
    from PIL import Image

    image = tmp_path / "screen.png"
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(image)

    first = image_data_url(image, marker=1)
    second = image_data_url(image, marker=2)

    assert first != second


def test_load_manifest_rejects_incomplete_case(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"schema_version": 1, "branch_count": 2, "cases": '
        '[{"case_id": "case", "branches": [{}]}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid branch count"):
        load_manifest(path)
