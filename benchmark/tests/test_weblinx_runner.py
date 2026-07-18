import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from weblinx_runner import (
    _request,
    branch_image_case_index,
    expanded_branch_specs,
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
    assert content[2]["type"] == "text"
    assert content[2]["text"].startswith("long shared state")
    assert len(messages) == 2


def test_expanded_branch_specs_match_pressure32k_shape() -> None:
    specs = expanded_branch_specs(
        case_count=8,
        candidates_per_case=8,
        rollouts_per_candidate=4,
        branch_order="shuffle",
        seed=2026,
    )

    assert len(specs) == 256
    assert len(set(specs)) == 256
    assert sum(case == 3 for case, _, _ in specs) == 32
    assert sum(case == 3 and candidate == 5 for case, candidate, _ in specs) == 4
    assert specs != sorted(specs)


def test_expanded_branch_specs_reject_invalid_order() -> None:
    with pytest.raises(ValueError, match="unsupported branch order"):
        expanded_branch_specs(
            case_count=8,
            candidates_per_case=8,
            rollouts_per_candidate=4,
            branch_order="round_robin",
            seed=2026,
        )


def test_request_leaves_internal_dp_routing_to_server() -> None:
    captured: dict[str, object] = {}

    class Completions:
        async def create(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
                choices=[SimpleNamespace(message=SimpleNamespace(content="action()"))],
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()),
    )
    result = asyncio.run(
        _request(
            client,
            model="model",
            messages=[{"role": "user", "content": "prompt"}],
            output_tokens=4,
            stream=False,
            priority=10,
        )
    )

    assert "extra_headers" not in captured
    assert captured["extra_body"] == {"ignore_eos": True, "priority": 10}
    assert result.output_tokens == 4


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
