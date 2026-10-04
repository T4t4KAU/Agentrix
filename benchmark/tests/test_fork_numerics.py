import asyncio
import json

import pytest
from benchmark_fork_numerics import capture_requests, compare_outputs, validate_logprobs


def output(ids, tops):
    tokens = [f"token_id:{token}" for token in ids]
    return {
        "token_ids": ids,
        "logprob_tokens": tokens,
        "token_logprobs": [top[token] for token, top in zip(tokens, tops)],
        "top_logprobs": tops,
    }


def test_probability_comparison_stops_when_conditioning_history_changes():
    a = output(
        [1, 2, 3],
        [
            {"token_id:1": -0.1, "token_id:2": -2},
            {"token_id:2": -0.68, "token_id:4": -0.69},
            {"token_id:3": -0.1},
        ],
    )
    b = output(
        [1, 4, 3],
        [
            {"token_id:1": -0.1, "token_id:2": -2},
            {"token_id:2": -0.69, "token_id:4": -0.68},
            {"token_id:3": -20},
        ],
    )
    result = compare_outputs(a, b)
    assert result["first_difference_index"] == 1
    assert result["same_history_steps"] == 2
    assert result["max_common_logprob_difference"] == pytest.approx(0.01)
    assert result["first_difference"]["baseline"]["top1_top2_gap"] == pytest.approx(
        0.01
    )


def test_nonfinite_or_misaligned_logprobs_fail_instead_of_looking_identical():
    with pytest.raises(RuntimeError, match="lengths"):
        validate_logprobs([1], [], [], [])
    with pytest.raises(RuntimeError, match="Nonfinite"):
        validate_logprobs([1], ["token_id:1"], [float("nan")], [{"token_id:1": -1}])
    with pytest.raises(RuntimeError, match="Missing chosen"):
        validate_logprobs([1], ["token_id:1"], [-1], [{"token_id:2": -1}])


def test_batched_stream_keeps_interleaved_choices_separate():
    def choice(index, token, final=False):
        return {
            "index": index,
            "token_ids": [token],
            "finish_reason": "length" if final else None,
            "logprobs": {
                "tokens": [f"token_id:{token}"],
                "token_logprobs": [-0.1],
                "top_logprobs": [{f"token_id:{token}": -0.1, "token_id:99": -3}],
            },
        }

    class Response:
        status = 200

        @property
        def content(self):
            async def events():
                for row in [
                    {"choices": [choice(1, 7), choice(0, 8)]},
                    {"choices": [choice(0, 9, True), choice(1, 10, True)]},
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 4, "completion_tokens": 4},
                    },
                ]:
                    yield ("data: " + json.dumps(row) + "\n").encode()

            return events()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Session:
        def post(self, url, *, json, headers):
            assert json["prompt"] == [[1, 2], [3, 4]]
            assert json["return_tokens_as_token_ids"] is True
            assert headers == {"X-Session-ID": "document-3"}
            return Response()

    rows = asyncio.run(
        capture_requests(
            Session(), "http://test", "model", [[1, 2], [3, 4]], 3, 2, batched=True
        )
    )
    assert [r.token_ids for r in rows] == [[8, 9], [7, 10]]
    assert rows[0].output_token_sha256 != rows[1].output_token_sha256
    assert all(r.prompt_tokens == 2 and r.completion_tokens == 2 for r in rows)
