import asyncio
import json
import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "allow,details,expected",
    [(False, None, None), (True, None, 0), (True, {"cached_tokens": 8}, 8)],
)
def test_legacy_usage_requires_explicit_opt_in(allow, details, expected):
    module = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/benchmark_prefix_aware_dp.py")
    )
    usage = {"prompt_tokens": 12, "completion_tokens": 1}
    if details is not None:
        usage["prompt_tokens_details"] = details

    class Response:
        status = 200

        @property
        def content(self):
            async def stream():
                yield (
                    "data: "
                    + json.dumps({"choices": [{"token_ids": [7]}], "usage": usage})
                    + "\n"
                ).encode()

            return stream()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class Session:
        def post(self, *args, **kwargs):
            return Response()

    request = module["run_request"](
        Session(),
        "http://test",
        "model",
        [1] * 12,
        0,
        1,
        allow_missing_prompt_details=allow,
    )
    if expected is None:
        with pytest.raises(RuntimeError, match="prompt-token details"):
            asyncio.run(request)
    else:
        assert asyncio.run(request).cached_tokens == expected
