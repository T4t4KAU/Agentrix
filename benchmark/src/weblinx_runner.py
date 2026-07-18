from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import mimetypes
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from api_runner import APIRequestResult, _time_per_output_token
from metrics import compare_trace
from models import BenchmarkTrace, BranchTrace
from reporting import write_results
from tokens import count_tokens, fit_text_to_tokens


SYSTEM_PROMPT = (
    "You are a web-navigation agent. Inspect the screenshot, conversation, "
    "browser history, and pruned DOM. Evaluate the candidate supplied in the "
    "final message and return exactly one WebLINX browser action."
)
SHARED_ACK = "The shared webpage state is loaded and ready for candidate evaluation."


@dataclass(frozen=True)
class WebLINXBranchResult:
    branch_id: int
    case_index: int
    case_id: str
    candidate_uid: str
    is_ground_truth: bool
    image_case_index: int
    image_variant_id: int | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    text: str


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if manifest.get("schema_version") != 1 or not isinstance(cases, list) or not cases:
        raise ValueError("invalid WebLINX manifest")
    branch_count = int(manifest.get("branch_count", 0))
    if branch_count <= 0:
        raise ValueError("manifest branch_count must be positive")
    for case in cases:
        if len(case.get("branches") or []) != branch_count:
            raise ValueError(f"case {case.get('case_id')} has an invalid branch count")
    return manifest


def image_data_url(path: Path, marker: int | None = None) -> str:
    payload_bytes = path.read_bytes()
    if marker is not None:
        from PIL import Image

        with Image.open(io.BytesIO(payload_bytes)) as image:
            marked = image.convert("RGB")
            marked.putpixel(
                (0, 0),
                (
                    marker & 0xFF,
                    (marker >> 8) & 0xFF,
                    (marker >> 16) & 0xFF,
                ),
            )
            output = io.BytesIO()
            marked.save(output, format="PNG")
            payload_bytes = output.getvalue()
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(payload_bytes).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def branch_image_case_index(
    case_index: int,
    branch_index: int,
    case_count: int,
    image_mode: str,
) -> int:
    if image_mode == "same":
        return case_index
    if image_mode == "different":
        return (case_index + branch_index) % case_count
    raise ValueError(f"unsupported image mode: {image_mode}")


def shared_messages(shared_text: str, data_url: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Observe this webpage screenshot before reading its state.",
                },
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": shared_text},
            ],
        },
        {"role": "assistant", "content": SHARED_ACK},
    ]


async def _request(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    output_tokens: int,
    rank: int,
    stream: bool,
) -> APIRequestResult:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": output_tokens,
        "temperature": 0,
        "extra_headers": {"X-data-parallel-rank": str(rank)},
        "extra_body": {"ignore_eos": True},
    }
    if stream:
        started = time.perf_counter()
        response_stream = await client.chat.completions.create(
            **kwargs,
            stream=True,
            stream_options={"include_usage": True},
        )
        text_parts: list[str] = []
        first_token_at: float | None = None
        usage = None
        async for chunk in response_stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content or ""
            if text:
                first_token_at = first_token_at or time.perf_counter()
                text_parts.append(text)
        completed_at = time.perf_counter()
        if usage is None:
            raise RuntimeError("streaming response omitted token usage")
        latency_ms = (completed_at - started) * 1000
        ttft_ms = (
            (first_token_at - started) * 1000 if first_token_at is not None else None
        )
        return APIRequestResult(
            text="".join(text_parts),
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
            tpot_ms=_time_per_output_token(
                latency_ms, ttft_ms, usage.completion_tokens
            ),
        )

    started = time.perf_counter()
    response = await client.chat.completions.create(**kwargs)
    latency_ms = (time.perf_counter() - started) * 1000
    if response.usage is None:
        raise RuntimeError("response omitted token usage")
    return APIRequestResult(
        text=response.choices[0].message.content or "",
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        latency_ms=latency_ms,
    )


async def run_benchmark(
    *,
    manifest_path: Path,
    model: str,
    base_url: str,
    output_dir: Path,
    dp_size: int,
    text_prefix_tokens: int,
    output_tokens: int,
    concurrency: int,
    image_mode: str,
    warm_prefix: bool,
    stream: bool,
    kv_bytes_per_token: int,
) -> tuple[BenchmarkTrace, dict[str, Any]]:
    if dp_size <= 0 or concurrency <= 0 or output_tokens <= 0:
        raise ValueError("DP size, concurrency, and output tokens must be positive")
    manifest = load_manifest(manifest_path)
    cases = manifest["cases"]
    if len(cases) != dp_size:
        raise ValueError(
            f"the fixed DP workload needs one case per rank: {len(cases)} != {dp_size}"
        )
    if image_mode == "different" and len(cases) < 2:
        raise ValueError("different-image mode requires at least two cases")

    root = manifest_path.parent
    image_paths = [root / case["screenshot"] for case in cases]
    image_urls = [image_data_url(path) for path in image_paths]
    control_image_urls = None
    if image_mode == "different":
        control_image_urls = [
            [
                image_data_url(
                    image_paths[
                        branch_image_case_index(
                            case_index,
                            branch_index,
                            len(cases),
                            image_mode,
                        )
                    ],
                    marker=case_index * manifest["branch_count"] + branch_index + 1,
                )
                for branch_index in range(manifest["branch_count"])
            ]
            for case_index in range(len(cases))
        ]
    fitted_texts = [
        fit_text_to_tokens(case["shared_text"], text_prefix_tokens, model)
        for case in cases
    ]
    estimated_prefix_tokens = [count_tokens(text, model) for text in fitted_texts]
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "vllm-local"),
        base_url=base_url,
        timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "900")),
    )

    warmup_results: list[APIRequestResult] = []
    warmup_started = time.perf_counter()
    if warm_prefix and image_mode == "same":
        warmup_results = await asyncio.gather(
            *(
                _request(
                    client,
                    model=model,
                    messages=shared_messages(fitted_texts[index], image_urls[index])
                    + [
                        {
                            "role": "user",
                            "content": "Confirm that this webpage state is ready.",
                        }
                    ],
                    output_tokens=1,
                    rank=index,
                    stream=False,
                )
                for index in range(len(cases))
            )
        )
    warmup_latency_ms = (time.perf_counter() - warmup_started) * 1000

    semaphore = asyncio.Semaphore(concurrency)

    async def run_branch(case_index: int, branch_index: int) -> WebLINXBranchResult:
        case = cases[case_index]
        branch = case["branches"][branch_index]
        image_index = branch_image_case_index(
            case_index, branch_index, len(cases), image_mode
        )
        image_variant_id = None
        data_url = image_urls[image_index]
        if image_mode == "different":
            assert control_image_urls is not None
            image_variant_id = case_index * manifest["branch_count"] + branch_index + 1
            data_url = control_image_urls[case_index][branch_index]
        messages = shared_messages(fitted_texts[case_index], data_url)
        messages.append({"role": "user", "content": branch["private_text"]})
        async with semaphore:
            result = await _request(
                client,
                model=model,
                messages=messages,
                output_tokens=output_tokens,
                rank=case_index % dp_size,
                stream=stream,
            )
        return WebLINXBranchResult(
            branch_id=case_index * manifest["branch_count"] + branch_index,
            case_index=case_index,
            case_id=case["case_id"],
            candidate_uid=branch["uid"],
            is_ground_truth=bool(branch["is_ground_truth"]),
            image_case_index=image_index,
            image_variant_id=image_variant_id,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
            ttft_ms=result.ttft_ms,
            tpot_ms=result.tpot_ms,
            text=result.text,
        )

    started = time.perf_counter()
    branch_results = await asyncio.gather(
        *(
            run_branch(case_index, branch_index)
            for case_index in range(len(cases))
            for branch_index in range(manifest["branch_count"])
        )
    )
    branch_latency_ms = (time.perf_counter() - started) * 1000
    await client.close()

    actual_prefix_tokens = [
        min(
            result.input_tokens
            for result in branch_results
            if result.case_index == case_index
        )
        for case_index in range(len(cases))
    ]
    mean_prefix_tokens = int(statistics.fmean(actual_prefix_tokens))
    trace = BenchmarkTrace(
        case_id=f"weblinx_8dp_{image_mode}_p{mean_prefix_tokens}",
        prefix_tokens=mean_prefix_tokens,
        branches=[
            BranchTrace(
                branch_id=result.branch_id,
                suffix_tokens=max(
                    0,
                    result.input_tokens - actual_prefix_tokens[result.case_index],
                ),
                decode_tokens=result.output_tokens,
                input_tokens=result.input_tokens,
                latency_ms=result.latency_ms,
                ttft_ms=result.ttft_ms,
                tpot_ms=result.tpot_ms,
                strategy=f"candidate:{result.candidate_uid}",
            )
            for result in branch_results
        ],
        output_tokens=output_tokens,
        metadata={
            "model": model,
            "manifest": str(manifest_path),
            "dp_size": dp_size,
            "image_mode": image_mode,
            "warm_prefix": warm_prefix,
            "text_prefix_tokens": text_prefix_tokens,
            "estimated_prefix_tokens": estimated_prefix_tokens,
            "actual_prefix_tokens": actual_prefix_tokens,
            "case_rank_map": {
                case["case_id"]: index for index, case in enumerate(cases)
            },
        },
    )
    raw = {
        "model": model,
        "manifest": str(manifest_path),
        "dp_size": dp_size,
        "image_mode": image_mode,
        "warm_prefix": warm_prefix,
        "common": {
            "latency_ms": warmup_latency_ms,
            "input_tokens": sum(item.input_tokens for item in warmup_results),
            "output_tokens": sum(item.output_tokens for item in warmup_results),
        },
        "warmup_latency_ms": warmup_latency_ms,
        "warmup_input_tokens": sum(item.input_tokens for item in warmup_results),
        "branch_phase_latency_ms": branch_latency_ms,
        "total_latency_ms": warmup_latency_ms + branch_latency_ms,
        "branch_total_output_tokens": sum(
            item.output_tokens for item in branch_results
        ),
        "actual_prefix_tokens": actual_prefix_tokens,
        "branches": [asdict(result) for result in branch_results],
    }
    result = compare_trace(trace, kv_bytes_per_token=kv_bytes_per_token)
    write_results(output_dir, [trace], [result], [raw])
    return trace, raw


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed WebLINX 8-DP workload")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dp-size", type=int, default=8)
    parser.add_argument("--text-prefix-tokens", type=int, default=28000)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--image-mode", choices=["same", "different"], default="same")
    parser.add_argument(
        "--warm-prefix", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kv-bytes-per-token", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    asyncio.run(
        run_benchmark(
            manifest_path=args.manifest,
            model=args.model,
            base_url=args.base_url,
            output_dir=args.output_dir,
            dp_size=args.dp_size,
            text_prefix_tokens=args.text_prefix_tokens,
            output_tokens=args.output_tokens,
            concurrency=args.concurrency,
            image_mode=args.image_mode,
            warm_prefix=args.warm_prefix,
            stream=args.stream,
            kv_bytes_per_token=args.kv_bytes_per_token,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
