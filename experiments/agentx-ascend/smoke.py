"""Check both DP ranks can advance and reuse a multi-block Qwen3.5 prompt."""

import argparse
import json
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default="Qwen3.5-9B")
    args = parser.parse_args()
    payload = json.dumps(
        {
            "model": args.model,
            "messages": [
                {"role": "system", "content": "以下为背景材料。" * 2048},
                {"role": "user", "content": "1+1等于几？只输出数字。"},
            ],
            "temperature": 0,
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    for rank in range(2):
        for repeat in range(2):
            request = urllib.request.Request(
                args.backend.rstrip("/") + "/v1/chat/completions",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-data-parallel-rank": str(rank),
                },
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.load(response)
            choice = result["choices"][0]
            assert choice["message"]["content"].strip() == "2", result
            assert choice["finish_reason"] == "stop", result
            assert result["usage"]["prompt_tokens"] > 4096, result
            cached = (result["usage"].get("prompt_tokens_details") or {}).get(
                "cached_tokens", 0
            )
            if repeat:
                assert cached > 0, result
            print(
                json.dumps({"rank": rank, "repeat": repeat, "usage": result["usage"]}),
                flush=True,
            )


if __name__ == "__main__":
    main()
