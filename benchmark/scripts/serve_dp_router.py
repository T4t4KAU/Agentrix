"""Launch the pinned official vLLM router without a local routing policy."""

import importlib.metadata
import json
import sys


def main():
    expected = "0.1.15"
    try:
        version = importlib.metadata.version("vllm-router")
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit("Install benchmark/requirements-router.txt in the router venv")
    if version != expected:
        raise SystemExit(f"Expected vllm-router=={expected}, found {version}")

    from vllm_router.launch_router import Router, launch_router, parse_router_args

    if Router is None:
        raise SystemExit("The official Rust router extension is required")
    if sys.argv[1:] == ["--check"]:
        print(json.dumps({"implementation": "vllm-project/router", "version": version}))
        return
    args = parse_router_args(sys.argv[1:])
    if args.mini_lb:
        raise SystemExit("Use the official Rust router for these experiments")
    launch_router(args)


if __name__ == "__main__":
    main()
