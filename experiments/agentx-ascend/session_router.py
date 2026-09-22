"""Single-process AgentX proxy with a first-turn control for sticky DP routing."""

import argparse
from collections import Counter

from aiohttp import ClientSession, ClientTimeout, web


def forwarded_headers(headers):
    """Remove hop-by-hop fields, including fields named by Connection."""
    excluded = {
        "host",
        "content-length",
        "connection",
        "transfer-encoding",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "upgrade",
    }
    excluded.update(
        value.strip().lower() for value in headers.get("Connection", "").split(",")
    )
    return {key: value for key, value in headers.items() if key.lower() not in excluded}


class SessionRouter:
    def __init__(self, upstream: str, replicas: int, policy: str):
        if replicas < 1 or policy not in {"native", "first-turn-only", "sticky"}:
            raise ValueError(
                "Expected positive replicas and native/first-turn-only/sticky policy"
            )
        self.upstream = upstream.rstrip("/")
        self.replicas = replicas
        self.policy = policy
        # Retain mappings for the full benchmark, including inter-turn idle gaps.
        # Restart the router between runs. This is not a multi-worker gateway.
        self.ranks: dict[str, int] = {}
        self.counts: Counter = Counter()

    async def client_context(self, app):
        async with ClientSession(
            timeout=ClientTimeout(total=1800), auto_decompress=False
        ) as client:
            self.client = client
            yield

    async def proxy(self, request):
        if request.path == "/routing-stats":
            return web.json_response(
                {
                    "policy": self.policy,
                    "replicas": self.replicas,
                    "sessions": dict(Counter(self.ranks.values())),
                    "requests": dict(self.counts),
                }
            )
        headers = {
            key: value
            for key, value in forwarded_headers(request.headers).items()
            if key.lower() != "x-data-parallel-rank"
        }
        session = request.headers.get("X-Session-ID") or request.headers.get(
            "X-Correlation-ID"
        )
        if request.method == "POST" and request.path.startswith("/v1/"):
            rank = None
            if self.policy != "native" and session:
                first_turn = session not in self.ranks
                if first_turn:
                    self.ranks[session] = len(self.ranks) % self.replicas
                if first_turn or self.policy == "sticky":
                    rank = self.ranks[session]
            if rank is not None:
                headers["X-data-parallel-rank"] = str(rank)
                self.counts[str(rank)] += 1
            else:
                self.counts["native"] += 1
        async with self.client.request(
            request.method,
            self.upstream + request.path_qs,
            headers=headers,
            data=await request.read(),
            allow_redirects=False,
        ) as upstream:
            response = web.StreamResponse(
                status=upstream.status, headers=forwarded_headers(upstream.headers)
            )
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            return response


def create_app(upstream="http://127.0.0.1:8001", replicas=2, policy="sticky"):
    router = SessionRouter(upstream, replicas, policy)
    app = web.Application(client_max_size=64 * 1024**2)
    app.cleanup_ctx.append(router.client_context)
    app.router.add_route("*", "/{path:.*}", router.proxy)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:8001")
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument(
        "--policy", choices=("native", "first-turn-only", "sticky"), default="sticky"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    web.run_app(
        create_app(args.upstream, args.replicas, args.policy),
        host=args.host,
        port=args.port,
        handler_cancellation=True,
    )
