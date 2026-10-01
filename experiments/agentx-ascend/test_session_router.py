"""Contract checks against the installed official Rust router, without a GPU."""

import asyncio
import json
import socket
import sys
import unittest
from pathlib import Path

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

ROOT = Path(__file__).resolve().parents[2]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class OfficialRouterTests(unittest.IsolatedAsyncioTestCase):
    policy = "consistent_hash"

    async def asyncSetUp(self):
        self.received = []
        self.release_stream = asyncio.Event()
        app = web.Application()

        async def health(request):
            return web.Response(text="ok")

        async def models(request):
            return web.json_response({"data": []})

        app.router.add_get("/health", health)
        app.router.add_get("/v1/models", models)
        app.router.add_get("/metrics", self.metrics)
        app.router.add_post("/v1/chat/completions", self.backend)
        app.router.add_post("/v1/completions", self.backend)
        self.worker = TestServer(app)
        await self.worker.start_server()
        self.addAsyncCleanup(self.worker.close)
        self.url = f"http://127.0.0.1:{free_port()}"
        metrics_port = free_port()
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(ROOT / "benchmark/scripts/serve_dp_router.py"),
            "--worker-urls",
            str(self.worker.make_url("/")).rstrip("/"),
            "--policy",
            self.policy,
            "--intra-node-data-parallel-size",
            "2",
            "--host",
            "127.0.0.1",
            "--port",
            self.url.rsplit(":", 1)[1],
            "--prometheus-port",
            str(metrics_port),
            "--worker-startup-check-interval",
            "1",
            "--worker-startup-timeout-secs",
            "10",
            "--disable-retries",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self.log_task = asyncio.create_task(self.process.stdout.read())
        self.addAsyncCleanup(self.stop_router)
        self.client = ClientSession()
        self.addAsyncCleanup(self.client.close)
        for _ in range(100):
            if self.process.returncode is not None:
                self.fail((await self.log_task).decode())
            try:
                async with self.client.get(self.url + "/health") as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            await asyncio.sleep(0.1)
        else:
            self.fail("Official router did not become ready")

    async def stop_router(self):
        self.release_stream.set()
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 5)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        await self.log_task

    async def metrics(self, request):
        return web.Response(
            text="\n".join(
                f'vllm:num_requests_running{{engine="{rank}"}} 0' for rank in range(2)
            )
        )

    async def backend(self, request):
        payload = await request.json()
        rank = request.headers.get("X-data-parallel-rank")
        self.received.append((dict(request.headers), payload, rank))
        if payload.get("stream"):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b'data: {"choices":[{"text":"hello"}]}\n\n')
            await self.release_stream.wait()
            await response.write(b"data: [DONE]\n\n")
            return response
        return web.json_response({"choices": [], "rank": rank})

    async def post(self, session_id, **extra):
        payload = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        payload.update(extra)
        async with self.client.post(
            self.url + "/v1/chat/completions",
            json=payload,
            headers={"X-Session-ID": session_id},
        ) as response:
            self.assertEqual(response.status, 200, await response.text())
            result = await response.json()
            self.assertIn(result["rank"], ("0", "1"))
            return result["rank"]

    async def test_growing_sessions_stay_on_their_dp_rank(self):
        routes = {}
        for session in range(20):
            routes[str(session)] = await self.post(str(session))
        self.assertEqual(set(routes.values()), {"0", "1"})
        for session, rank in routes.items():
            self.assertEqual(
                await self.post(
                    session,
                    messages=[
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "tool call"},
                        {"role": "user", "content": "tool result"},
                    ],
                ),
                rank,
            )

    async def test_hint_metadata_and_token_prompt_reach_backend(self):
        payload = {
            "model": "test",
            "prompt": [1, 2, 3],
            "max_tokens": 1,
            "vllm_xargs": {"example_metadata": 2},
        }
        async with self.client.post(
            self.url + "/v1/completions",
            json=payload,
            headers={"X-Session-ID": "tokens"},
        ) as response:
            self.assertEqual(response.status, 200, await response.text())
        _, forwarded, rank = self.received[-1]
        self.assertEqual(forwarded["prompt"], payload["prompt"])
        self.assertEqual(forwarded["vllm_xargs"], payload["vllm_xargs"])
        self.assertIn(rank, ("0", "1"))

    async def test_stream_arrives_before_backend_completion(self):
        async with self.client.post(
            self.url + "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers={"X-Session-ID": "stream"},
        ) as response:
            self.assertEqual(
                response.status,
                200,
                await response.text() if response.status != 200 else "",
            )
            line = await asyncio.wait_for(response.content.readline(), 2)
            self.assertEqual(
                json.loads(line.removeprefix(b"data: "))["choices"][0]["text"], "hello"
            )
            self.release_stream.set()
            self.assertIn(b"[DONE]", await response.read())


class OfficialCacheAwareTests(OfficialRouterTests):
    policy = "cache_aware"
    # Cache-aware routing has a different contract from session hashing.
    test_growing_sessions_stay_on_their_dp_rank = None


if __name__ == "__main__":
    unittest.main()
