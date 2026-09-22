import asyncio
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from session_router import create_app


class RouterTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.received = []
        self.release_stream = asyncio.Event()

        async def backend(request):
            self.received.append(
                (dict(request.headers), await request.read(), request.path_qs)
            )
            if request.query.get("stream"):
                response = web.StreamResponse(
                    headers={"Content-Type": "text/event-stream"}
                )
                await response.prepare(request)
                await response.write(b'data: {"text":"hello"}\n\n')
                await self.release_stream.wait()
                await response.write(b"data: [DONE]\n\n")
                return response
            return web.json_response(
                {"rank": request.headers.get("X-data-parallel-rank")}
            )

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", backend)
        self.backend = TestServer(app)
        await self.backend.start_server()
        self.clients = []

    async def asyncTearDown(self):
        self.release_stream.set()
        for client in self.clients:
            await client.close()
        await self.backend.close()

    async def proxy(self, policy="sticky"):
        client = TestClient(
            TestServer(create_app(str(self.backend.make_url("/")), policy=policy))
        )
        await client.start_server()
        self.clients.append(client)
        return client

    async def test_session_affinity_and_body_preservation(self):
        client = await self.proxy()
        for session, rank in [("parent", "0"), ("child", "1"), ("parent", "0")]:
            response = await client.post(
                "/v1/chat/completions",
                data=b'{"messages":[]}',
                headers={"X-Session-ID": session},
            )
            self.assertEqual(await response.json(), {"rank": rank})
            self.assertEqual(self.received[-1][1], b'{"messages":[]}')
        stats = await (await client.get("/routing-stats")).json()
        self.assertEqual(stats["requests"], {"0": 2, "1": 1})

    async def test_native_clears_caller_rank(self):
        client = await self.proxy("native")
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "X-Session-ID": "s",
                "x-data-parallel-rank": "999",
            },
        )
        self.assertEqual(await response.json(), {"rank": None})

    async def test_missing_session_uses_native(self):
        client = await self.proxy()
        response = await client.post("/v1/chat/completions")
        self.assertEqual(await response.json(), {"rank": None})

    async def test_first_turn_control_only_pins_new_sessions(self):
        client = await self.proxy("first-turn-only")
        for session, rank in [
            ("parent", "0"),
            ("child", "1"),
            ("parent", None),
            ("child", None),
            ("new", "0"),
        ]:
            response = await client.post(
                "/v1/chat/completions", headers={"X-Session-ID": session}
            )
            self.assertEqual(await response.json(), {"rank": rank})
        stats = await (await client.get("/routing-stats")).json()
        self.assertEqual(stats["requests"], {"0": 2, "1": 1, "native": 2})

    async def test_correlation_fallback_and_session_precedence(self):
        client = await self.proxy()
        for headers, rank in [
            ({"X-Correlation-ID": "c"}, "0"),
            ({"X-Session-ID": "s", "X-Correlation-ID": "c"}, "1"),
            ({"X-Session-ID": "c"}, "0"),
        ]:
            response = await client.post("/v1/chat/completions", headers=headers)
            self.assertEqual(await response.json(), {"rank": rank})

    async def test_streaming_reaches_client_before_completion(self):
        client = await self.proxy()
        response = await client.post(
            "/v1/chat/completions?stream=1", headers={"X-Session-ID": "s"}
        )
        first = await asyncio.wait_for(response.content.readline(), timeout=2)
        self.assertEqual(first, b'data: {"text":"hello"}\n')
        self.release_stream.set()
        self.assertIn(b"[DONE]", await response.read())
        self.assertEqual(self.received[-1][2], "/v1/chat/completions?stream=1")

    async def test_connection_scoped_headers_are_removed(self):
        client = await self.proxy()
        await client.post(
            "/v1/chat/completions", headers={"Connection": "X-Hop", "X-Hop": "secret"}
        )
        self.assertNotIn("X-Hop", self.received[-1][0])


if __name__ == "__main__":
    unittest.main()
