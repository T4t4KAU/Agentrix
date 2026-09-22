import asyncio
from collections import Counter
from aiohttp import ClientSession, ClientTimeout, web

async def session_context(app):
    async with ClientSession(timeout=ClientTimeout(total=1800)) as session:
        app['client'] = session
        yield

async def proxy(request):
    if request.path == '/routing-stats':
        return web.json_response({'sessions': dict(Counter(request.app['ranks'].values())), 'requests': dict(request.app['counts'])})
    headers = {k: v for k, v in request.headers.items() if k.lower() not in {'host', 'content-length', 'connection', 'transfer-encoding'}}
    key = request.headers.get('X-Session-ID') or request.headers.get('X-Correlation-ID')
    if key and request.method == 'POST':
        ranks = request.app['ranks']
        if key not in ranks:
            ranks[key] = len(ranks) % 2
        rank = ranks[key]
        headers['X-data-parallel-rank'] = str(rank)
        request.app['counts'][str(rank)] += 1
    body = await request.read()
    async with request.app['client'].request(request.method, 'http://127.0.0.1:8001' + request.path_qs, headers=headers, data=body) as upstream:
        response = web.StreamResponse(status=upstream.status, headers={k: v for k, v in upstream.headers.items() if k.lower() not in {'content-length', 'transfer-encoding', 'connection', 'content-encoding'}})
        await response.prepare(request)
        async for chunk in upstream.content.iter_any():
            await response.write(chunk)
        await response.write_eof()
        return response

app = web.Application(client_max_size=64 * 1024**2)
app['ranks'] = {}
app['counts'] = Counter()
app.cleanup_ctx.append(session_context)
app.router.add_route('*', '/{path:.*}', proxy)
web.run_app(app, host='127.0.0.1', port=8000, print=None)
