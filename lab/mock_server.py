"""Deterministic protocol fixture. Its results are never hardware measurements."""
import argparse
import asyncio
import json
from aiohttp import web


async def completion(request):
    body = await request.json()
    n = body.get("max_tokens", 8)
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)
    for _ in range(n):
        await asyncio.sleep(0.002)
        await response.write(("data: " + json.dumps({"choices": [{"text": " x", "finish_reason": None}]}) + "\n\n").encode())
    event = {"choices": [{"text": "", "finish_reason": "length"}],
             "usage": {"completion_tokens": n, "prompt_tokens": len(body["prompt"])}}
    await response.write(("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode())
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app = web.Application()
    app.router.add_post("/v1/completions", completion)
    async def models(request):
        return web.json_response({"data": [{"id": "mock"}]})
    app.router.add_get("/v1/models", models)
    web.run_app(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
