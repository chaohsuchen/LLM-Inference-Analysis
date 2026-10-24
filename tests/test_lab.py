import asyncio
import tempfile
import unittest
from pathlib import Path

from aiohttp import web
from lab.bench import percentile, request, summarize, benchmark
from lab.mock_server import completion
from lab.run import MockTokenizer, CONFIGS, command
from lab.workloads import make_workload


class MetricsTests(unittest.TestCase):
    def test_quantiles_and_failed_goodput(self):
        self.assertEqual(percentile([1, 2, 3, 4], .5), 2.5)
        self.assertIsNone(percentile([], .99))
        rows = [{"ok": True, "class": "short", "input_tokens": 4, "ttft_s": 1,
                 "e2e_s": 2, "tpot_s": .1, "output_tokens": 10},
                {"ok": False, "input_tokens": 4}]
        s = summarize(rows, 4, 2)
        self.assertEqual(s["output_tokens_per_s"], 2.5)
        self.assertEqual(s["ttft_goodput_req_s"], .25)
        self.assertEqual(s["failed"], 1)
        rows[0]["output_tokens"] = None
        self.assertIsNone(summarize(rows, 4, 2)["output_tokens_per_s"])

    def test_workload_replay_and_early_unique_ids(self):
        tok = MockTokenizer()
        a = make_workload(tok, "mixed", 20)
        self.assertEqual(a, make_workload(tok, "mixed", 20))
        self.assertEqual(sum(r["class"] == "long" for r in a), 5)
        self.assertEqual(len({r["prompt"] for r in a}), 20)
        self.assertNotEqual(a, make_workload(tok, "mixed", 20, namespace="other"))
        warmup = make_workload(tok, "mixed", 2, id_offset=20)
        self.assertTrue({r["id"] for r in a}.isdisjoint(r["id"] for r in warmup))


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app = web.Application()
        self.app.router.add_post("/v1/completions", completion)
        async def truncated(request):
            return web.Response(text='data: {"choices":[{"text":"hello"}]}\n\n',
                                content_type="text/event-stream")
        async def coalesced(request):
            return web.Response(text='data: {"choices":[{"text":"many tokens in one chunk","finish_reason":"length"}]}\n\n'
                                     'data: {"choices":[],"usage":{"completion_tokens":6,"prompt_tokens":1}}\n\n'
                                     'data: [DONE]\n\n', content_type="text/event-stream")
        self.app.router.add_post("/truncated/v1/completions", truncated)
        self.app.router.add_post("/coalesced/v1/completions", coalesced)
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        self.url = f"http://127.0.0.1:{self.site._server.sockets[0].getsockname()[1]}"

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def test_streaming_usage_and_summary_files(self):
        rows = make_workload(MockTokenizer(), "mixed", 4, mini=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cell"
            s = await benchmark(self.url, "mock", rows, 2, 8, "mock", path, {})
            self.assertEqual(s["successful"], 4)
            self.assertGreater(s["all_ttft_s_p50"], 0)
            self.assertEqual(len((path / "requests.jsonl").read_text().splitlines()), 4)
            self.assertTrue((path / "summary.json").exists())

    async def test_http_failure_is_recorded(self):
        import aiohttp
        async with aiohttp.ClientSession() as session:
            r = await request(session, self.url + "/wrong", "mock",
                              {"id": 0, "class": "short", "prompt": "test", "input_tokens": 1},
                              8, "mock", 2)
        self.assertFalse(r["ok"])
        self.assertIn("404", r["error"])

    async def test_truncation_and_chunk_token_distinction(self):
        import aiohttp
        row = {"id": 0, "class": "short", "prompt": "test", "input_tokens": 1}
        async with aiohttp.ClientSession() as session:
            bad = await request(session, self.url + "/truncated", "mock", row, 8, "mock", 2)
            good = await request(session, self.url + "/coalesced", "mock", row, 8, "mock", 2)
        self.assertFalse(bad["ok"])
        self.assertTrue(good["ok"])
        self.assertEqual(good["content_chunks"], 1)
        self.assertEqual(good["output_tokens"], 6)


if __name__ == "__main__":
    unittest.main()
