"""Offline arithmetic and export checks; no simulated inference service."""
import json
import tempfile
import unittest
from pathlib import Path

from lab.bench import percentile, summarize
from lab.report import export_data


class MetricsTests(unittest.TestCase):
    def test_quantiles_and_failed_goodput(self):
        self.assertEqual(percentile([1, 2, 3, 4], .5), 2.5)
        self.assertIsNone(percentile([], .99))
        rows = [{"ok": True, "class": "short", "input_tokens": 4, "ttft_s": 1,
                 "e2e_s": 2, "tpot_s": .1, "output_tokens": 10},
                {"ok": False, "input_tokens": 4}]
        summary = summarize(rows, 4, 2)
        self.assertEqual(summary["output_tokens_per_s"], 2.5)
        self.assertEqual(summary["ttft_goodput_req_s"], .25)
        self.assertEqual(summary["failed"], 1)
        rows[0]["output_tokens"] = None
        self.assertIsNone(summarize(rows, 4, 2)["output_tokens_per_s"])

    def test_export_preserves_timing_and_excludes_prompt_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.json"
            summary.write_text(json.dumps({"measurement_source": "live_vllm_http"}))
            (root / "requests.jsonl").write_text(json.dumps({
                "id": 1, "dispatch_s": 0.1, "first_content_s": 0.3,
                "last_content_s": 0.7, "completion_s": 0.8,
                "prompt": "private input", "sample": "private output"}) + "\n")
            data = export_data([summary], root)
            request = data["cells"][0]["requests"][0]
            self.assertEqual(request["first_content_s"], .3)
            self.assertNotIn("prompt", request)
            self.assertNotIn("sample", request)
            self.assertEqual(json.loads((root / "data.json").read_text()), data)

    def test_export_rejects_unproven_measurement_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "summary.json"
            path.write_text("{}")
            with self.assertRaises(ValueError):
                export_data([path], root)


if __name__ == "__main__":
    unittest.main()
