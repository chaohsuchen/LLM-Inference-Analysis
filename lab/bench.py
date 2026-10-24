"""Streaming HTTP benchmark. Closed-loop concurrency, no automatic retries."""
import asyncio
import json
import math
import time
from pathlib import Path

import aiohttp


def percentile(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    pos = (len(values) - 1) * q
    low = math.floor(pos)
    return values[low] + (values[math.ceil(pos)] - values[low]) * (pos - low)


async def request(session, base_url, model, row, output_tokens, backend, timeout):
    payload = {"model": model, "prompt": row["prompt"], "stream": True,
               "max_tokens": output_tokens, "temperature": 0,
               "stream_options": {"include_usage": True}}
    if backend == "vllm":
        payload["ignore_eos"] = True
    start = time.perf_counter()
    first = last = None
    chunks = 0
    usage = {}
    finish = None
    done = False
    sample = ""
    result = {k: v for k, v in row.items() if k != "prompt"}
    try:
        async with session.post(base_url.rstrip("/") + "/v1/completions",
                                json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:500]}")
            # aiohttp's line iterator handles arbitrary network chunk boundaries.
            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                    content = choice.get("text") or ""
                    if content:
                        now = time.perf_counter()
                        first = now if first is None else first
                        last = now
                        chunks += 1
                        sample = (sample + content)[:240]
        if not done or first is None or finish is None:
            raise RuntimeError("Incomplete stream, missing finish reason, or no generated text")
        tokens = usage.get("completion_tokens")
        result.update(ok=True, error=None, ttft_s=first - start,
                      e2e_s=time.perf_counter() - start,
                      tpot_s=(last - first) / (tokens - 1) if tokens and tokens > 1 else None,
                      output_tokens=tokens, server_prompt_tokens=usage.get("prompt_tokens"),
                      content_chunks=chunks, finish_reason=finish, sample=sample)
    except Exception as exc:
        result.update(ok=False, error=f"{type(exc).__name__}: {exc}",
                      ttft_s=None, e2e_s=time.perf_counter() - start,
                      tpot_s=None, output_tokens=None, sample=sample)
    return result


def summarize(rows, wall_s, slo_s):
    ok = [r for r in rows if r["ok"]]
    tokens_known = bool(ok) and all(r.get("output_tokens") is not None for r in ok)
    result = {"requests": len(rows), "successful": len(ok),
              "failed": len(rows) - len(ok), "wall_s": wall_s,
              "requests_per_s": len(ok) / wall_s,
              "output_tokens_per_s": sum(r["output_tokens"] for r in ok) / wall_s if tokens_known else None,
              "ttft_slo_s": slo_s,
              "ttft_goodput_req_s": sum(r["ttft_s"] <= slo_s for r in ok) / wall_s,
              "observed_input_tokens_min": min((r["input_tokens"] for r in rows), default=0),
              "observed_input_tokens_max": max((r["input_tokens"] for r in rows), default=0)}
    for label, subset in [("all", ok), ("short", [r for r in ok if r["class"] == "short"]),
                          ("long", [r for r in ok if r["class"] == "long"])]:
        result[label + "_count"] = len(subset)
        for metric in ("ttft_s", "e2e_s", "tpot_s"):
            for p in (50, 95, 99):
                result[f"{label}_{metric}_p{p}"] = percentile([r[metric] for r in subset], p / 100)
    return result


async def benchmark(base_url, model, rows, concurrency, output_tokens, backend,
                    out_dir, metadata, warmup_rows=(), timeout=180, slo_s=2):
    if concurrency < 1 or not rows:
        raise ValueError("Positive concurrency and nonempty workload required")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "workload.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    queue = asyncio.Queue()
    for row in rows:
        queue.put_nowait(row)
    results = []
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=concurrency)) as session:
        for row in warmup_rows:
            result = await request(session, base_url, model, row, output_tokens, backend, timeout)
            if not result["ok"]:
                raise RuntimeError(f"Warmup failed: {result['error']}")
        with (out_dir / "requests.jsonl").open("w") as file:
            async def worker():
                while not queue.empty():
                    row = queue.get_nowait()
                    result = await request(session, base_url, model, row, output_tokens, backend, timeout)
                    results.append(result)
                    file.write(json.dumps(result) + "\n")
                    file.flush()
            start = time.perf_counter()
            await asyncio.gather(*(worker() for _ in range(min(concurrency, len(rows)))))
            wall = time.perf_counter() - start
    summary = {**metadata, **summarize(results, wall, slo_s)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
