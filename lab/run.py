"""Start a backend, run controlled cells, and preserve experiment provenance."""
import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from lab.bench import benchmark
from lab.workloads import make_workload

CONFIGS = {
    "serial": (1, False, False, 8192),
    "batch": (32, False, False, 8192),
    "cache": (32, True, False, 8192),
    "chunk512": (32, False, True, 512),
    "chunk2048": (32, False, True, 2048),
    "combined": (32, True, True, 512),
}


def command(args, config):
    if args.backend == "mock":
        return [sys.executable, "-m", "lab.mock_server", "--port", str(args.port)]
    if args.backend == "mlx":
        return [sys.executable, "-m", "mlx_lm.server", "--model", args.model,
                "--host", "127.0.0.1", "--port", str(args.port)]
    seqs, cache, chunk, budget = CONFIGS[config]
    return [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
            "--host", "127.0.0.1", "--port", str(args.port),
            "--dtype", "half", "--max-model-len", "8192",
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-num-seqs", str(seqs), "--max-num-batched-tokens", str(budget),
            "--enable-prefix-caching" if cache else "--no-enable-prefix-caching",
            "--enable-chunked-prefill" if chunk else "--no-enable-chunked-prefill",
            "--seed", str(args.seed)]


def fetch(url, path):
    try:
        with urlopen(url, timeout=3) as response:
            Path(path).write_bytes(response.read())
    except Exception as exc:
        Path(path).write_text(f"Unavailable: {exc}\n")


def wait_ready(proc, url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited ({proc.returncode}); inspect server.log")
        try:
            with urlopen(url + "/v1/models", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError("Server startup timed out; inspect server.log")


def provenance():
    versions = {}
    for package in ("vllm", "mlx-lm", "mlx", "mlx-metal", "torch", "transformers", "aiohttp"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"platform": platform.platform(), "machine": platform.machine(),
            "python": sys.version, "packages": versions,
            "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


class MockTokenizer:
    """Only used in explicit mock mode; never a real token count."""
    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return bytes(ids).decode(errors="replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mlx", "vllm", "mock"], required=True)
    parser.add_argument("--model")
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--profile", choices=["mini", "full"], default="mini")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--concurrency", nargs="+", type=int)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()
    mini = args.profile == "mini"
    args.model = args.model or ("mlx-community/Qwen2.5-0.5B-Instruct-4bit" if args.backend == "mlx"
                                else "Qwen/Qwen2.5-1.5B-Instruct")
    if args.backend == "vllm" and args.tokenizer == "Qwen/Qwen2.5-0.5B-Instruct":
        args.tokenizer = args.model
    counts = args.requests if args.requests is not None else (8 if mini else 128)
    repeats = args.repeats if args.repeats is not None else (1 if mini else 3)
    concurrency = args.concurrency if args.concurrency is not None else ([1, 2] if mini else [1, 8, 32])
    output_tokens = args.output_tokens if args.output_tokens is not None else (24 if mini else 128)
    if counts < 1 or repeats < 1 or output_tokens < 1 or min(concurrency) < 1:
        parser.error("Counts, concurrency and output tokens must be positive")
    if output_tokens > 1024:
        parser.error("Use <=1024 output tokens to preserve context headroom")
    if args.request_timeout < 1 or args.startup_timeout < 1 or args.warmup_requests < 1:
        parser.error("Timeouts and warmup count must be positive")
    if args.backend == "mlx" and not mini:
        parser.error("MLX is for the mini validation profile; use vllm for full experiments")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "environment.json").write_text(json.dumps(provenance(), indent=2))
    with (args.output / "pip-freeze.txt").open("w") as file:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=file, check=False)
    if args.backend == "mock":
        tokenizer = MockTokenizer()
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        (args.output / "tokenizer.json").write_text(json.dumps({
            "name": args.tokenizer, "commit": tokenizer.init_kwargs.get("_commit_hash")}, indent=2))
    base = f"http://127.0.0.1:{args.port}"
    # Refuse to benchmark an unrelated pre-existing service.
    import socket
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"Port {args.port} is already occupied; use --port")
    for repeat in range(repeats):
        configs = list(args.configs) if args.backend == "vllm" else [args.backend]
        random.Random(args.seed + repeat).shuffle(configs)
        for config in configs:
            server_dir = args.output / f"repeat-{repeat}" / config
            server_dir.mkdir(parents=True)
            cmd = command(args, config)
            (server_dir / "command.json").write_text(json.dumps(cmd, indent=2))
            with (server_dir / "server.log").open("w") as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                gpu = None
                gpu_log = None
                try:
                    wait_ready(proc, base, args.startup_timeout)
                    fetch(base + "/v1/models", server_dir / "models.json")
                    try:
                        gpu_log = (server_dir / "gpu.csv").open("w")
                        gpu = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,uuid,name,memory.used,utilization.gpu,power.draw",
                                                "--format=csv", "-l", "1"], stdout=gpu_log, stderr=subprocess.DEVNULL)
                    except FileNotFoundError:
                        pass
                    for workload in ("unique", "shared", "mixed"):
                        for c in concurrency:
                            cell = server_dir / f"{workload}-c{c}"
                            # Each cell gets fresh cache keys; identical text is replayed across configs.
                            ns = f"seed{args.seed}-repeat{repeat}-{workload}-c{c}"
                            measured = make_workload(tokenizer, workload, counts, args.seed, ns, mini)
                            warmup = make_workload(tokenizer, workload, args.warmup_requests,
                                                   args.seed, ns, mini, id_offset=counts)
                            metadata = {"backend": args.backend, "config": config, "repeat": repeat,
                                        "model": args.model, "tokenizer": args.tokenizer,
                                        "workload": workload, "concurrency": c, "seed": args.seed,
                                        "profile": args.profile, "synthetic": args.backend == "mock",
                                        "requested_output_tokens": output_tokens,
                                        "warmup_requests": args.warmup_requests,
                                        "cache_regime": "warm shared prefix; fresh cell namespace",
                                        "arrival_process": "closed-loop fixed concurrency"}
                            fetch(base + "/metrics", server_dir / f"{workload}-c{c}-metrics-before.txt")
                            result = asyncio.run(benchmark(base, args.model, measured, c, output_tokens,
                                                          args.backend, cell, metadata, warmup,
                                                          timeout=args.request_timeout))
                            fetch(base + "/metrics", server_dir / f"{workload}-c{c}-metrics-after.txt")
                            print(f"{config} r{repeat} {workload} c{c}: "
                                  f"{result['successful']}/{counts}, {result['requests_per_s']:.2f} req/s", flush=True)
                            if result["failed"]:
                                raise RuntimeError(f"Failed requests in {cell}; inspect requests.jsonl")
                finally:
                    if gpu:
                        gpu.terminate()
                        gpu.wait(timeout=10)
                    if gpu_log is not None:
                        gpu_log.close()
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
    print(f"Completed. Create report: python -m lab.report {args.output}")


if __name__ == "__main__":
    main()
