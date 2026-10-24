# LLM Inference Lab: When Do Serving Optimizations Help?

An experimental platform for studying the latency–throughput trade-off under
agent-inspired shared policies and mixed-length tool observations.

**Research question:** Under what concurrency and prompt-sharing conditions do
batching, prefix caching, and chunked prefill improve serving efficiency?

This project uses existing inference engines. It does not implement a scheduler,
PagedAttention, tool execution, or an autonomous agent. NVIDIA measurements do
not establish AMD hardware performance. The portable experiment methodology,
cache analysis, and scheduling trade-offs are the relevant AMD team discussion.

## Start on an M3 Mac

Requirements: Apple Silicon, Python 3.11 (or `uv` installed), internet for the first
dependency/model download, and enough free unified memory for a 0.5B 4-bit model.

```bash
cd llm-inference-lab
bash scripts/setup_mac.sh
bash scripts/mini_mac.sh
```

The mini run starts a local MLX server, runs 3 workloads at concurrency 1 and 2
(8 measured requests plus 2 warmups per cell, 24-token output cap), stops the
server, and writes a report. The default model is
`mlx-community/Qwen2.5-0.5B-Instruct-4bit`. Allow time for the first download.
MLX may stop at EOS earlier than the output cap. Inspect actual output counts.

The Mac validates requests, streaming metrics, artifacts, and the workload design.
It does not validate vLLM switches or predict CUDA/ROCm speedups. Engines,
quantization, model sizes, and hardware differ; never plot Mac and CUDA speed
as a controlled optimization comparison.

```bash
# No model download: explicitly synthetic protocol check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m lab.run --backend mock --output results/mock-check
.venv/bin/python -m lab.report results/mock-check
```

## Upload and run on SLURM

Upload this project without `.venv`, `__pycache__`, or local caches. A Mac virtual
environment cannot be reused on Linux. Use one GPU per job; the default 1.5B FP16
model is intentionally modest for the A4500/A6000 class. Actual memory availability
and software compatibility must be checked on your node.

```bash
cd llm-inference-lab
# Load site-specific Python/CUDA modules first, if your site requires them.
bash scripts/setup_cuda.sh

# Prefetch on a node with internet if compute nodes cannot download models.
.venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen2.5-1.5B-Instruct")'

# Substitute your cluster's partition/account/GPU resource syntax.
sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT scripts/slurm.sbatch
```

If your site uses GPU types, pass its exact `--gres=gpu:TYPE:1` syntax. Do not mix
GPU types within a comparison. Submit from the project root; the script resolves
it through `SLURM_SUBMIT_DIR`. Model downloads must be available on shared storage
if they are prefetched outside the compute node. Increase the job time if needed.
The configured six hours is a limit, not a runtime guarantee.

Run a cluster smoke test before the complete matrix:

```bash
# Inside an allocated GPU job or interactive allocation:
.venv/bin/python -m lab.run --backend vllm --profile mini \
  --configs batch cache chunk512 --output results/cuda-smoke
.venv/bin/python -m lab.report results/cuda-smoke
```

Full run: 6 engine configurations × 3 workloads × 3 client concurrency levels ×
3 repetitions = 162 cells, with 128 measured requests/cell and 128 output tokens.
The runner randomizes configuration order within each repetition. Each server is
fresh per configuration/repetition. Each cell uses fresh prompt namespaces to
avoid cache reuse across cells; warmups prime shared prefixes within the cell.
All configurations replay the same workload order and text for a given cell.

If time is tight, run a smaller pilot before committing to all cells:

```bash
.venv/bin/python -m lab.run --backend vllm --profile full \
  --configs batch cache chunk512 --repeats 1 --requests 64 \
  --concurrency 1 8 --output results/pilot
```

This pilot is exploratory; it is not the complete experiment. To assess p99
reliably, increase request counts (for example `--requests 1000`) and repetitions.
128 requests/cell gives only a few observations near the upper tail.

## Controlled comparisons

| Config | Max sequences | Prefix cache | Chunked prefill | Token budget |
|---|---:|---|---|---:|
| serial | 1 | off | off | 8192 |
| batch | 32 | off | off | 8192 |
| cache | 32 | on | off | 8192 |
| chunk512 | 32 | off | on | 512 |
| chunk2048 | 32 | off | on | 2048 |
| combined | 32 | on | on | 512 |

vLLM always uses its scheduler. `serial` restricts active sequences; it is **not**
a continuous-batching-off implementation. Both client concurrency and engine
sequence limit affect observed queueing. The chunked-prefill comparison changes
prefill policy and token budget together: describe it as a scheduling configuration
comparison, not the isolated effect of a Boolean switch.

| Hypothesis | Compare | Evidence to inspect |
|---|---|---|
| Batching pays off at higher concurrency | serial vs batch, unique | throughput, TTFT, GPU utilization |
| Prefix reuse helps shared policies more than unique requests | batch vs cache, shared AND unique | TTFT, actual prompt tokens, cache metric deltas |
| Smaller prefill budgets reduce short-request interference | batch vs chunk512 vs chunk2048, mixed | short TTFT p95, long TTFT, throughput |
| Techniques interact | combined vs cache and chunk512 | the same latency/throughput measures |

`unique`: approximately 256 policy tokens with a unique identifier near the start.
`shared`: approximately 2048 identical policy tokens followed by a unique task.
`mixed`: 25% long (~3072 tokens), 75% short (~256), unique early identifiers.
Exact lengths including wrappers are recorded using the model tokenizer and
server-reported usage. Tokenizer decode/encode can move lengths slightly. Mini
uses smaller prefixes. These repeated runbooks are synthetic workload structure,
not a representative real agent dataset or a language-quality benchmark.

Requests use the completions endpoint and a fixed raw prompt format, avoiding
chat-template drift. FP16, model, seed, output length cap, and context limit remain
fixed on the cluster. EOS is ignored in vLLM for controlled output lengths; this
is a throughput workload and can force unnatural continuations. It makes no
model-quality preservation claim. Inspect generated samples manually.

## What is measured

- TTFT: request dispatch to first nonempty content event; includes server queueing
  and HTTP overhead. It is a client-observed estimate of first-token latency.
- End-to-end: dispatch to stream completion. No time waiting in the client's
  undispatched work queue is included. This is closed-loop fixed-concurrency load,
  not an open-loop production arrival process or saturation proof.
- TPOT: time from first to last content event divided by (reported output tokens − 1).
  Stream coalescing makes this an approximation; no per-token ITL p99 is claimed.
- Throughput: successful output tokens / measured batch wall time, including the
  drain. Only server usage counts are used. Missing usage is `null`, never inferred
  from chunk counts. Request throughput and TTFT-SLO goodput are also recorded.
- p50/p95/p99: linear interpolation over successful requests; failures are counted
  separately and terminate a run after persisting the failing cell.
- GPU memory/utilization/power: one-second `nvidia-smi` samples. `/metrics` snapshots
  before/after each cell preserve vLLM cache counters. They are raw evidence;
  the report does not invent cache-hit rates or claim per-request GPU memory.

Warmups are excluded from timings. This studies a warm shared-prefix regime,
not cold-start cache latency. Unique warmup IDs are disjoint from measured IDs.
Two requests warm the engine; inspect stability and expand warmups for rigorous
publication work. There are no retries hiding overload or network errors.

## Results and provenance

Each result directory contains environment/package versions, exact server commands,
server model metadata and logs, complete generated workloads, per-request records,
per-cell summaries, GPU samples (CUDA), and raw server metrics where available.
`report/cells.csv`, `REPORT.md`, and `tradeoffs.png` are produced by:

```bash
.venv/bin/python -m lab.report results/YOUR_RUN
```

Reports show medians across repeats with observed min/max, not confidence intervals.
Do not pool unrelated machines or report synthetic mock numbers as model results.
Dependency ranges ease first setup; preserve `pip-freeze.txt` and immutable model
snapshot paths for later reproduction. The CUDA engine is pinned to vLLM 0.20.0;
its official installation requirements determine the necessary driver/toolchain.
Do not force a different Torch/CUDA wheel into that environment.

The Mac extra pins MLX LM 0.29.1 with MLX 0.31.1. A first test with MLX 0.32.4
hit a cross-thread GPU-stream failure in that MLX LM version; the earlier core
version is used to keep this small-model workflow stable. See the related
[upstream stream compatibility report](https://github.com/ml-explore/mlx-lm/issues/1179).
`requirements-mac-tested.txt` records the tested Mac dependency versions; use it
only on macOS if you need to recreate that environment (`pip install -r
requirements-mac-tested.txt` followed by `pip install -e .`).
Use `--warmup-requests` to increase warmups and `--request-timeout` to tune the
per-request timeout (default 180 seconds).

## One-day execution plan

1. First hour: complete the Mac mini run and inspect raw requests and plots.
2. Next hour: upload, check drivers, install the Linux environment, prefetch weights,
   and pass a GPU smoke test.
3. Next 3–5 hours: run the full matrix on one homogeneous GPU. Use the pilot to
   estimate runtime; reduce scope explicitly if the queue or runtime prevents it.
4. Final 1–2 hours: check failures/output lengths/cache counters, compare hypotheses,
   write two supported observations and one limitation, then polish the repository.

Use [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md) as the results worksheet.
Do not substitute expected speedups for measured data.

## Verified upstream references

- [MLX LM server](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md)
- [vLLM 0.20.0 CLI flags](https://docs.vllm.ai/en/v0.20.0/cli/serve/)
- [vLLM optimization and tuning](https://docs.vllm.ai/en/v0.20.0/configuration/optimization/)
- [vLLM GPU installation](https://docs.vllm.ai/en/v0.20.0/getting_started/installation/gpu/)

## Troubleshooting

- Port occupied: pass `--port 8010` to `lab.run`.
- Startup error or unsupported flag: inspect `server.log`, verify installed vLLM
  is 0.20.0, and run the mini GPU profile before the full matrix.
- GPU OOM: ensure another job is not sharing the GPU; reduce active concurrency
  and/or the model size, record the changed settings, and rerun all paired configs.
- Compute nodes have no internet: prefetch packages and model snapshots using
  your site's supported workflow, then use shared cache or local model paths.
- Model download/authentication error: the defaults are public; check outbound
  access and available disk. Point `--model` and `--tokenizer` at local snapshots.
- MLX fails: check the macOS requirement of the resolved MLX release and inspect
  the saved server log. The mock profile tests the harness without MLX.
