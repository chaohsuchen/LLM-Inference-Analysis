# LLM Inference Lab: When Do Serving Optimizations Help?

A Linux CUDA/SLURM experiment platform for measuring the latency–throughput
trade-off under shared agent policies and mixed-length tool observations.

**Workflow:** upload source → run real GPU experiments → analyze measured results
→ build an interactive animation → publish the code and visualization on GitHub.

There is one inference engine: vLLM. Experiments generate real model responses.
Prompt workloads are deterministic synthetic inputs inspired by agent services;
they do not execute tools or implement a complete autonomous agent.

## 1. Upload and install

Extract the source ZIP into a directory you own on your Linux cluster. Requirements:
Python 3.11, a compatible NVIDIA driver, a GPU allocation, and network access or
prefetched packages/weights. Load site-specific Python/CUDA modules if required.

```bash
cd llm-inference-lab
bash scripts/setup_cuda.sh
```

This creates a project-local virtual environment without administrator access.
The CUDA extra pins vLLM 0.20.0. Follow its official driver/toolchain requirements;
avoid overriding its Torch/CUDA dependencies independently.

If compute nodes cannot access the internet, prefetch the public default model
on a permitted node, using storage shared with your compute allocation:

```bash
.venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen2.5-1.5B-Instruct")'
```

You can set `HF_HOME` to a shared cache before both prefetching and submitting,
or provide a local snapshot path through `MODEL`. Use your site's supported
offline installation workflow if it also restricts package downloads.

## 2. Submit a real GPU smoke test

From the project root, replace the partition/account values with your site's
settings. Supply a GPU type using its exact resource syntax if required.

```bash
PROFILE=smoke sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT scripts/slurm.sbatch
```

The smoke profile runs the same real model and prompt lengths as the full study,
with fewer requests: 8 per cell, concurrency 1 and 2, one repetition, and a
24-token output length. It validates all six serving configurations. It is not
enough data for research conclusions about p99.

Inspect `slurm-JOB_ID.out`, server logs and request failures before the full run.
If your site restricts environment export, submit with
`sbatch --export=ALL,PROFILE=smoke ... scripts/slurm.sbatch`.

Inside an existing GPU allocation, the equivalent direct command is:

```bash
.venv/bin/python -m lab.run --profile smoke --output results/gpu-smoke
.venv/bin/python -m lab.report results/gpu-smoke
```

Do not run GPU experiments on an unallocated login node.

## 3. Run the full experiment

```bash
PROFILE=full sbatch --partition=YOUR_PARTITION --account=YOUR_ACCOUNT scripts/slurm.sbatch
```

Defaults: Qwen2.5-1.5B-Instruct, FP16, one GPU, context limit 8192, 128 output
tokens, 128 measured requests per cell, 2 warmups per cell, and 3 repetitions.
The matrix is 6 configurations × 3 workloads × 3 concurrency levels × 3 repeats
= **162 cells**. Concurrency levels are 1, 8 and 32. The six-hour SLURM limit
is configurable and is not a runtime guarantee.

Use one homogeneous GPU for a comparison. Never pool A4500 and A6000 measurements
as though they were repetitions on the same hardware. The small default model
keeps the first study practical; verify memory availability on the allocated GPU.

The Python entry point is `lab/run.py:main()`:

```text
scripts/slurm.sbatch → lab.run
  → start vLLM for a configuration
  → generate deterministic prompts with workloads.py
  → stream real responses and measure them with bench.py
  → preserve raw records, metrics, logs and GPU samples
  → report.py produces tables, plots and animation data
```

Configuration order is randomized within each repetition. The engine restarts
for each configuration/repetition. Each cell has fresh cache keys and identical
replayed input text/order across configurations. Warmups prime shared prefixes
and are excluded from the timed region. Warmup IDs are disjoint from measured IDs.

For a reduced exploratory pilot inside a GPU allocation:

```bash
.venv/bin/python -m lab.run --profile full \
  --configs batch cache chunk512 --requests 64 --repeats 1 \
  --concurrency 1 8 --output results/pilot
.venv/bin/python -m lab.report results/pilot
```

For stronger tail estimates, increase `--requests` and repetitions after checking
runtime. 128 requests per cell provides limited evidence near p99.

## Research design

| Configuration | Max sequences | Prefix cache | Chunked prefill | Token budget |
|---|---:|---|---|---:|
| serial | 1 | off | off | 8192 |
| batch | 32 | off | off | 8192 |
| cache | 32 | on | off | 8192 |
| chunk512 | 32 | off | on | 512 |
| chunk2048 | 32 | off | on | 2048 |
| combined | 32 | on | on | 512 |

| Hypothesis | Comparison | Main evidence |
|---|---|---|
| Batching helps more at higher concurrency | serial vs batch, unique | throughput, TTFT, GPU utilization |
| Prefix reuse benefits shared policies | batch vs cache, shared and unique control | TTFT, cache-counter deltas |
| Prefill budgets change short/long interference | batch vs chunk512 vs chunk2048, mixed | short TTFT p95, long TTFT, throughput |
| Techniques interact | combined vs cache and chunk512 | latency–throughput trade-off |

`serial` limits active sequences to one; it does not disable vLLM's scheduler.
Chunk comparisons change prefill policy and token budget together, so report
them as scheduling-configuration comparisons rather than an isolated switch effect.

Workloads:

- `unique`: roughly 256 policy tokens, with a unique ID near the start to prevent
  long prefix reuse.
- `shared`: roughly 2048 shared policy tokens followed by a unique task.
- `mixed`: 25% long (~3072 tokens), 75% short (~256), with early unique IDs.

Exact lengths including wrappers are recorded using the model tokenizer and
server-reported usage. Inputs are raw completions prompts to avoid chat-template
drift. Generation is greedy; vLLM ignores EOS to control output lengths. Forced
continuations can be unnatural, and this experiment makes no answer-quality claim.

## Measurements and saved evidence

- TTFT: dispatch to first nonempty content event, including server queueing and
  HTTP overhead. This is client-observed first-content latency.
- End-to-end latency: dispatch to stream completion. Time in the client's
  undispatched work queue is excluded. Load is closed-loop fixed concurrency.
- TPOT: first-to-last content duration / (reported output tokens − 1). It is an
  approximation under stream coalescing, and unavailable for a single content event.
- Throughput: successful output tokens / timed batch wall duration including drain.
  Missing token usage produces `null`; stream chunks are never counted as tokens.
- Quantiles: linear interpolation over successful requests. Failures are counted
  separately, saved, and cause the run to stop after the cell. There are no retries.
- GPU evidence: one-second `nvidia-smi` samples of memory, utilization and power.
  Raw `/metrics` snapshots before/after each cell preserve cache counters. They
  do not establish per-request memory or scheduler activity.
- Replay timestamps: dispatch, first content, last content and completion, measured
  in seconds relative to each measured cell start. These are client HTTP events.

Each result directory also records package versions, the tokenizer identity,
server commands/model metadata, complete workloads, per-request records and
per-cell summaries. Keep these files for reproducibility and preserve an immutable
model snapshot for later replication. Dependency ranges are recorded in pip-freeze.txt.

## 4. Build the interactive animation after collecting results

The SLURM script generates `results/cuda-JOB_ID/report/` containing:

| File | Purpose |
|---|---|
| `REPORT.md` | Per-comparison tables and measurement caveats |
| `cells.csv` | All cell summaries |
| `tradeoffs.png` | Static latency/throughput figure |
| `data.json` | Measured summaries and request events for an interactive replay |

The JSON export omits prompt text, output samples, and error text. Review model
paths and other identifiers in the summary metadata before publishing.

The planned animation will let readers select configuration, workload, concurrency
and repetition; replay requests; and compare throughput with latency. It should
label waiting-for-first-content separately from observed response streaming.
It must not invent GPU queue stages, KV-cache hits, or exact token timestamps.

The frontend is intentionally deferred until real cluster data exists. See
[docs/VISUALIZATION.md](docs/VISUALIZATION.md) for the data contract and
[docs/GITHUB.md](docs/GITHUB.md) for manual publication and GitHub Pages preparation.

## Verification and limitations

```bash
.venv/bin/python -m unittest discover -s tests -v
```

These are offline arithmetic/export tests; they do not replace the real GPU smoke
run. Linux CUDA execution must be validated on your cluster. The repository ships
no experiment results. Summaries use medians across repeats and observed min/max,
not confidence intervals. Small samples, synthetic prompts, closed-loop traffic
and model size limit generalization.

NVIDIA results do not measure AMD/ROCm performance. The reproducible methodology
and observed scheduling trade-offs are the relevant research discussion. A future
AMD replication requires its own supported ROCm environment.

Use [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md) to record supported findings,
negative results and limitations. Do not substitute predicted speedups for data.

## References and troubleshooting

- [vLLM 0.20.0 CLI](https://docs.vllm.ai/en/v0.20.0/cli/serve/)
- [vLLM optimization](https://docs.vllm.ai/en/v0.20.0/configuration/optimization/)
- [GPU installation requirements](https://docs.vllm.ai/en/v0.20.0/getting_started/installation/gpu/)

If a server fails to start, inspect its `server.log` and confirm the installed
vLLM version. For a busy port, pass `--port 8010`. For timeouts, adjust
`--startup-timeout` or `--request-timeout`. Increase warmups with
`--warmup-requests`. If memory is insufficient, record any model/concurrency
changes and rerun all paired configurations consistently.
