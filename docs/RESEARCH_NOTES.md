# Experiment worksheet

## Problem and hypotheses

Agent services often reuse policy/tool descriptions and mix short questions with
long tool observations. Test whether reuse and prefill scheduling improve user
latency under fixed hardware resources. State the predicted outcome BEFORE
inspecting the experiment; accept negative or mixed results.

## Run identity

- GPU name / UUID and memory:
- Backend version / model snapshot / precision:
- Driver / CUDA / Python:
- Result directory and exact commands:
- Measured request counts, repeats, concurrency and output lengths:

## Observations (fill only after measurements)

| Comparison | Workload/concurrency | TTFT p50/p99 | Short TTFT p95 | Output tok/s | Evidence files |
|---|---|---|---|---|---|
| serial → batch | | | | | |
| batch → cache | shared | | | | |
| batch → cache | unique control | | | | |
| batch → chunk512 | mixed | | | | |
| chunk512 → chunk2048 | mixed | | | | |

Speedup = optimized throughput / baseline throughput.
Latency reduction = 1 − optimized latency / baseline latency.
Report absolute numbers alongside ratios, with repeat ranges and failure counts.

## Cache evidence

Inspect the before/after Prometheus snapshots. Identify the exact metric names,
then calculate counter deltas within a cell rather than using cumulative totals.
Warmup is already complete at the before snapshot. Check unique controls as well
as shared-prefix requests. A latency reduction alone does not prove a cache hit.

## Interpretation

1. Which hypothesis was supported, contradicted, or inconclusive?
2. Did faster throughput worsen short/long request latency?
3. Did failure rate, output length, tokenization, or preemption confound the result?
4. What changes would you test next, and why?

## Limits and AMD relevance

Synthetic agent-inspired prompts, closed-loop arrivals, small model and finite
samples limit generalization. No actual agent tool execution or answer-quality
evaluation occurs. Apple/NVIDIA observations do not measure ROCm or AMD GPUs.
A future AMD replication should use a supported ROCm vLLM environment, preserve
the same prompts and protocol, and report hardware/backend differences explicitly.
Do not install NVIDIA wheels on ROCm hosts.

## Interview explanation

Describe one measured trade-off, the controlled comparison that supports it,
one surprising/negative result, and one limitation. Cite raw evidence. Avoid
claiming a scheduler implementation or AMD speedup that this project did not test.
