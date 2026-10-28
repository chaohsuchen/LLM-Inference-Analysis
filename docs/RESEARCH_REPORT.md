# Faster serving depends on what you ask the model to do

**Have you ever wondered what happens to your short question when someone else sends a long document to the same AI server?** Does your request get through quickly, or do you spend longer staring at an empty response?

That question led me to a broader one: when several users share one GPU, which serving optimizations actually improve their experience? A server can produce more text per second while some users still wait longer for the first word. So a single “faster” number would not tell the whole story.

To investigate, I built a controlled experiment with real model responses and synthetic prompts. I compared processing requests together, reusing work on shared prompt beginnings, and splitting long-input processing into smaller pieces. I measured both total output and the wait for the first response, then repeated the comparisons at different levels of concurrent demand.

The results gave me a useful surprise: a smaller processing budget improved typical first-response latency but made the slower short requests wait longer under heavier load. Here is how I designed the experiment, checked the evidence, and worked out what that trade-off means.

![Three measured findings](figures/research_highlights.png)

## How I turned the question into an experiment

I used Qwen2.5-1.5B-Instruct through vLLM 0.20.0 on one RTX 4090 with 24 GB of memory. The full run began on October 24, 2026. I kept the model, FP16 precision, GPU allocation, and requested output length fixed. Every successful measured request produced 128 output tokens.

I created three synthetic workloads. The first gave each request a distinct beginning, so there was little reusable prefix. The second reused a roughly 2,048-token policy before a unique task. The third mixed 75% short inputs with 25% long inputs, representing short questions arriving alongside larger observations. These prompts resemble an infrastructure assistant's inputs, but the experiment never calls real tools.

Then I compared six serving configurations: one active sequence, batching up to 32 sequences, batching with prefix caching, chunked prefill with budgets of 512 or 2,048 tokens, and caching combined with the 512-token configuration. I tested client concurrency of 1, 8, and 32. Each cell had two warmup requests followed by 128 measured requests. Three repetitions produced 162 cells and 20,736 measured requests. None failed. The separate smoke test is excluded from these findings.

The controls matter. I replayed identical input text and order across configurations within each workload/concurrency/repetition, used fresh cell namespaces, and randomized configuration order within each repetition. Each configuration started a new server. Workload and concurrency order within a server stayed fixed, so that ordering remains a possible source of bias.

My main measurements were output tokens per second and time to first nonempty response content. The latter includes the server wait and HTTP overhead after dispatch. It excludes time sitting in the client's undispatched queue. This is a fixed-concurrency, closed-loop test: when one request finishes, a worker sends another.

## Finding 1: batching helped when there was concurrent work

On unique prompts at concurrency 32, the one-active-sequence configuration produced **225.9 output tokens/s**, while batching produced **4,534.4 tokens/s**: a **20.1× throughput ratio**. Across the three runs, those values ranged from 225.8–226.0 and 4,494.2–4,536.6 respectively.

Median first-content latency also fell from **17.58 seconds to 0.223 seconds**. The baseline accepts concurrent HTTP requests but processes only one active sequence, so much of that wait is consistent with queueing. This comparison measures the benefit over that deliberately constrained baseline; it does not compare vLLM against another serving engine.

At concurrency 1, both configurations produced about 224 tokens/s. That control is useful: the large gain appeared when there was enough simultaneous work to batch.

## Finding 2: caching helped when there was something to reuse

On shared-policy inputs at concurrency 32, enabling prefix caching increased throughput from **1,755.7 to 4,044.3 tokens/s**, a **2.30× ratio**. Median first-content latency fell from **0.782 to 0.268 seconds**, about a 66% reduction. Throughput ranges across repetitions were 1,746.8–1,758.5 without caching and 4,013.2–4,084.1 with caching.

I also checked the unique-prompt control. Throughput changed from **4,534.4 to 4,529.1 tokens/s**, about −0.1%. That is essentially unchanged at this experiment's resolution. The combination of a clear shared-input benefit and a near-flat control supports the interpretation that prefix reuse matters here.

The published cache-counter snapshots provide additional evidence, with an important limitation: the runner takes the “before” snapshot before warmup. Their differences therefore include warmup and measured traffic. They cannot be presented as measured-request-only hit rates or individual request cache hits.

## Finding 3: a better median can hide a worse experience for slower requests

The mixed workload gave the most useful caution. At concurrency 32:

| Configuration | Output tokens/s | First-content p50, all requests | First-content p95, short requests |
|---|---:|---:|---:|
| Batch | 2,705.9 | 456 ms | 796 ms |
| Chunk 2048 | 2,707.3 | 276 ms | 833 ms |
| Chunk 512 | 2,439.8 | 236 ms | 1,082 ms |

Each entry is the median of three per-run values, rather than a quantile computed by pooling requests across runs.

With the 512-token budget, typical first-content latency improved about **48%**, but the short-request p95 worsened about **36%**, and throughput fell about **9.8%**. Short-request p95 ranged from 783–811 ms for batch and 1,022–1,125 ms for chunk512. The direction was consistent across the three observed runs.

At concurrency 8, the same comparison improved short-request p95 from 257 to 212 ms. So the behavior changed with load. I would choose a configuration by the latency target and expected traffic, rather than assuming a smaller budget always helps short requests.

These comparisons change both chunked-prefill policy and token budget. They establish a configuration trade-off, not the isolated causal effect of a single switch. Client timestamps also cannot reveal the exact GPU scheduling sequence responsible for it.

## What I would and would not conclude

The evidence supports three narrow conclusions: batching helped under concurrent load; prefix caching helped when inputs shared a long prefix; and a smaller prefill budget traded better median latency for worse short-request tail latency under heavier mixed load.

The study has limits. There is one small model, one GPU allocation, synthetic repeated text, three repetitions, and 128 requests per cell. That is useful for an exploratory comparison, but weak evidence for stable p99 estimates. The ranges shown are observed min–max values, not confidence intervals. Repetitions share one experimental session, and configuration randomization does not remove every time-dependent effect.

Greedy generation with EOS ignored makes output lengths comparable but can produce unnatural continuations. No answer-quality evaluation was performed. Closed-loop traffic does not reproduce bursty open-loop arrivals. The tokenizer metadata has a null commit and the model revision was not pinned, so exact weight/tokenizer replication is not guaranteed. These NVIDIA observations do not establish performance on AMD hardware, other GPUs, larger models, or production workloads.

## What I would investigate next

I would first repeat the mixed-workload comparison with more requests, more independent sessions, and randomized cell order to test whether the tail-latency penalty holds up. I would add open-loop arrivals to study queue growth at a fixed offered load. Then I would sweep input lengths, shared-prefix proportions, and prefill budgets, pin the model snapshot, and compare cold and warm cache regimes explicitly.

For a stronger mechanism explanation, I would add scheduler traces and take cache snapshots after warmup. That would help separate what the HTTP client observed from what happened inside the engine.

## How to check this work

The [evidence guide](results/README.md) links all 162 cell summaries, 20,736 compressed request records, cache snapshots, package versions, and file hashes. `scripts/validate_evidence.py` independently recomputes the request-derived metrics with the benchmark's summary function. `scripts/research_figure.py` rebuilds the figure from the CSV without a GPU. The [methodology](METHODOLOGY.md) describes the full protocol and rerun commands.

The research contribution is connecting a testable question to controls and measurements, then explaining the trade-offs—including a result that goes against the simple “smaller chunks are better” expectation.
