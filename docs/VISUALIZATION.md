# Interactive replay of measured inference results

Build this stage after the real cluster experiment. The experiment runner and
exporter are ready; the interactive frontend is not implemented yet.

## Data contract

`python -m lab.report results/cuda-JOB_ID` creates `report/data.json`:

- `schema_version`: 1.
- `measurement_source`: `live_vllm_http`.
- `timebase`: seconds since the start of each measured cell; cells are independent.
- `cells`: an array, each containing a `summary` and `requests`.
- `summary`: config, workload, concurrency, repetition, model, counts, metrics,
  warmup regime, and measurement provenance.
- `requests`: sorted by dispatch time, with the following selected fields.

| Field | Meaning |
|---|---|
| `id`, `class` | Request ID within the cell and short/long classification |
| `dispatch_s` | Client begins sending the HTTP request |
| `first_content_s` | First nonempty response content arrives |
| `last_content_s` | Last nonempty response content arrives |
| `completion_s` | Client finishes handling the response or error |
| `ok` | Whether the request completed successfully |
| `input_tokens`, `server_prompt_tokens` | Local and server input counts |
| `output_tokens` | Server-reported output count; may be null |
| `ttft_s`, `e2e_s`, `tpot_s` | Client latency measurements |
| `content_chunks`, `finish_reason` | Stream observations |

Null first-content times can occur for failed requests. Do not coerce missing
measurements to zero. Request IDs repeat across cells; combine the selected cell
identity with the ID for a global UI key. Prompt/output text is not exported.

## Planned interaction

1. Choose workload, configuration, concurrency and repetition.
2. Play/pause the request timeline or scrub time manually.
3. Show each request transitioning from dispatched/waiting to receiving content,
   then complete or failed.
4. Compare paired configurations side by side using the same workload/concurrency.
5. Show measured throughput, latency quantiles, failure counts and repeat ranges.

Time before first content includes HTTP overhead, server queueing and prefill.
It cannot be labeled solely as queue time or prefill time. The span between first
and last content is an observed streaming span, not a GPU kernel trace.
There are no per-token timestamps; do not depict invented exact token emission.
There are no per-request cache hits or GPU scheduling events. Explain those
mechanisms separately from a clearly labeled conceptual illustration if desired.

## GitHub publication

Copy the reviewed JSON to `docs/results/data.json`. A later static frontend at
`docs/index.html` can load it with a relative fetch and run entirely in the browser.
Do not embed API keys or connect viewers to the cluster. The repository README
should link to the eventual published visualization and include a static figure
for readers who do not open it.
