"""Export all cells and plot medians across independent repetitions."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = [json.loads(p.read_text()) for p in sorted(args.results.rglob("summary.json"))]
    if not rows:
        raise SystemExit("No summary.json files found")
    environments = {(r["backend"], r["model"], r["profile"]) for r in rows}
    if len(environments) != 1:
        raise SystemExit("Report one backend/model/profile at a time to avoid invalid comparisons")
    report = args.results / "report"
    report.mkdir(exist_ok=True)
    with (report / "cells.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)
    metrics = ["output_tokens_per_s", "all_ttft_s_p50", "all_ttft_s_p99", "short_ttft_s_p95"]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["workload"], row["concurrency"], row["config"])].append(row)
    title = "SYNTHETIC PROTOCOL TEST" if rows[0]["synthetic"] else "Measured serving results"
    lines = [f"# {title}", "", f"Backend: {rows[0]['backend']}; model: {rows[0]['model']}.", "",
             "Each table value is the median across runs, with [min, max]. Per-run quantiles are in cells.csv.",
             "A p99 from a small sample is descriptive only. Do not claim stable tail latency from the mini run.", "",
             "| Workload | Concurrency | Config | Runs | Output tok/s | TTFT p50 (s) | TTFT p99 (s) | Short TTFT p95 (s) |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for key, values in sorted(groups.items()):
        cells = []
        for metric in metrics:
            numbers = [v[metric] for v in values if v.get(metric) is not None]
            cells.append(f"{statistics.median(numbers):.3f} [{min(numbers):.3f}, {max(numbers):.3f}]" if numbers else "unavailable")
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(values)} | " + " | ".join(cells) + " |")
    failed = sum(r["failed"] for r in rows)
    lines += ["", f"Failed requests: {failed}.", "",
              "No automatic causal conclusion is made. Compare serial/batch on unique, batch/cache on shared and unique,",
              "and batch/chunk512/chunk2048 on mixed. Chunk comparisons vary both prefill policy and token budget.",
              "Combined is an interaction check. Check actual output lengths, raw samples, GPU traces, and prefix-cache metrics.",
              "Missing token usage produces unavailable token throughput; streaming chunks are never counted as tokens."]
    (report / "REPORT.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    workloads = sorted({r["workload"] for r in rows})
    fig, axes = plt.subplots(2, len(workloads), figsize=(5 * len(workloads), 7), squeeze=False)
    for col, workload in enumerate(workloads):
        for config in sorted({r["config"] for r in rows}):
            keys = sorted(k for k in groups if k[0] == workload and k[2] == config)
            for index, metric in enumerate(("output_tokens_per_s", "all_ttft_s_p99")):
                points = []
                for key in keys:
                    vals = [v[metric] for v in groups[key] if v.get(metric) is not None]
                    if vals:
                        points.append((key[1], statistics.median(vals), min(vals), max(vals)))
                if points:
                    x, y, low, high = zip(*points)
                    axes[index, col].errorbar(x, y, yerr=[[a-b for a,b in zip(y,low)], [b-a for a,b in zip(y,high)]],
                                             marker="o", capsize=3, label=config)
        for index in range(2):
            axes[index, col].set_title(workload)
            axes[index, col].set_xlabel("Closed-loop client concurrency")
            axes[index, col].set_ylabel("Output tokens/s" if index == 0 else "TTFT p99 (s)")
            axes[index, col].grid(alpha=0.25)
            if axes[index, col].lines:
                axes[index, col].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(report / "tradeoffs.png", dpi=160)
    plt.close(fig)
    print(report / "REPORT.md")


if __name__ == "__main__":
    main()
