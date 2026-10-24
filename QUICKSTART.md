# Start here

The Desktop project also has a local Git repository. See
[`docs/GITHUB.md`](docs/GITHUB.md) for saving changes and manually publishing
your experiment results to GitHub later.

The Mac environment was installed and a real 0.5B Qwen MLX mini experiment
completed successfully: **48/48 measured requests**, six cells, no failures.
Five automated metric/HTTP/workload tests also passed.

Open `results/mac-pinned/report/REPORT.md` and `tradeoffs.png` for the measured
Mac results. This small run validates the workflow. It is not evidence for vLLM
optimization speedups. CUDA/SLURM execution still needs to be verified on your
cluster. All code and documentation are in English.

## Run again on this Mac

If using the prepared project in the Codex outputs folder:

```bash
bash scripts/mini_mac.sh
```

If you extracted the archive or use the copy in your Desktop project folder:

```bash
bash scripts/setup_mac.sh
bash scripts/mini_mac.sh
```

## Run on your server

Upload/extract the source archive. Do not upload the Mac virtual environment.

```bash
bash scripts/setup_cuda.sh
# Add your actual partition/account flags:
sbatch --partition=YOUR_PARTITION scripts/slurm.sbatch
```

Read the README first for model prefetching, GPU smoke testing, cluster-specific
resource flags, full experiment design, and interpretation. The default full
matrix contains 162 cells and three independent repetitions per configuration.

## One-day deliverable

Aim for a working repository, saved raw cluster data, one latency/throughput
figure, and two evidence-backed conclusions. Use `docs/RESEARCH_NOTES.md` to
record the reasoning. Queue delays and driver setup can affect the schedule;
run the pilot before launching the full matrix.
