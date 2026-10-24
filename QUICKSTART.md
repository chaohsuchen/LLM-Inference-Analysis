# Server-only quickstart

1. Upload/extract `llm-inference-lab.zip` into your own Linux cluster directory.
2. Load your site's required Python/CUDA modules, then install:

```bash
cd llm-inference-lab
bash scripts/setup_cuda.sh
```

3. Prefetch weights on a permitted node if compute nodes lack internet. See README.
4. Submit a real GPU smoke test with your cluster's actual resource flags:

```bash
PROFILE=smoke sbatch --partition=YOUR_PARTITION scripts/slurm.sbatch
```

5. Inspect the job log and saved server logs. After success, submit the full matrix:

```bash
PROFILE=full sbatch --partition=YOUR_PARTITION scripts/slurm.sbatch
```

Add an account or GPU type if your site requires it. Submit from the project root.
The full matrix uses Qwen2.5-1.5B-Instruct, one GPU, 162 cells, and three repetitions.
`lab/run.py:main()` is the experiment entry point. No administrator access is needed.

After the run, retrieve `results/cuda-JOB_ID/`, especially `report/data.json`,
`REPORT.md`, `cells.csv`, and `tradeoffs.png`. Use those measured results to build
the interactive animation. The frontend is a later stage; no results are fabricated.

Read `docs/VISUALIZATION.md` for animation data and `docs/GITHUB.md` for publishing.
CUDA execution still needs validation on your server.
