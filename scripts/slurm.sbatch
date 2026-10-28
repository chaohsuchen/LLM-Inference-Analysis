#!/usr/bin/env bash
#SBATCH --job-name=inference-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
# Submit from the project root. Supply your site's partition/account/GPU type
# via sbatch flags, and load any required modules below.
cd "${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the project root}"
test -x .venv/bin/python || { echo "Run bash scripts/setup_cuda.sh first"; exit 1; }
nvidia-smi
# Set HF_HUB_OFFLINE=1 only after downloading the model onto shared storage.
# Keep one homogeneous GPU per run; never pool A4500/A6000 results.
OUT="results/cuda-${SLURM_JOB_ID}"
srun --ntasks=1 .venv/bin/python -m lab.run \
  --profile "${PROFILE:-full}" --model "${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}" \
  --output "$OUT"
.venv/bin/python -m lab.report "$OUT"
