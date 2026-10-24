#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-results/mac-$(date +%Y%m%d-%H%M%S)}"
.venv/bin/python -m lab.run --backend mlx --profile mini --output "$OUT"
.venv/bin/python -m lab.report "$OUT"
