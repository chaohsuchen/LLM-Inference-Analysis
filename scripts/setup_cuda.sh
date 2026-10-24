#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# First load your site's Python/CUDA modules if required.
if [[ "$(uname -s)" != Linux ]]; then
  echo "CUDA setup requires Linux." >&2
  exit 1
fi
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[cuda]'
.venv/bin/python -m pip check
echo "Check nvidia-smi and vLLM driver compatibility before submitting."
