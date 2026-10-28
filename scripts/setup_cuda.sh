#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# First load your site's Python/CUDA modules if required.
if [[ "$(uname -s)" != Linux ]]; then
  echo "CUDA setup requires Linux." >&2
  exit 1
fi
# Override with e.g. PYTHON=python3.14 bash scripts/setup_cuda.sh.
PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN. Set PYTHON to its name or path." >&2
  exit 1
fi
"$PYTHON_BIN" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 15) else "Python 3.10–3.14 is required by vLLM 0.20.0.")'
"$PYTHON_BIN" --version
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[cuda]'
.venv/bin/python -m pip check
echo "Check nvidia-smi and vLLM driver compatibility before submitting."
