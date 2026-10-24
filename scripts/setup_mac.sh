#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "This setup requires an Apple Silicon Mac." >&2
  exit 1
fi
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.11 --seed .venv
  uv pip install --python .venv/bin/python -e '.[mac]'
else
  python3.11 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -e '.[mac]'
fi
echo "Ready. Run: bash scripts/mini_mac.sh"
