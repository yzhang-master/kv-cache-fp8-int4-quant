#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHONPATH="$ROOT/src" "${PYTHON_BIN:-python}" -m pytest -q "$ROOT/tests/test_smoke.py"
