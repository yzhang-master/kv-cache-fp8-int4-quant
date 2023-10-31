#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHONPATH="$ROOT/src" "${PYTHON_BIN:-python}" -m kvquant.bench "$@"
