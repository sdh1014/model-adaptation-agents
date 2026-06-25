#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

"$PYTHON_BIN" -m py_compile \
  "$ROOT/scripts/model_runtime.py" \
  "$ROOT/scripts/migrate_runbooks.py" \
  "$ROOT/scripts/validation/"*.py \
  "$ROOT/scripts/benchmark/"*.py
for file in \
  "$ROOT/scripts/validation/"*.sh \
  "$ROOT/scripts/benchmark/"*.sh \
  "$ROOT/examples/runbooks/"*/checks/*.sh \
  "$ROOT/install.sh"; do
  bash -n "$file"
done
"$PYTHON_BIN" -m unittest discover -s "$ROOT/tests" -p 'test_*.py' -v
