#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x /usr/bin/python3 ]]; then
  PYTHON_BIN=/usr/bin/python3
else
  PYTHON_BIN=python3
fi
"$PYTHON_BIN" -m compileall -q "$ROOT/scripts" "$ROOT/tests"
while IFS= read -r -d '' file; do bash -n "$file"; done < <(find "$ROOT/scripts" "$ROOT/templates" -type f -name '*.sh' -print0)
cd "$ROOT"
for test in \
  tests.test_core.RuntimeTests.test_engine_overlay_and_sync \
  tests.test_core.RuntimeTests.test_smoke_cleanup \
  tests.test_core.RuntimeTests.test_persistent_lifecycle \
  tests.test_core.StageTests.test_validate_and_benchmark \
  tests.test_core.HelperTests.test_model_inspect \
  tests.test_core.HelperTests.test_implement_scope_and_run \
  tests.test_core.HelperTests.test_flat_run_paths \
  tests.test_core.HelperTests.test_implement_run_uses_flat_target_key
do
  echo "[TEST] $test"
  "$PYTHON_BIN" -m unittest -v "$test"
done
