#!/usr/bin/env bash
set -uo pipefail

MODEL_PATH=""
TARGET_REPO=""
UPSTREAM_REPO=""
ENGINE=""
PYTHON_BIN="python"
REQUIRED_DEVICES="1"
RUN_DIR=""
REQUIRE_READY="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --target-repo) TARGET_REPO="$2"; shift 2 ;;
    --upstream-repo) UPSTREAM_REPO="$2"; shift 2 ;;
    --engine) ENGINE="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --required-devices) REQUIRED_DEVICES="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --require-ready) REQUIRE_READY="1"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$RUN_DIR" ]] || { echo "--run-dir is required" >&2; exit 2; }
[[ -n "$ENGINE" ]] || { echo "--engine is required" >&2; exit 2; }
mkdir -p "$RUN_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=(
  --output "$RUN_DIR/environment.json"
  --engine "$ENGINE"
  --python "$PYTHON_BIN"
  --required-devices "$REQUIRED_DEVICES"
)
[[ -n "$MODEL_PATH" ]] && ARGS+=(--model-path "$MODEL_PATH")
[[ -n "$TARGET_REPO" ]] && ARGS+=(--target-repo "$TARGET_REPO")
[[ -n "$UPSTREAM_REPO" ]] && ARGS+=(--upstream-repo "$UPSTREAM_REPO")

"$PYTHON_BIN" "$SCRIPT_DIR/collect_environment.py" "${ARGS[@]}" > "$RUN_DIR/environment-summary.txt" 2>&1
COLLECT_RC=$?
if [[ $COLLECT_RC -ne 0 ]]; then
  echo "environment collection failed; see $RUN_DIR/environment-summary.txt" >&2
  exit $COLLECT_RC
fi

READINESS="$($PYTHON_BIN - "$RUN_DIR/environment.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(json.load(f).get("readiness", "unknown"))
PY
)"

echo "P800 assessment readiness: $READINESS"
if [[ "$REQUIRE_READY" == "1" && "$READINESS" != "ready" && "$READINESS" != "degraded" ]]; then
  exit 1
fi
exit 0
