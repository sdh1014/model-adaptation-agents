#!/usr/bin/env bash
set -euo pipefail

TARGET_REPO=""
RUN_DIR=""
BASE_REF=""
BEFORE_SNAPSHOT=""
TIMEOUT_SECONDS="600"
ALLOW=()
PYTHON_BIN="${PYTHON_BIN:-}"

usage() {
  cat >&2 <<'EOF'
Usage:
  check_implementation.sh \
    --target-repo <path> \
    --run-dir <path> \
    --base-ref <git-ref> \
    --before-snapshot <repo-before.json> \
    --allow <glob> [--allow <glob> ...] \
    [--timeout-seconds <n>] \
    -- <verification command and args>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-repo) TARGET_REPO="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --base-ref) BASE_REF="$2"; shift 2 ;;
    --before-snapshot) BEFORE_SNAPSHOT="$2"; shift 2 ;;
    --allow) ALLOW+=("$2"); shift 2 ;;
    --timeout-seconds) TIMEOUT_SECONDS="$2"; shift 2 ;;
    --) shift; break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -d "$TARGET_REPO" ]] || { echo "invalid --target-repo: $TARGET_REPO" >&2; exit 2; }
[[ -n "$RUN_DIR" ]] || { echo "--run-dir is required" >&2; exit 2; }
[[ -n "$BASE_REF" ]] || { echo "--base-ref is required" >&2; exit 2; }
[[ -f "$BEFORE_SNAPSHOT" ]] || { echo "valid --before-snapshot is required" >&2; exit 2; }
[[ ${#ALLOW[@]} -gt 0 ]] || { echo "at least one --allow pattern is required" >&2; exit 2; }
[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "--timeout-seconds must be a positive integer" >&2; exit 2; }
[[ $# -gt 0 ]] || { echo "an explicit verification command is required after --" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$RUN_DIR/pre-verification" "$RUN_DIR/post-verification"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

scope_args=()
for pattern in "${ALLOW[@]}"; do
  scope_args+=(--allow "$pattern")
done

# Check the code edits before running tests. If the edit already exceeds scope,
# do not execute the verification command.
"$PYTHON_BIN" "$SCRIPT_DIR/snapshot_repo.py" \
  --target-repo "$TARGET_REPO" \
  --run-dir "$RUN_DIR/pre-verification" \
  --phase after \
  --base-ref "$BASE_REF"

set +e
"$PYTHON_BIN" "$SCRIPT_DIR/check_scope.py" \
  --target-repo "$TARGET_REPO" \
  --base-ref "$BASE_REF" \
  --before-snapshot "$BEFORE_SNAPSHOT" \
  "${scope_args[@]}" \
  --output "$RUN_DIR/scope-before.json"
SCOPE_BEFORE_RC=$?
set -e

COMMAND_RC=-1
if [[ $SCOPE_BEFORE_RC -eq 0 ]]; then
  set +e
  "$PYTHON_BIN" "$SCRIPTS_DIR/run_bash.py" \
    --run-dir "$RUN_DIR/command" \
    --cwd "$TARGET_REPO" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    -- "$@"
  COMMAND_RC=$?
  set -e
fi

# Verification itself can create or modify files. Capture and scope-check the
# final tree as well, even when the command fails or times out.
"$PYTHON_BIN" "$SCRIPT_DIR/snapshot_repo.py" \
  --target-repo "$TARGET_REPO" \
  --run-dir "$RUN_DIR/post-verification" \
  --phase after \
  --base-ref "$BASE_REF"

set +e
"$PYTHON_BIN" "$SCRIPT_DIR/check_scope.py" \
  --target-repo "$TARGET_REPO" \
  --base-ref "$BASE_REF" \
  --before-snapshot "$BEFORE_SNAPSHOT" \
  "${scope_args[@]}" \
  --output "$RUN_DIR/scope-after.json"
SCOPE_AFTER_RC=$?
set -e

"$PYTHON_BIN" - "$RUN_DIR/summary.json" "$SCOPE_BEFORE_RC" "$COMMAND_RC" "$SCOPE_AFTER_RC" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output = Path(sys.argv[1])
scope_before = int(sys.argv[2])
command = int(sys.argv[3])
scope_after = int(sys.argv[4])
if scope_before != 0:
    status = "scope_violation_before_verification"
elif scope_after != 0:
    status = "scope_violation_after_verification"
elif command != 0:
    status = "verification_failed"
else:
    status = "passed"
payload = {
    "schema_version": 1,
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "status": status,
    "scope_before_exit_code": scope_before,
    "verification_exit_code": command,
    "scope_after_exit_code": scope_after,
    "command_executed": command >= 0,
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

if [[ $SCOPE_BEFORE_RC -ne 0 || $SCOPE_AFTER_RC -ne 0 ]]; then
  exit 4
fi
if [[ $COMMAND_RC -ne 0 ]]; then
  exit "$COMMAND_RC"
fi
exit 0
