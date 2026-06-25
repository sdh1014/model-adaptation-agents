#!/usr/bin/env bash
set -o pipefail

TARGET_REPO=""
UPSTREAM_REPO=""
RUN_DIR=""
PYTHON_BIN="${PYTHON_BIN:-}"
ARCHITECTURES=()
MODEL_TYPES=()
MODEL_FAMILIES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-repo) TARGET_REPO="$2"; shift 2 ;;
    --upstream-repo) UPSTREAM_REPO="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --architecture) ARCHITECTURES+=("$2"); shift 2 ;;
    --model-type) MODEL_TYPES+=("$2"); shift 2 ;;
    --model-family) MODEL_FAMILIES+=("$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -d "$TARGET_REPO" ]] || { echo "invalid --target-repo: $TARGET_REPO" >&2; exit 2; }
[[ -n "$RUN_DIR" ]] || { echo "--run-dir is required" >&2; exit 2; }
mkdir -p "$RUN_DIR/static-scan"

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

git -C "$TARGET_REPO" status --short > "$RUN_DIR/repo-status-before.txt" 2>&1 || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ARGS=(
  --engine vllm-kunlun
  --target-repo "$TARGET_REPO"
  --output "$RUN_DIR/static-scan/repositories.json"
)
[[ -n "$UPSTREAM_REPO" ]] && ARGS+=(--upstream-repo "$UPSTREAM_REPO")
for value in "${ARCHITECTURES[@]}"; do ARGS+=(--architecture "$value"); done
for value in "${MODEL_TYPES[@]}"; do ARGS+=(--model-type "$value"); done
for value in "${MODEL_FAMILIES[@]}"; do ARGS+=(--model-family "$value"); done

"$PYTHON_BIN" "$ROOT_DIR/scripts/assess/inspect_target_repo.py" "${ARGS[@]}"
RC=$?

git -C "$TARGET_REPO" status --short > "$RUN_DIR/repo-status-after-static-scan.txt" 2>&1 || true
exit $RC
