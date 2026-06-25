#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-$PWD}"
DEST="$(cd "$DEST" && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="$DEST/.model-run-backup/$STAMP"

mkdir -p "$BACKUP"

backup_path() {
  local relative="$1"
  if [[ -e "$DEST/$relative" ]]; then
    mkdir -p "$BACKUP/$(dirname "$relative")"
    mv "$DEST/$relative" "$BACKUP/$relative"
  fi
}

backup_path ".claude/skills/model-run"
backup_path ".claude/skills/model-runtime"
backup_path "scripts/model_runtime.py"
backup_path "scripts/model_runtime"

mkdir -p \
  "$DEST/.claude/skills" \
  "$DEST/scripts" \
  "$DEST/knowledge/common/runtime" \
  "$DEST/knowledge/engines/vllm-kunlun" \
  "$DEST/knowledge/engines/sglang-kunlun" \
  "$DEST/integration"

cp -a "$SOURCE_DIR/.claude/skills/model-run" "$DEST/.claude/skills/"
cp -a "$SOURCE_DIR/scripts/model_runtime.py" "$DEST/scripts/"
cp -a "$SOURCE_DIR/knowledge/common/runtime/runbook.md" "$DEST/knowledge/common/runtime/"
cp -a "$SOURCE_DIR/knowledge/engines/vllm-kunlun/run.md" "$DEST/knowledge/engines/vllm-kunlun/"
cp -a "$SOURCE_DIR/knowledge/engines/sglang-kunlun/run.md" "$DEST/knowledge/engines/sglang-kunlun/"
cp -a "$SOURCE_DIR/integration/adapt-validate.md" "$DEST/integration/"
chmod +x "$DEST/scripts/model_runtime.py"

printf 'installed: %s\n' "$DEST"
printf 'backup:    %s\n' "$BACKUP"
printf 'next: merge CLAUDE.addendum.md into %s/CLAUDE.md\n' "$DEST"
printf 'then: /model-run <model-id>/<target-id> --init\n'
