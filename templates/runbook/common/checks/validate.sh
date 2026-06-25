#!/usr/bin/env bash
set -euo pipefail

# 最简：直接粘贴完整验证命令，并确保断言失败时返回非零。
# 推荐：使用辅助库生成逐 case 结构化结果。
#
# source "$CONTROL_ROOT/scripts/validation/lib.sh"
# validation_init
# validation_case full-suite required -- bash /path/to/validate_all.sh
# validation_finish

echo "MODEL_RUN_NOT_CONFIGURED: 请编辑 $RUNBOOK_DIR/checks/validate.sh" >&2
exit 64
