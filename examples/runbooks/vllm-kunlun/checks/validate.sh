#!/usr/bin/env bash
set -euo pipefail

source "$CONTROL_ROOT/scripts/validation/lib.sh"
validation_init

# 直接把已有测试命令放在 validation_case 后面。
# 命令退出 0 表示该 case 满足 oracle；退出 64 表示缺少必要输入。

validation_case api-smoke required -- \
  bash "$RUNBOOK_DIR/checks/smoke.sh"

# 示例：固定输入的参考结果比较。请替换为项目实际命令。
if [[ -n "${REFERENCE_RESPONSE_JSON:-}" && -f "${REFERENCE_RESPONSE_JSON:-}" ]]; then
  validation_case deterministic-generation required -- \
    python "$CONTROL_ROOT/scripts/validation/compare_json.py" \
      --reference "$REFERENCE_RESPONSE_JSON" \
      --actual "$RUN_DIR/smoke-response.json" \
      --ignore id \
      --ignore created \
      --output "$RUN_DIR/validation/deterministic-comparison.json"
else
  validation_mark deterministic-generation required blocked \
    "请设置 REFERENCE_RESPONSE_JSON，或替换为项目的 logits/token parity 命令"
fi

# 根据 model-analysis.md 增加 batch、long-context、TP、MoE 等 case。
validation_finish
