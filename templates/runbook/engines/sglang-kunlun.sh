#!/usr/bin/env bash
set -euo pipefail

cd "$TARGET_REPO"

# 将当前 SGLang-Kunlun 环境已验证的完整命令粘贴到这里。
# 示例（参数以当前版本 --help 为准）：
# exec "${RUNTIME_PYTHON:-python}" -m sglang.launch_server \
#   --model-path "$MODEL_PATH" \
#   --host "$MODEL_HOST" \
#   --port "$MODEL_PORT" \
#   --tp "${TENSOR_PARALLEL_SIZE:-1}"

echo "MODEL_RUN_NOT_CONFIGURED: 请编辑 SGLang-Kunlun start.sh" >&2
exit 64
