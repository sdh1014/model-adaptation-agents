#!/usr/bin/env bash
set -euo pipefail

cd "$TARGET_REPO"

# 将当前 vLLM-Kunlun 环境已验证的完整命令粘贴到这里。
# 示例（参数以当前版本 --help 为准）：
# exec "${RUNTIME_PYTHON:-python}" -m vllm.entrypoints.openai.api_server \
#   --model "$MODEL_PATH" \
#   --served-model-name "$MODEL_NAME" \
#   --host "$MODEL_HOST" \
#   --port "$MODEL_PORT" \
#   --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}"

echo "MODEL_RUN_NOT_CONFIGURED: 请编辑 vLLM-Kunlun start.sh" >&2
exit 64
