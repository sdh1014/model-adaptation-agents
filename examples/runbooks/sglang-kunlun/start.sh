#!/usr/bin/env bash
set -euo pipefail

cd "$TARGET_REPO"

# 将这里替换为项目当前已验证的完整 SGLang-Kunlun 启动命令。
exec "${RUNTIME_PYTHON:-python}" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host "$MODEL_HOST" \
  --port "$MODEL_PORT" \
  --tp-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --dtype bfloat16
