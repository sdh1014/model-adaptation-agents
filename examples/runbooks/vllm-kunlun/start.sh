#!/usr/bin/env bash
set -euo pipefail

cd "$TARGET_REPO"

# 将这里替换为项目当前已验证的完整 vLLM-Kunlun 启动命令。
exec "${RUNTIME_PYTHON:-python}" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host "$MODEL_HOST" \
  --port "$MODEL_PORT" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --dtype bfloat16 \
  --max-model-len 4096
