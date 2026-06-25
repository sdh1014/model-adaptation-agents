#!/usr/bin/env bash
set -euo pipefail
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -X POST "${MODEL_BASE_URL}/v1/chat/completions" \
  -d "$(cat <<JSON
{
  \"model\": \"${MODEL_NAME}\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],
  \"temperature\": 0,
  \"max_tokens\": 8
}
JSON
)" | tee "$RUN_DIR/smoke-response.json"
