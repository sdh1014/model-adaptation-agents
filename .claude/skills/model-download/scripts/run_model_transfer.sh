#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/model_download_config.sh"

if [[ "${1:-}" == "--config" ]]; then
  CONFIG_PATH="$2"
  shift 2
fi

source "$CONFIG_PATH"

require_value() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo "Missing required config: $name in $CONFIG_PATH" >&2
    exit 2
  fi
}

require_value "Local_model_path" "$Local_model_path"

cmd=(
  python3 "$SCRIPT_DIR/model_transfer.py"
  --source "$Source"
  --local-dir "$Local_model_path"
  --max-workers "$Max_workers"
  --upload-concurrency "$Upload_concurrency"
  --progress-interval "$Progress_interval"
)

case "$Source" in
  hf)
    require_value "HF_model_path" "$HF_model_path"
    cmd+=(--model-id "$HF_model_path")
    ;;
  modelscope)
    require_value "MS_model_path" "$MS_model_path"
    cmd+=(--model-id "$MS_model_path")
    ;;
  local)
    ;;
  *)
    echo "Source must be hf, modelscope, or local: $Source" >&2
    exit 2
    ;;
esac

if [[ -n "$Proxy_url" ]]; then
  cmd+=(--proxy "$Proxy_url")
fi

if [[ "$Upload" == "1" ]]; then
  require_value "BOS_model_path" "$BOS_model_path"
  cmd+=(--upload --bos-path "$BOS_model_path")
fi

if [[ "$Private_model" == "1" ]]; then
  cmd+=(--private)
fi

if [[ "$Skip_upload_confirmation" == "1" ]]; then
  cmd+=(--yes)
fi

cmd+=("$@")

printf '[CMD]'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
