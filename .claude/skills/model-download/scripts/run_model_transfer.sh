#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="$SCRIPT_DIR/model_download_config.sh"

# Pass 1: pick the config file (needed before sourcing). Keep the rest.
rest=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"; shift 2 ;;
    *)
      rest+=("$1"); shift ;;
  esac
done
set -- ${rest[@]+"${rest[@]}"}

source "$CONFIG_PATH"

# Pass 2: wrapper-level overrides so the model URL, local path, and BOS path can
# be given as skill arguments instead of editing the config. Arguments always win
# over config values. Everything else (e.g. --dry-run, --preflight) is passed
# through to model_transfer.py.
passthrough=()
positional_url=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --url|--model-url|--model-id)
      Model_url="$2"; shift 2 ;;
    --local|--local-dir)
      Local_model_path="$2"; shift 2 ;;
    --bos|--bos-path)
      BOS_model_path="$2"; shift 2 ;;
    --source)
      Source="$2"; shift 2 ;;
    --pack)
      Pack="$2"; shift 2 ;;
    --upload)
      Upload=1; shift ;;
    --no-upload)
      Upload=0; shift ;;
    -*)
      passthrough+=("$1"); shift ;;
    *)
      if [[ -z "$positional_url" ]]; then
        positional_url="$1"
      else
        passthrough+=("$1")
      fi
      shift ;;
  esac
done

# A positional URL argument overrides whatever the config set.
if [[ -n "$positional_url" ]]; then
  Model_url="$positional_url"
fi

# Preflight only checks tooling; it needs no model or path values.
for a in ${passthrough[@]+"${passthrough[@]}"}; do
  if [[ "$a" == "--preflight" ]]; then
    exec python3 "$SCRIPT_DIR/model_transfer.py" --preflight
  fi
done

require_value() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" ]]; then
    echo "Missing required value: $name (set it in $CONFIG_PATH or pass it as an argument)" >&2
    exit 2
  fi
}

require_value "Local_model_path" "$Local_model_path"

cmd=(
  python3 "$SCRIPT_DIR/model_transfer.py"
  --local-dir "$Local_model_path"
  --max-workers "$Max_workers"
  --upload-concurrency "$Upload_concurrency"
  --progress-interval "$Progress_interval"
)

# Prefer a single Model_url (hf or modelscope auto-detected). Fall back to the
# per-source fields for backward compatibility.
if [[ -n "${Model_url:-}" ]]; then
  cmd+=(--model-id "$Model_url")
  if [[ -n "${Source:-}" ]]; then
    cmd+=(--source "$Source")
  fi
else
  case "${Source:-}" in
    hf)
      require_value "HF_model_path" "$HF_model_path"
      cmd+=(--source hf --model-id "$HF_model_path")
      ;;
    modelscope)
      require_value "MS_model_path" "$MS_model_path"
      cmd+=(--source modelscope --model-id "$MS_model_path")
      ;;
    local)
      cmd+=(--source local)
      ;;
    "")
      echo "Provide a model URL (Model_url or as an argument), or set Source to hf|modelscope|local in $CONFIG_PATH" >&2
      exit 2
      ;;
    *)
      echo "Source must be hf, modelscope, or local: $Source" >&2
      exit 2
      ;;
  esac
fi

if [[ -n "$Proxy_url" ]]; then
  cmd+=(--proxy "$Proxy_url")
fi

if [[ -n "${Pack:-}" ]]; then
  cmd+=(--pack "$Pack")
fi

if [[ "$Upload" == "1" ]]; then
  require_value "BOS_model_path" "$BOS_model_path"
  cmd+=(--upload --bos-path "$BOS_model_path")
elif [[ -n "${BOS_model_path:-}" ]]; then
  # Download-only: still pass the BOS path so the final manual-upload hint is exact.
  cmd+=(--bos-path "$BOS_model_path")
fi

if [[ "$Skip_upload_confirmation" == "1" ]]; then
  cmd+=(--yes)
fi

cmd+=(${passthrough[@]+"${passthrough[@]}"})

printf '[CMD]'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
