#!/usr/bin/env bash
# shellcheck shell=bash

_validation_tools_root() {
  printf '%s\n' "${CONTROL_ROOT:?CONTROL_ROOT 未设置}/scripts/validation"
}

_validation_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
  elif command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "python3"
  else
    printf '%s\n' "python"
  fi
}

_validation_require_init() {
  : "${RUN_DIR:?RUN_DIR 未设置}"
  export VALIDATION_DIR="${VALIDATION_DIR:-$RUN_DIR/validation}"
  [[ -f "$VALIDATION_DIR/events.jsonl" ]] || {
    echo "validation 尚未初始化，请先调用 validation_init" >&2
    return 64
  }
}

_validation_slug() {
  local value="$1"
  [[ "$value" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "非法 validation case 名称: $value" >&2
    return 64
  }
  printf '%s\n' "$value"
}

validation_init() {
  : "${RUN_DIR:?RUN_DIR 未设置}"
  export VALIDATION_DIR="${VALIDATION_DIR:-$RUN_DIR/validation}"
  "$(_validation_python)" "$(_validation_tools_root)/cases.py" init --output-dir "$VALIDATION_DIR" "$@" >/dev/null
}

validation_case() {
  local name="${1:?缺少 case 名称}"
  local level="${2:?缺少 required|optional}"
  shift 2
  [[ "${1:-}" == "--" ]] && shift
  _validation_require_init || return $?
  local slug
  slug="$(_validation_slug "$name")" || return $?
  [[ "$level" == "required" || "$level" == "optional" ]] || {
    echo "level 必须为 required 或 optional" >&2
    return 64
  }
  if [[ "$#" -eq 0 ]]; then
    "$(_validation_python)" "$(_validation_tools_root)/cases.py" record \
      --output-dir "$VALIDATION_DIR" --name "$name" --level "$level" \
      --status blocked --reason "未提供验证命令" >/dev/null
    return 0
  fi

  local case_dir="$VALIDATION_DIR/cases/$slug"
  mkdir -p "$case_dir"
  local stdout="$case_dir/stdout.log"
  local stderr="$case_dir/stderr.log"
  local started ended duration rc status
  started="$(date +%s%N)"
  if "$@" >"$stdout" 2>"$stderr"; then
    rc=0
    status=passed
  else
    rc=$?
    if [[ "$rc" -eq 64 ]]; then
      status=blocked
    else
      status=failed
    fi
  fi
  ended="$(date +%s%N)"
  duration="$(awk -v start="$started" -v end="$ended" 'BEGIN { printf "%.6f", (end-start)/1000000000 }')"
  "$(_validation_python)" "$(_validation_tools_root)/cases.py" record \
    --output-dir "$VALIDATION_DIR" --name "$name" --level "$level" \
    --status "$status" --exit-code "$rc" --duration-seconds "$duration" \
    --stdout "$stdout" --stderr "$stderr" >/dev/null
  return 0
}

validation_mark() {
  local name="${1:?缺少 case 名称}"
  local level="${2:?缺少 required|optional}"
  local status="${3:?缺少状态}"
  local reason="${4:-}"
  _validation_require_init || return $?
  _validation_slug "$name" >/dev/null || return $?
  case "$status" in
    blocked|skipped|not_applicable) ;;
    *) echo "validation_mark 只接受 blocked、skipped、not_applicable" >&2; return 64 ;;
  esac
  "$(_validation_python)" "$(_validation_tools_root)/cases.py" record \
    --output-dir "$VALIDATION_DIR" --name "$name" --level "$level" \
    --status "$status" --reason "$reason" >/dev/null
}

validation_finish() {
  _validation_require_init || return $?
  "$(_validation_python)" "$(_validation_tools_root)/cases.py" finalize --output-dir "$VALIDATION_DIR"
}
