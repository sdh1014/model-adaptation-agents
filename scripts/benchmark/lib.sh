#!/usr/bin/env bash
# shellcheck shell=bash

_benchmark_tools_root() {
  printf '%s\n' "${CONTROL_ROOT:?CONTROL_ROOT 未设置}/scripts/benchmark"
}

_benchmark_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "$PYTHON_BIN"
  elif command -v python3 >/dev/null 2>&1; then
    printf '%s\n' "python3"
  else
    printf '%s\n' "python"
  fi
}

_benchmark_require_init() {
  : "${RUN_DIR:?RUN_DIR 未设置}"
  export BENCHMARK_DIR="${BENCHMARK_DIR:-$RUN_DIR/benchmark}"
  [[ -f "$BENCHMARK_DIR/events.jsonl" ]] || {
    echo "benchmark 尚未初始化，请先调用 benchmark_init" >&2
    return 64
  }
}

_benchmark_name() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "非法 benchmark 名称: $1" >&2
    return 64
  }
}

_benchmark_duration() {
  awk -v start="$1" -v end="$2" 'BEGIN { printf "%.6f", (end-start)/1000000000 }'
}

benchmark_init() {
  : "${RUN_DIR:?RUN_DIR 未设置}"
  export BENCHMARK_DIR="${BENCHMARK_DIR:-$RUN_DIR/benchmark}"
  "$(_benchmark_python)" "$(_benchmark_tools_root)/records.py" init --output-dir "$BENCHMARK_DIR" "$@" >/dev/null
}

benchmark_case() {
  local name="${1:?缺少 case 名称}"
  local level="${2:?缺少 required|optional}"
  shift 2
  _benchmark_require_init || return $?
  _benchmark_name "$name" || return $?
  [[ "$level" == "required" || "$level" == "optional" ]] || {
    echo "level 必须为 required 或 optional" >&2
    return 64
  }
  local warmup=0 repeat=3
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --warmup) warmup="${2:?缺少 warmup 值}"; shift 2 ;;
      --repeat) repeat="${2:?缺少 repeat 值}"; shift 2 ;;
      --) shift; break ;;
      *) echo "未知 benchmark_case 参数: $1" >&2; return 64 ;;
    esac
  done
  [[ "$warmup" =~ ^[0-9]+$ && "$repeat" =~ ^[1-9][0-9]*$ ]] || {
    echo "warmup/repeat 必须是有效整数" >&2
    return 64
  }
  "$(_benchmark_python)" "$(_benchmark_tools_root)/records.py" declare \
    --output-dir "$BENCHMARK_DIR" --case "$name" --level "$level" \
    --warmup "$warmup" --repeat "$repeat" >/dev/null
  if [[ "$#" -eq 0 ]]; then
    "$(_benchmark_python)" "$(_benchmark_tools_root)/records.py" failure \
      --output-dir "$BENCHMARK_DIR" --case "$name" --level "$level" \
      --phase prepare --status blocked --reason "未提供 benchmark 命令" >/dev/null
    return 0
  fi

  local case_dir="$BENCHMARK_DIR/cases/$name"
  mkdir -p "$case_dir/warmup" "$case_dir/runs"
  local i rc started ended duration stdout stderr

  for ((i=1; i<=warmup; i++)); do
    export BENCHMARK_CASE="$name"
    export BENCHMARK_ITERATION="warmup-$i"
    export BENCHMARK_SAMPLE_DIR="$case_dir/warmup/$i"
    export BENCHMARK_SAMPLE_FILE="$BENCHMARK_SAMPLE_DIR/sample.json"
    mkdir -p "$BENCHMARK_SAMPLE_DIR"
    stdout="$BENCHMARK_SAMPLE_DIR/stdout.log"
    stderr="$BENCHMARK_SAMPLE_DIR/stderr.log"
    if "$@" >"$stdout" 2>"$stderr"; then
      :
    else
      rc=$?
      "$(_benchmark_python)" "$(_benchmark_tools_root)/records.py" failure \
        --output-dir "$BENCHMARK_DIR" --case "$name" --level "$level" \
        --iteration "$i" --phase warmup --status "$([[ "$rc" -eq 64 ]] && echo blocked || echo failed)" \
        --exit-code "$rc" --stdout "$stdout" --stderr "$stderr" \
        --reason "warmup 失败" >/dev/null
      return 0
    fi
  done

  for ((i=1; i<=repeat; i++)); do
    export BENCHMARK_CASE="$name"
    export BENCHMARK_ITERATION="$i"
    export BENCHMARK_SAMPLE_DIR="$case_dir/runs/$i"
    export BENCHMARK_SAMPLE_FILE="$BENCHMARK_SAMPLE_DIR/sample.json"
    mkdir -p "$BENCHMARK_SAMPLE_DIR"
    stdout="$BENCHMARK_SAMPLE_DIR/stdout.log"
    stderr="$BENCHMARK_SAMPLE_DIR/stderr.log"
    started="$(date +%s%N)"
    if "$@" >"$stdout" 2>"$stderr"; then
      rc=0
    else
      rc=$?
    fi
    ended="$(date +%s%N)"
    duration="$(_benchmark_duration "$started" "$ended")"
    if [[ "$rc" -ne 0 ]]; then
      "$(_benchmark_python)" "$(_benchmark_tools_root)/records.py" failure \
        --output-dir "$BENCHMARK_DIR" --case "$name" --level "$level" \
        --iteration "$i" --phase measure --status "$([[ "$rc" -eq 64 ]] && echo blocked || echo failed)" \
        --exit-code "$rc" --duration-seconds "$duration" \
        --stdout "$stdout" --stderr "$stderr" --reason "benchmark 命令失败" >/dev/null
      continue
    fi
    if [[ ! -s "$BENCHMARK_SAMPLE_FILE" ]]; then
      "$(_benchmark_python)" "$(_benchmark_tools_root)/records.py" failure \
        --output-dir "$BENCHMARK_DIR" --case "$name" --level "$level" \
        --iteration "$i" --phase collect --status failed --exit-code 66 \
        --duration-seconds "$duration" --stdout "$stdout" --stderr "$stderr" \
        --reason "命令成功但未写入 BENCHMARK_SAMPLE_FILE" >/dev/null
      continue
    fi
    if "$(_benchmark_python)" "$(_benchmark_tools_root)/records.py" sample \
      --output-dir "$BENCHMARK_DIR" --case "$name" --level "$level" \
      --iteration "$i" --sample-file "$BENCHMARK_SAMPLE_FILE" \
      --duration-seconds "$duration" --stdout "$stdout" --stderr "$stderr" >/dev/null; then
      :
    else
      rc=$?
      "$(_benchmark_python)" "$(_benchmark_tools_root)/records.py" failure \
        --output-dir "$BENCHMARK_DIR" --case "$name" --level "$level" \
        --iteration "$i" --phase collect --status failed --exit-code "$rc" \
        --duration-seconds "$duration" --stdout "$stdout" --stderr "$stderr" \
        --reason "样本 JSON 无效或没有数值指标" >/dev/null
    fi
  done
  return 0
}

benchmark_mark() {
  local name="${1:?缺少 case 名称}"
  local level="${2:?缺少 required|optional}"
  local status="${3:?缺少状态}"
  local reason="${4:-}"
  _benchmark_require_init || return $?
  _benchmark_name "$name" || return $?
  case "$status" in
    blocked|skipped|not_applicable) ;;
    *) echo "benchmark_mark 只接受 blocked、skipped、not_applicable" >&2; return 64 ;;
  esac
  "$(_benchmark_python)" "$(_benchmark_tools_root)/records.py" declare \
    --output-dir "$BENCHMARK_DIR" --case "$name" --level "$level" --warmup 0 --repeat 1 >/dev/null
  "$(_benchmark_python)" "$(_benchmark_tools_root)/records.py" mark \
    --output-dir "$BENCHMARK_DIR" --case "$name" --level "$level" \
    --status "$status" --reason "$reason" >/dev/null
}

benchmark_expect() {
  local case_name="${1:?缺少 case}"
  local metric="${2:?缺少 metric}"
  local direction="${3:?缺少 higher|lower}"
  local threshold="${4:?缺少 threshold}"
  local statistic="${5:-median}"
  local unit="${6:-}"
  _benchmark_require_init || return $?
  local -a args=(
    expect --output-dir "$BENCHMARK_DIR" --case "$case_name" --metric "$metric"
    --direction "$direction" --threshold "$threshold" --statistic "$statistic"
  )
  [[ -n "$unit" ]] && args+=(--unit "$unit")
  "$(_benchmark_python)" "$(_benchmark_tools_root)/records.py" "${args[@]}" >/dev/null
}

benchmark_finish() {
  _benchmark_require_init || return $?
  "$(_benchmark_python)" "$(_benchmark_tools_root)/records.py" finalize --output-dir "$BENCHMARK_DIR"
}
