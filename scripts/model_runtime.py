#!/usr/bin/env python3
"""Model runtime runner based on target-scoped executable runbooks.

The public contract is intentionally small:

    init   Create a runbook template for one model target.
    run    Start a fresh server, run one named check, and always clean up.
    serve  Start a managed persistent server.
    exec   Run one named check against the managed persistent server.
    status Inspect the managed persistent server.
    stop   Stop the managed persistent server safely.

Only Python's standard library is required. Complex engine commands and environment
variables stay in shell files under:

    tasks/<model>/targets/<target>/runbook/
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

EXIT_OK = 0
EXIT_BLOCKED = 2
EXIT_FAILED = 3
EXIT_CLEANUP_FAILED = 4
EXIT_BUSY = 5
EXIT_PARTIAL = 6

TARGET_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
CHECK_RE = re.compile(r"^[A-Za-z0-9._-]+$")
SENSITIVE_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|API[_-]?KEY|PRIVATE[_-]?KEY|CREDENTIAL)", re.I)

TEMPLATE_FILES: dict[str, str] = {
    "README.md": r'''# Runtime Runbook

本目录是当前模型目标的**唯一运行定义**。可以直接粘贴已有的环境变量、启动命令和测试命令。

```text
env.sh              环境变量和公共参数
start.sh            前台启动模型服务
ready.sh            单次就绪探测；运行器负责重试
stop.sh             可选的优雅停止动作
checks/smoke.sh     最小调用测试
checks/validate.sh  正确性验证入口，供 adapt-validate 复用
checks/benchmark.sh 性能测试入口，供 adapt-benchmark 复用
```

## 约束

- `start.sh` 必须保持前台运行，不使用 `nohup`、后台 `&` 或容器内二次 daemonize。
- 所有脚本由 Bash 执行，可以直接粘贴多行命令、管道和 here-document。
- `env.sh` 只设置变量，不安装软件、不修改代码、不启动进程。
- 所有脚本都可以使用 `$RUN_DIR` 保存额外证据。
- `ready.sh` 每次只执行一次探测：成功返回 0，未就绪返回非 0。
- `checks/*.sh`：0 表示通过，64 表示阻塞，65 表示部分完成，其他非 0 表示失败。

## 自动注入变量

运行器会提供：

```text
CONTROL_ROOT MODEL_ID TARGET_ID MODEL_TASK_DIR TARGET_TASK_DIR
RUNBOOK_DIR RUN_DIR MODEL_CONFIG TARGET_CONFIG
MODEL_NAME MODEL_PATH MODEL_REVISION ENGINE HARDWARE
TARGET_REPO UPSTREAM_REPO RUNTIME_PYTHON TENSOR_PARALLEL_SIZE
```

`env.sh` 可以覆盖非路径类默认值，例如 `MODEL_PATH`、`TARGET_REPO`、`MODEL_PORT`。
''',
    "env.sh": r'''#!/usr/bin/env bash
# 只在这里放环境变量和公共参数。支持直接粘贴：
#   export FOO=bar
# 或：
#   FOO=bar

# ===== 在此粘贴环境变量 =====

: "${MODEL_HOST:=127.0.0.1}"
: "${MODEL_PORT:=8000}"
: "${MODEL_BASE_URL:=http://${MODEL_HOST}:${MODEL_PORT}}"
: "${MODEL_STARTUP_TIMEOUT:=600}"
: "${MODEL_CHECK_TIMEOUT:=300}"
: "${MODEL_CHECK_VALIDATE_TIMEOUT:=${MODEL_CHECK_TIMEOUT}}"
: "${MODEL_CHECK_BENCHMARK_TIMEOUT:=3600}"
: "${MODEL_SHUTDOWN_TIMEOUT:=30}"
: "${MODEL_READY_INTERVAL:=2}"
: "${MODEL_READY_PROBE_TIMEOUT:=15}"

export MODEL_HOST MODEL_PORT MODEL_BASE_URL
export MODEL_STARTUP_TIMEOUT MODEL_CHECK_TIMEOUT
export MODEL_CHECK_VALIDATE_TIMEOUT MODEL_CHECK_BENCHMARK_TIMEOUT
export MODEL_SHUTDOWN_TIMEOUT MODEL_READY_INTERVAL MODEL_READY_PROBE_TIMEOUT
''',
    "start.sh": r'''#!/usr/bin/env bash
set -euo pipefail

# 将完整启动命令直接粘贴到这里。
# 命令必须保持前台运行；不要使用 nohup、后台 & 或 daemon 模式。
# 推荐最后一条命令使用 exec，使运行器准确管理服务进程。
#
# 示例：
#   cd "$TARGET_REPO"
#   exec "$RUNTIME_PYTHON" -m ... \
#     --model "$MODEL_PATH" \
#     --host "$MODEL_HOST" \
#     --port "$MODEL_PORT"

echo "MODEL_RUN_NOT_CONFIGURED: 请编辑 $RUNBOOK_DIR/start.sh" >&2
exit 64
''',
    "ready.sh": r'''#!/usr/bin/env bash
set -euo pipefail

# 单次就绪探测。运行器会循环调用本脚本，直到成功或超时。
curl --fail --silent --show-error "${MODEL_BASE_URL}/health" >/dev/null
''',
    "stop.sh": r'''#!/usr/bin/env bash
set -euo pipefail

# 可选：在运行器发送 SIGTERM 之前执行一次优雅停止动作。
# 没有特殊停止命令时保持为空即可。
exit 0
''',
    "checks/smoke.sh": r'''#!/usr/bin/env bash
set -euo pipefail

# 将最小模型调用命令直接粘贴到这里。
# stdout/stderr 会自动保存；额外文件可写入 "$RUN_DIR"。

curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -X POST "${MODEL_BASE_URL}/v1/chat/completions" \
  -d "$(cat <<JSON
{
  \"model\": \"${MODEL_NAME:-model}\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],
  \"temperature\": 0,
  \"max_tokens\": 8
}
JSON
)" | tee "$RUN_DIR/smoke-response.json"
''',
    "checks/validate.sh": r'''#!/usr/bin/env bash
set -euo pipefail

# 可直接粘贴已有验证命令。需要结构化汇总时使用下面的辅助函数。
source "$CONTROL_ROOT/scripts/validation/lib.sh"
validation_init

validation_mark reference-parity required blocked \
  "请在 checks/validate.sh 中配置可信参考 oracle 和验证命令"

# 示例：
# validation_case deterministic-generation required -- \
#   python "$CONTROL_ROOT/scripts/validation/your_test.py" \
#     --endpoint "$MODEL_BASE_URL" \
#     --output-dir "$RUN_DIR/validation/custom"

validation_finish
''',
    "checks/benchmark.sh": r'''#!/usr/bin/env bash
set -euo pipefail

# 可直接粘贴已有 benchmark 命令。命令需把 JSON 指标写入
# $BENCHMARK_SAMPLE_FILE；辅助函数负责 warmup、重复执行和统计。
source "$CONTROL_ROOT/scripts/benchmark/lib.sh"
benchmark_init

benchmark_mark serving-default required blocked \
  "请在 checks/benchmark.sh 中配置 workload 和 benchmark 命令"

# 示例：
# benchmark_case serving-c16 required --warmup 1 --repeat 3 -- \
#   bash -lc 'your-benchmark-command --output "$BENCHMARK_SAMPLE_FILE"'
# benchmark_expect serving-c16 output_throughput higher 900 median token_per_second

benchmark_finish
''',
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def append_lifecycle(run_dir: Path, event: str, **data: Any) -> None:
    payload = {"at": utc_now(), "event": event, **data}
    path = run_dir / "lifecycle.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"控制仓不存在: {root}")
        return root
    env_root = os.environ.get("MODEL_ADAPTATION_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".claude").exists() and (candidate / "tasks").exists():
            return candidate
    return current


def parse_target_ref(value: str) -> tuple[str, str]:
    if not TARGET_RE.fullmatch(value):
        raise ValueError("目标必须使用 <model-id>/<target-id>，且只能包含字母、数字、点、下划线和连字符")
    return tuple(value.split("/", 1))  # type: ignore[return-value]


def target_paths(root: Path, target_ref: str) -> dict[str, Path]:
    model_id, target_id = parse_target_ref(target_ref)
    model_dir = root / "tasks" / model_id
    target_dir = model_dir / "targets" / target_id
    return {
        "model_dir": model_dir,
        "target_dir": target_dir,
        "runbook_dir": target_dir / "runbook",
        "model_config": model_dir / "model.yaml",
        "target_config": target_dir / "target.yaml",
        "run_root": root / "runs" / model_id / target_id,
        "active_state": root / "runs" / model_id / target_id / ".runtime" / "active.json",
    }


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except ValueError:
            pass
    if re.fullmatch(r"-?\d+\.\d+", value):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _minimal_yaml(text: str) -> dict[str, Any]:
    """Parse the small mapping-only YAML shape used by task files.

    PyYAML is used when installed. This fallback intentionally supports only nested
    mappings and scalar values; complex YAML should install PyYAML.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            continue
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    except ImportError:
        return _minimal_yaml(text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法解析 YAML: {path}: {exc}") from exc


def resolve_path(root: Path, value: Any) -> str:
    if value in (None, ""):
        return ""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    return str(path)


def context_defaults(root: Path, target_ref: str, run_dir: Path) -> dict[str, str]:
    model_id, target_id = parse_target_ref(target_ref)
    paths = target_paths(root, target_ref)
    model_cfg = load_yaml(paths["model_config"])
    target_cfg = load_yaml(paths["target_config"])
    runtime_cfg = target_cfg.get("runtime") if isinstance(target_cfg.get("runtime"), dict) else {}
    model_name = str(model_cfg.get("name") or model_id)
    model_path = resolve_path(root, model_cfg.get("model_path"))
    target_repo = resolve_path(root, target_cfg.get("target_repo"))
    upstream_repo = resolve_path(root, target_cfg.get("upstream_repo"))
    return {
        "CONTROL_ROOT": str(root),
        "MODEL_ID": model_id,
        "TARGET_ID": target_id,
        "MODEL_TASK_DIR": str(paths["model_dir"]),
        "TARGET_TASK_DIR": str(paths["target_dir"]),
        "RUNBOOK_DIR": str(paths["runbook_dir"]),
        "RUN_DIR": str(run_dir),
        "MODEL_CONFIG": str(paths["model_config"]),
        "TARGET_CONFIG": str(paths["target_config"]),
        "MODEL_NAME": model_name,
        "MODEL_PATH": model_path,
        "MODEL_REVISION": str(model_cfg.get("revision") or ""),
        "ENGINE": str(target_cfg.get("engine") or target_id),
        "HARDWARE": str(target_cfg.get("hardware") or ""),
        "TARGET_REPO": target_repo,
        "UPSTREAM_REPO": upstream_repo,
        "RUNTIME_PYTHON": str(runtime_cfg.get("python") or sys.executable),
        "TENSOR_PARALLEL_SIZE": str(runtime_cfg.get("tensor_parallel_size") or ""),
    }


def shell_env(
    root: Path,
    target_ref: str,
    run_dir: Path,
    forced: Mapping[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    defaults = context_defaults(root, target_ref, run_dir)
    env.update(defaults)
    for key, value in defaults.items():
        env[f"__MR_{key}"] = value
    for key, value in (forced or {}).items():
        env[f"__MR_FORCE_{key}"] = str(value)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def shell_prefix() -> str:
    reserved = [
        "CONTROL_ROOT",
        "MODEL_ID",
        "TARGET_ID",
        "MODEL_TASK_DIR",
        "TARGET_TASK_DIR",
        "RUNBOOK_DIR",
        "RUN_DIR",
        "MODEL_CONFIG",
        "TARGET_CONFIG",
    ]
    restore = "\n".join(f'export {name}="$__MR_{name}"' for name in reserved)
    forced = "\n".join(
        f'if [[ -n "${{__MR_FORCE_{name}+x}}" ]]; then export {name}="$__MR_FORCE_{name}"; fi'
        for name in ("MODEL_HOST", "MODEL_PORT", "MODEL_BASE_URL")
    )
    return (
        'set -a\n'
        'if [[ -f "$RUNBOOK_DIR/env.sh" ]]; then source "$RUNBOOK_DIR/env.sh"; fi\n'
        'set +a\n'
        f"{restore}\n"
        f"{forced}\n"
    )


def resolve_sourced_env(root: Path, target_ref: str, run_dir: Path) -> dict[str, str]:
    env = shell_env(root, target_ref, run_dir)
    command = shell_prefix() + "env -0"
    completed = subprocess.run(
        ["bash", "-c", command],
        env=env,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "env.sh 加载失败: " + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    result: dict[str, str] = {}
    for item in completed.stdout.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        result[key.decode(errors="replace")] = value.decode(errors="replace")
    host = result.get("MODEL_HOST", "127.0.0.1")
    port = result.get("MODEL_PORT", "8000")
    result.setdefault("MODEL_BASE_URL", f"http://{host}:{port}")
    return result


def safe_public_env(env: Mapping[str, str]) -> dict[str, str]:
    keys = [
        "MODEL_NAME",
        "MODEL_PATH",
        "MODEL_REVISION",
        "ENGINE",
        "HARDWARE",
        "TARGET_REPO",
        "UPSTREAM_REPO",
        "RUNTIME_PYTHON",
        "TENSOR_PARALLEL_SIZE",
        "MODEL_HOST",
        "MODEL_PORT",
        "MODEL_BASE_URL",
        "MODEL_STARTUP_TIMEOUT",
        "MODEL_CHECK_TIMEOUT",
        "MODEL_CHECK_VALIDATE_TIMEOUT",
        "MODEL_CHECK_BENCHMARK_TIMEOUT",
        "MODEL_SHUTDOWN_TIMEOUT",
        "MODEL_READY_INTERVAL",
        "MODEL_READY_PROBE_TIMEOUT",
    ]
    return {
        key: ("<redacted>" if SENSITIVE_RE.search(key) else str(env[key]))
        for key in keys
        if key in env and env[key] != ""
    }


def validate_runbook(root: Path, target_ref: str, check: str | None = None) -> tuple[bool, list[str]]:
    paths = target_paths(root, target_ref)
    missing: list[str] = []
    required = ["env.sh", "start.sh", "ready.sh"]
    if check:
        if not CHECK_RE.fullmatch(check):
            raise ValueError("check 名称只能包含字母、数字、点、下划线和连字符")
        required.append(f"checks/{check}.sh")
    for relative in required:
        if not (paths["runbook_dir"] / relative).is_file():
            missing.append(relative)
    return not missing, missing


def make_run_dir(root: Path, target_ref: str, operation: str, suffix: str, explicit: str | None) -> Path:
    paths = target_paths(root, target_ref)
    if explicit:
        run_dir = Path(explicit).expanduser()
        if not run_dir.is_absolute():
            run_dir = (root / run_dir).resolve()
    else:
        clean_suffix = re.sub(r"[^A-Za-z0-9._-]+", "-", suffix).strip("-")
        name = f"{timestamp()}-{operation}" + (f"-{clean_suffix}" if clean_suffix else "")
        run_dir = paths["run_root"] / name
        counter = 1
        while run_dir.exists():
            run_dir = paths["run_root"] / f"{name}-{counter}"
            counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir()
    return run_dir


def script_hashes(runbook_dir: Path, check: str | None) -> dict[str, str]:
    names = ["env.sh", "start.sh", "ready.sh", "stop.sh"]
    if check:
        names.append(f"checks/{check}.sh")
    result: dict[str, str] = {}
    for name in names:
        path = runbook_dir / name
        if path.is_file():
            result[name] = sha256_file(path)
    return result


def command_for_script(script_path: Path) -> str:
    return shell_prefix() + f'exec bash {shlex.quote(str(script_path))}\n'


def spawn_server(root: Path, target_ref: str, run_dir: Path) -> tuple[subprocess.Popen[bytes], dict[str, Any]]:
    paths = target_paths(root, target_ref)
    env = shell_env(root, target_ref, run_dir)
    command = command_for_script(paths["runbook_dir"] / "start.sh")
    stdout_path = run_dir / "logs" / "server.stdout.log"
    stderr_path = run_dir / "logs" / "server.stderr.log"
    stdout_handle = stdout_path.open("ab", buffering=0)
    stderr_handle = stderr_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            ["bash", "-c", command],
            env=env,
            cwd=str(paths["runbook_dir"]),
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()
    identity = process_identity(process.pid) or {"pid": process.pid}
    process_info = {
        **identity,
        "pid": process.pid,
        "pgid": process.pid,
        "started_at": utc_now(),
        "run_dir": str(run_dir),
    }
    atomic_write_json(run_dir / "process.json", process_info)
    return process, process_info


def process_identity(pid: int) -> dict[str, Any] | None:
    proc = Path("/proc") / str(pid)
    if not proc.exists():
        try:
            os.kill(pid, 0)
            return {"pid": pid}
        except ProcessLookupError:
            return None
        except PermissionError:
            return {"pid": pid}
    try:
        stat_text = (proc / "stat").read_text(encoding="utf-8")
        right = stat_text.rsplit(")", 1)[1].strip().split()
        start_ticks = right[19]  # field 22 overall; right starts at field 3
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        return {"pid": pid, "state": right[0], "start_ticks": start_ticks, "cmdline": cmdline}
    except (OSError, IndexError):
        return {"pid": pid}


def same_process(expected: Mapping[str, Any]) -> bool:
    current = process_identity(int(expected.get("pid", -1)))
    if current is None or current.get("state") == "Z":
        return False
    expected_ticks = expected.get("start_ticks")
    return not expected_ticks or current.get("start_ticks") == expected_ticks


def _float_env(env: Mapping[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, default))
    except (TypeError, ValueError):
        return default


def run_probe_script(
    root: Path,
    target_ref: str,
    run_dir: Path,
    relative_script: str,
    *,
    timeout: float,
    stdout_path: Path,
    stderr_path: Path,
    append: bool = False,
    forced_env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    paths = target_paths(root, target_ref)
    script = paths["runbook_dir"] / relative_script
    env = shell_env(root, target_ref, run_dir, forced=forced_env)
    command = command_for_script(script)
    mode = "ab" if append else "wb"
    with stdout_path.open(mode) as stdout_handle, stderr_path.open(mode) as stderr_handle:
        process = subprocess.Popen(
            ["bash", "-c", command],
            env=env,
            cwd=str(paths["runbook_dir"]),
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
            close_fds=True,
        )
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            raise subprocess.TimeoutExpired(["bash", str(script)], timeout)
        return subprocess.CompletedProcess(["bash", str(script)], return_code)


def wait_ready(
    process: subprocess.Popen[bytes],
    root: Path,
    target_ref: str,
    run_dir: Path,
    *,
    timeout: float,
    interval: float,
    probe_timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    attempts = 0
    log_path = run_dir / "logs" / "ready.log"
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            return {"status": "failed", "reason": "server_exited", "exit_code": code, "attempts": attempts}
        attempts += 1
        with log_path.open("ab") as handle:
            handle.write(f"\n[{utc_now()}] attempt={attempts}\n".encode())
        try:
            completed = run_probe_script(
                root,
                target_ref,
                run_dir,
                "ready.sh",
                timeout=min(max(probe_timeout, 1.0), max(deadline - time.monotonic(), 1.0)),
                stdout_path=log_path,
                stderr_path=log_path,
                append=True,
            )
            if completed.returncode == 0:
                return {"status": "passed", "attempts": attempts, "elapsed_seconds": round(timeout - max(deadline - time.monotonic(), 0), 3)}
        except subprocess.TimeoutExpired:
            with log_path.open("ab") as handle:
                handle.write(b"probe timeout\n")
        time.sleep(max(interval, 0.1))
    return {"status": "failed", "reason": "readiness_timeout", "attempts": attempts, "timeout_seconds": timeout}


def run_check(
    root: Path,
    target_ref: str,
    run_dir: Path,
    check: str,
    timeout: float,
    forced_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    stdout_path = run_dir / "logs" / f"check-{check}.stdout.log"
    stderr_path = run_dir / "logs" / f"check-{check}.stderr.log"
    started = time.monotonic()
    try:
        completed = run_probe_script(
            root,
            target_ref,
            run_dir,
            f"checks/{check}.sh",
            timeout=timeout,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            forced_env=forced_env,
        )
        status = "passed" if completed.returncode == 0 else ("blocked" if completed.returncode == 64 else ("partial" if completed.returncode == 65 else "failed"))
        return {
            "status": status,
            "exit_code": completed.returncode,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "failed",
            "reason": "check_timeout",
            "timeout_seconds": timeout,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }


def run_stop_hook(root: Path, target_ref: str, run_dir: Path, timeout: float) -> dict[str, Any]:
    stop_script = target_paths(root, target_ref)["runbook_dir"] / "stop.sh"
    if not stop_script.is_file():
        return {"status": "skipped", "reason": "stop_script_missing"}
    stdout_path = run_dir / "logs" / "stop.stdout.log"
    stderr_path = run_dir / "logs" / "stop.stderr.log"
    try:
        completed = run_probe_script(
            root,
            target_ref,
            run_dir,
            "stop.sh",
            timeout=timeout,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        return {"status": "passed" if completed.returncode == 0 else "failed", "exit_code": completed.returncode}
    except subprocess.TimeoutExpired:
        return {"status": "failed", "reason": "stop_hook_timeout", "timeout_seconds": timeout}


def terminate_group(process_info: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    if not same_process(process_info):
        return {"status": "passed", "reason": "already_stopped"}
    pgid = int(process_info.get("pgid") or process_info["pid"])
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return {"status": "passed", "reason": "already_stopped"}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not same_process(process_info):
            return {"status": "passed", "reason": "terminated"}
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return {"status": "passed", "reason": "terminated"}
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not same_process(process_info):
            return {"status": "passed", "reason": "killed"}
        time.sleep(0.1)
    return {"status": "failed", "reason": "process_group_still_alive", "pid": process_info.get("pid")}


def terminate_spawned_process(
    process: subprocess.Popen[bytes],
    process_info: Mapping[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Terminate and reap a process started by the current runtime invocation."""
    code = process.poll()
    if code is not None:
        try:
            process.wait(timeout=0)
        except subprocess.TimeoutExpired:
            pass
        return {"status": "passed", "reason": "already_stopped", "exit_code": code}
    pgid = int(process_info.get("pgid") or process.pid)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        try:
            code = process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            code = process.poll()
        return {"status": "passed", "reason": "already_stopped", "exit_code": code}
    try:
        code = process.wait(timeout=max(timeout, 0.1))
        return {"status": "passed", "reason": "terminated", "exit_code": code}
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            code = process.wait(timeout=5)
            return {"status": "passed", "reason": "killed", "exit_code": code}
        except subprocess.TimeoutExpired:
            return {"status": "failed", "reason": "process_group_still_alive", "pid": process.pid}


def cleanup_server(
    root: Path,
    target_ref: str,
    run_dir: Path,
    process_info: Mapping[str, Any],
    timeout: float,
    process: subprocess.Popen[bytes] | None = None,
) -> dict[str, Any]:
    hook = run_stop_hook(root, target_ref, run_dir, min(timeout, 30))
    terminate = (
        terminate_spawned_process(process, process_info, timeout)
        if process is not None
        else terminate_group(process_info, timeout)
    )
    status = "passed" if terminate.get("status") == "passed" else "failed"
    return {"status": status, "stop_hook": hook, "terminate": terminate}


def active_state(root: Path, target_ref: str) -> dict[str, Any] | None:
    path = target_paths(root, target_ref)["active_state"]
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:  # noqa: BLE001
        return {"status": "invalid", "state_path": str(path)}


def clear_active(root: Path, target_ref: str) -> None:
    path = target_paths(root, target_ref)["active_state"]
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def emit(payload: Mapping[str, Any], code: int = EXIT_OK) -> int:
    print(json.dumps(dict(payload), ensure_ascii=False, indent=2))
    return code


def init_runbook(root: Path, target_ref: str, force: bool) -> int:
    paths = target_paths(root, target_ref)
    paths["target_dir"].mkdir(parents=True, exist_ok=True)
    runbook = paths["runbook_dir"]
    backup: str | None = None
    if runbook.exists() and any(runbook.iterdir()):
        if not force:
            return emit(
                {
                    "status": "blocked",
                    "reason": "runbook_exists",
                    "runbook": str(runbook),
                    "message": "Runbook 已存在；不会覆盖。需要重建时使用 --force，原目录会先备份。",
                },
                EXIT_BLOCKED,
            )
        backup_path = runbook.with_name(f"runbook.backup-{timestamp()}")
        shutil.move(str(runbook), str(backup_path))
        backup = str(backup_path)
    runbook.mkdir(parents=True, exist_ok=True)
    for relative, content in TEMPLATE_FILES.items():
        path = runbook / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        if path.suffix == ".sh":
            path.chmod(0o755)
    return emit({"status": "passed", "operation": "init", "runbook": str(runbook), "backup": backup})


def prepare_run(root: Path, target_ref: str, operation: str, check: str | None, explicit_run_dir: str | None) -> tuple[Path, dict[str, str], dict[str, Any]]:
    ok, missing = validate_runbook(root, target_ref, check)
    if not ok:
        raise FileNotFoundError("Runbook 缺少文件: " + ", ".join(missing))
    run_dir = make_run_dir(root, target_ref, operation, check or "", explicit_run_dir)
    env = resolve_sourced_env(root, target_ref, run_dir)
    paths = target_paths(root, target_ref)
    context = {
        "schema_version": 1,
        "created_at": utc_now(),
        "target": target_ref,
        "operation": operation,
        "check": check,
        "run_dir": str(run_dir),
        "runbook": str(paths["runbook_dir"]),
        "runbook_hashes": script_hashes(paths["runbook_dir"], check),
        "input_hashes": {
            name: sha256_file(path)
            for name, path in (
                ("model.yaml", paths["model_config"]),
                ("target.yaml", paths["target_config"]),
            )
            if path.is_file()
        },
        "runtime": safe_public_env(env),
    }
    atomic_write_json(run_dir / "context.json", context)
    return run_dir, env, context


def check_timeout_for(env: Mapping[str, str], check: str, explicit: float | None) -> float:
    if explicit is not None:
        return explicit
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", check).upper()
    key = f"MODEL_CHECK_{normalized}_TIMEOUT"
    return _float_env(env, key, _float_env(env, "MODEL_CHECK_TIMEOUT", 300))


def result_exit_code(status: str) -> int:
    return {
        "passed": EXIT_OK,
        "blocked": EXIT_BLOCKED,
        "partial": EXIT_PARTIAL,
    }.get(status, EXIT_FAILED)

def cmd_run(args: argparse.Namespace, root: Path) -> int:
    check = args.check
    try:
        existing = active_state(root, args.target)
        if existing and same_process(existing.get("process", {})):
            return emit(
                {
                    "status": "blocked",
                    "reason": "managed_service_active",
                    "run_dir": existing.get("run_dir"),
                    "message": "当前目标已有托管服务；先执行 --stop，或使用 exec 子命令在该服务上运行检查。",
                },
                EXIT_BUSY,
            )
        if existing:
            clear_active(root, args.target)
        run_dir, env, context = prepare_run(root, args.target, "run", check, args.run_dir)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        return emit({"status": "blocked", "phase": "prepare", "error": str(exc)}, EXIT_BLOCKED)

    startup_timeout = args.startup_timeout or _float_env(env, "MODEL_STARTUP_TIMEOUT", 600)
    check_timeout = check_timeout_for(env, check, args.check_timeout)
    shutdown_timeout = args.shutdown_timeout or _float_env(env, "MODEL_SHUTDOWN_TIMEOUT", 30)
    ready_interval = _float_env(env, "MODEL_READY_INTERVAL", 2)
    ready_probe_timeout = _float_env(env, "MODEL_READY_PROBE_TIMEOUT", 15)
    result: dict[str, Any] = {**context, "status": "failed", "phase": "launch"}
    process: subprocess.Popen[bytes] | None = None
    process_info: dict[str, Any] | None = None
    exit_code = EXIT_FAILED
    try:
        append_lifecycle(run_dir, "server_spawn_begin")
        process, process_info = spawn_server(root, args.target, run_dir)
        append_lifecycle(run_dir, "server_spawned", pid=process.pid)
        readiness = wait_ready(
            process,
            root,
            args.target,
            run_dir,
            timeout=startup_timeout,
            interval=ready_interval,
            probe_timeout=ready_probe_timeout,
        )
        result["readiness"] = readiness
        append_lifecycle(run_dir, "readiness_finished", status=readiness.get("status"))
        if readiness.get("status") != "passed":
            result.update({"status": "failed", "phase": "readiness"})
            exit_code = EXIT_FAILED
        else:
            append_lifecycle(run_dir, "check_begin", check=check, timeout=check_timeout)
            check_result = run_check(root, args.target, run_dir, check, check_timeout)
            append_lifecycle(run_dir, "check_finished", status=check_result.get("status"), exit_code=check_result.get("exit_code"))
            result["check_result"] = check_result
            result["phase"] = "check"
            result["status"] = check_result["status"]
            exit_code = result_exit_code(check_result["status"])
    except Exception as exc:  # noqa: BLE001
        result.update({"status": "failed", "phase": "exception", "error": f"{type(exc).__name__}: {exc}"})
        exit_code = EXIT_FAILED
    finally:
        if process_info is not None:
            append_lifecycle(run_dir, "cleanup_begin", timeout=shutdown_timeout)
            cleanup = cleanup_server(root, args.target, run_dir, process_info, shutdown_timeout, process)
            append_lifecycle(run_dir, "cleanup_finished", status=cleanup.get("status"))
            result["cleanup"] = cleanup
            if cleanup.get("status") != "passed":
                result["status"] = "failed"
                result["phase"] = "cleanup"
                exit_code = EXIT_CLEANUP_FAILED
        result["finished_at"] = utc_now()
        atomic_write_json(run_dir / "result.json", result)
        append_lifecycle(run_dir, "result_written", status=result.get("status"))
    return emit(result, exit_code)


def cmd_serve(args: argparse.Namespace, root: Path) -> int:
    existing = active_state(root, args.target)
    if existing and same_process(existing.get("process", {})):
        return emit({"status": "blocked", "reason": "already_running", **existing}, EXIT_BUSY)
    if existing:
        clear_active(root, args.target)
    try:
        run_dir, env, context = prepare_run(root, args.target, "serve", None, args.run_dir)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        return emit({"status": "blocked", "phase": "prepare", "error": str(exc)}, EXIT_BLOCKED)
    startup_timeout = args.startup_timeout or _float_env(env, "MODEL_STARTUP_TIMEOUT", 600)
    ready_interval = _float_env(env, "MODEL_READY_INTERVAL", 2)
    ready_probe_timeout = _float_env(env, "MODEL_READY_PROBE_TIMEOUT", 15)
    shutdown_timeout = args.shutdown_timeout or _float_env(env, "MODEL_SHUTDOWN_TIMEOUT", 30)
    process_info: dict[str, Any] | None = None
    try:
        process, process_info = spawn_server(root, args.target, run_dir)
        readiness = wait_ready(
            process, root, args.target, run_dir, timeout=startup_timeout,
            interval=ready_interval, probe_timeout=ready_probe_timeout
        )
        if readiness.get("status") != "passed":
            cleanup = cleanup_server(root, args.target, run_dir, process_info, shutdown_timeout, process)
            result = {**context, "status": "failed", "phase": "readiness", "readiness": readiness, "cleanup": cleanup, "finished_at": utc_now()}
            atomic_write_json(run_dir / "result.json", result)
            return emit(result, EXIT_FAILED)
        state = {
            "schema_version": 1,
            "target": args.target,
            "run_dir": str(run_dir),
            "started_at": utc_now(),
            "process": process_info,
            "runtime": safe_public_env(env),
            "runbook_hashes": context["runbook_hashes"],
        }
        atomic_write_json(target_paths(root, args.target)["active_state"], state)
        result = {**context, "status": "passed", "phase": "serving", "readiness": readiness, "active_state": str(target_paths(root, args.target)["active_state"])}
        atomic_write_json(run_dir / "result.json", result)
        return emit(result)
    except Exception as exc:  # noqa: BLE001
        cleanup = None
        if process_info is not None:
            cleanup = cleanup_server(root, args.target, run_dir, process_info, shutdown_timeout, process)
        clear_active(root, args.target)
        result = {**context, "status": "failed", "phase": "exception", "error": f"{type(exc).__name__}: {exc}", "cleanup": cleanup, "finished_at": utc_now()}
        atomic_write_json(run_dir / "result.json", result)
        return emit(result, EXIT_FAILED)


def cmd_exec(args: argparse.Namespace, root: Path) -> int:
    state = active_state(root, args.target)
    if not state or not same_process(state.get("process", {})):
        if state:
            clear_active(root, args.target)
        return emit({"status": "blocked", "reason": "no_active_service"}, EXIT_BLOCKED)
    try:
        ok, missing = validate_runbook(root, args.target, args.check)
        if not ok:
            raise FileNotFoundError("Runbook 缺少文件: " + ", ".join(missing))
        run_dir = make_run_dir(root, args.target, "exec", args.check, args.run_dir)
        env = resolve_sourced_env(root, args.target, run_dir)
        # Preserve the endpoint of the managed service even if env.sh changed later.
        active_runtime = state.get("runtime") if isinstance(state.get("runtime"), dict) else {}
        check_timeout = check_timeout_for(env, args.check, args.check_timeout)
        context = {
            "schema_version": 1,
            "created_at": utc_now(),
            "target": args.target,
            "operation": "exec",
            "check": args.check,
            "run_dir": str(run_dir),
            "active_run_dir": state.get("run_dir"),
            "runtime": active_runtime,
            "runbook_hashes": script_hashes(target_paths(root, args.target)["runbook_dir"], args.check),
            "input_hashes": {
                name: sha256_file(path)
                for name, path in (
                    ("model.yaml", target_paths(root, args.target)["model_config"]),
                    ("target.yaml", target_paths(root, args.target)["target_config"]),
                )
                if path.is_file()
            },
        }
        atomic_write_json(run_dir / "context.json", context)
        forced = {
            key: str(active_runtime[key])
            for key in ("MODEL_HOST", "MODEL_PORT", "MODEL_BASE_URL")
            if active_runtime.get(key)
        }
        check_result = run_check(
            root, args.target, run_dir, args.check, check_timeout, forced_env=forced
        )
        result = {**context, "status": check_result["status"], "check_result": check_result, "finished_at": utc_now()}
        atomic_write_json(run_dir / "result.json", result)
        code = result_exit_code(result["status"])
        return emit(result, code)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        return emit({"status": "blocked", "phase": "prepare", "error": str(exc)}, EXIT_BLOCKED)


def cmd_status(args: argparse.Namespace, root: Path) -> int:
    state = active_state(root, args.target)
    if not state:
        return emit({"status": "stopped", "target": args.target})
    process = state.get("process") if isinstance(state.get("process"), dict) else {}
    alive = same_process(process)
    payload = {**state, "status": "running" if alive else "stale", "alive": alive}
    if not alive:
        clear_active(root, args.target)
    return emit(payload)


def cmd_stop(args: argparse.Namespace, root: Path) -> int:
    state = active_state(root, args.target)
    if not state:
        return emit({"status": "passed", "reason": "already_stopped", "target": args.target})
    process_info = state.get("process") if isinstance(state.get("process"), dict) else {}
    if not same_process(process_info):
        clear_active(root, args.target)
        return emit({"status": "passed", "reason": "stale_state_cleared", "target": args.target})
    run_dir = Path(str(state.get("run_dir"))).resolve()
    runtime = state.get("runtime") if isinstance(state.get("runtime"), dict) else {}
    timeout = args.shutdown_timeout or _float_env(runtime, "MODEL_SHUTDOWN_TIMEOUT", 30)
    cleanup = cleanup_server(root, args.target, run_dir, process_info, timeout)
    if cleanup.get("status") == "passed":
        clear_active(root, args.target)
    stop_result = {"status": cleanup.get("status"), "target": args.target, "run_dir": str(run_dir), "cleanup": cleanup, "finished_at": utc_now()}
    atomic_write_json(run_dir / "stop-result.json", stop_result)
    return emit(stop_result, EXIT_OK if cleanup.get("status") == "passed" else EXIT_CLEANUP_FAILED)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于可执行 Runbook 的模型服务运行器")
    parser.add_argument("--repo-root", help="控制仓根目录；默认自动发现")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="创建目标 Runbook 模板")
    init.add_argument("target")
    init.add_argument("--force", action="store_true", help="先备份再重建现有 Runbook")

    run = sub.add_parser("run", help="启动新服务、执行命名检查并强制清理")
    run.add_argument("target")
    run.add_argument("--check", default="smoke")
    run.add_argument("--run-dir")
    run.add_argument("--startup-timeout", type=float)
    run.add_argument("--check-timeout", type=float)
    run.add_argument("--shutdown-timeout", type=float)

    serve = sub.add_parser("serve", help="启动托管的持久服务")
    serve.add_argument("target")
    serve.add_argument("--run-dir")
    serve.add_argument("--startup-timeout", type=float)
    serve.add_argument("--shutdown-timeout", type=float)

    execute = sub.add_parser("exec", help="在当前托管服务上执行命名检查")
    execute.add_argument("target")
    execute.add_argument("--check", default="smoke")
    execute.add_argument("--run-dir")
    execute.add_argument("--check-timeout", type=float)

    status = sub.add_parser("status", help="查看托管服务状态")
    status.add_argument("target")

    stop = sub.add_parser("stop", help="停止托管服务")
    stop.add_argument("target")
    stop.add_argument("--shutdown-timeout", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        root = discover_root(args.repo_root)
        parse_target_ref(args.target)
        if args.command == "init":
            return init_runbook(root, args.target, args.force)
        if args.command == "run":
            return cmd_run(args, root)
        if args.command == "serve":
            return cmd_serve(args, root)
        if args.command == "exec":
            return cmd_exec(args, root)
        if args.command == "status":
            return cmd_status(args, root)
        if args.command == "stop":
            return cmd_stop(args, root)
        parser.error("unknown command")
    except ValueError as exc:
        return emit({"status": "blocked", "error": str(exc)}, EXIT_BLOCKED)
    return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
