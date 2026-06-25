#!/usr/bin/env python3
"""Target-scoped Runbook model lifecycle manager.

The runner intentionally knows nothing about vLLM or SGLang flags. Engine-specific
commands live in templates/runbook/common plus templates/runbook/engines/<engine>.sh.
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
EXIT_CLEANUP = 4
EXIT_BUSY = 5
TARGET_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
CHECK_RE = re.compile(r"^[A-Za-z0-9._-]+$")
SENSITIVE_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|API[_-]?KEY|PRIVATE[_-]?KEY|CREDENTIAL)", re.I)
RESERVED = (
    "CONTROL_ROOT", "MODEL_ID", "TARGET_ID", "MODEL_TASK_DIR", "TARGET_TASK_DIR",
    "RUNBOOK_DIR", "RUN_DIR", "MODEL_CONFIG", "TARGET_CONFIG",
)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(dict(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"control repository does not exist: {root}")
        return root
    if os.environ.get("MODEL_ADAPTATION_ROOT"):
        return Path(os.environ["MODEL_ADAPTATION_ROOT"]).expanduser().resolve()
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".claude").is_dir() and (candidate / "tasks").is_dir():
            return candidate
    return current


def parse_target(value: str) -> tuple[str, str]:
    if not TARGET_RE.fullmatch(value):
        raise ValueError("target must be <model-id>/<target-id>")
    return tuple(value.split("/", 1))  # type: ignore[return-value]


def paths_for(root: Path, target: str) -> dict[str, Path]:
    model, target_id = parse_target(target)
    model_dir = root / "tasks" / model
    target_dir = model_dir / "targets" / target_id
    return {
        "model_dir": model_dir,
        "target_dir": target_dir,
        "model_yaml": model_dir / "model.yaml",
        "target_yaml": target_dir / "target.yaml",
        "runbook": target_dir / "runbook",
        "run_root": root / "runs" / model / target_id,
        "active": root / "runs" / model / target_id / ".runtime" / "active.json",
    }


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "Null", "NULL", "~"}:
        return None if value else ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def minimal_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, result)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not value.strip():
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = parse_scalar(value)
    return result


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text) or {}
        return data if isinstance(data, dict) else {}
    except ImportError:
        return minimal_yaml(text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid YAML {path}: {exc}") from exc


def resolve_path(root: Path, value: Any) -> str:
    if value in (None, ""):
        return ""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    return str(path)


def default_env(root: Path, target: str, run_dir: Path) -> dict[str, str]:
    model, target_id = parse_target(target)
    paths = paths_for(root, target)
    model_cfg = load_yaml(paths["model_yaml"])
    target_cfg = load_yaml(paths["target_yaml"])
    runtime = target_cfg.get("runtime") if isinstance(target_cfg.get("runtime"), dict) else {}
    defaults = {
        "CONTROL_ROOT": str(root),
        "MODEL_ID": model,
        "TARGET_ID": target_id,
        "MODEL_TASK_DIR": str(paths["model_dir"]),
        "TARGET_TASK_DIR": str(paths["target_dir"]),
        "RUNBOOK_DIR": str(paths["runbook"]),
        "RUN_DIR": str(run_dir),
        "MODEL_CONFIG": str(paths["model_yaml"]),
        "TARGET_CONFIG": str(paths["target_yaml"]),
        "MODEL_NAME": str(model_cfg.get("name") or model),
        "MODEL_PATH": resolve_path(root, model_cfg.get("path") or model_cfg.get("model_path")),
        "MODEL_REVISION": str(model_cfg.get("revision") or ""),
        "ENGINE": str(target_cfg.get("engine") or target_id),
        "HARDWARE": str(target_cfg.get("hardware") or ""),
        "TARGET_REPO": resolve_path(root, target_cfg.get("target_repo")),
        "UPSTREAM_REPO": resolve_path(root, target_cfg.get("upstream_repo")),
        "RUNTIME_PYTHON": str(runtime.get("python") or sys.executable),
        "TENSOR_PARALLEL_SIZE": str(runtime.get("tensor_parallel_size") or ""),
    }
    env = os.environ.copy()
    env.update(defaults)
    for key, value in defaults.items():
        env[f"__MR_{key}"] = value
    env["PYTHONUNBUFFERED"] = "1"
    return env


def shell_prefix() -> str:
    restore = "\n".join(f'export {key}="$__MR_{key}"' for key in RESERVED)
    return (
        'set -a\n'
        'source "$RUNBOOK_DIR/env.sh"\n'
        'if [[ -f "$RUNBOOK_DIR/env.local.sh" ]]; then source "$RUNBOOK_DIR/env.local.sh"; fi\n'
        'set +a\n'
        f"{restore}\n"
    )


def resolve_env(root: Path, target: str, run_dir: Path) -> dict[str, str]:
    env = default_env(root, target, run_dir)
    proc = subprocess.run(
        ["bash", "-c", shell_prefix() + "env -0"], cwd=str(root), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("env.sh failed: " + proc.stderr.decode(errors="replace").strip())
    result: dict[str, str] = {}
    for item in proc.stdout.split(b"\0"):
        if item and b"=" in item:
            key, value = item.split(b"=", 1)
            result[key.decode(errors="replace")] = value.decode(errors="replace")
    host = result.get("MODEL_HOST", "127.0.0.1")
    port = result.get("MODEL_PORT", "8000")
    result.setdefault("MODEL_BASE_URL", f"http://{host}:{port}")
    return result


def public_env(env: Mapping[str, str]) -> dict[str, str]:
    keys = (
        "MODEL_NAME", "MODEL_PATH", "MODEL_REVISION", "ENGINE", "HARDWARE",
        "TARGET_REPO", "UPSTREAM_REPO", "RUNTIME_PYTHON", "TENSOR_PARALLEL_SIZE",
        "MODEL_HOST", "MODEL_PORT", "MODEL_BASE_URL", "MODEL_STARTUP_TIMEOUT",
        "MODEL_CHECK_TIMEOUT", "MODEL_VALIDATE_TIMEOUT", "MODEL_BENCHMARK_TIMEOUT",
        "MODEL_SHUTDOWN_TIMEOUT", "MODEL_READY_INTERVAL", "MODEL_READY_PROBE_TIMEOUT",
    )
    sensitive = re.compile(r"(TOKEN|SECRET|PASSWORD|KEY|CREDENTIAL)", re.I)
    return {key: ("<redacted>" if sensitive.search(key) else env[key]) for key in keys if env.get(key)}


def runbook_ready(root: Path, target: str, check: str | None = None) -> tuple[bool, list[str]]:
    if check and not CHECK_RE.fullmatch(check):
        raise ValueError("invalid check name")
    runbook = paths_for(root, target)["runbook"]
    required = ["env.sh", "start.sh", "ready.sh"]
    if check:
        required.append(f"checks/{check}.sh")
    missing = [name for name in required if not (runbook / name).is_file()]
    return not missing, missing


def runbook_configured(root: Path, target: str, check: str | None = None) -> tuple[bool, list[str]]:
    ok, missing = runbook_ready(root, target, check)
    if not ok:
        return False, missing
    runbook = paths_for(root, target)["runbook"]
    candidates = ["start.sh"] + ([f"checks/{check}.sh"] if check else [])
    placeholders = [name for name in candidates if "MODEL_RUN_NOT_CONFIGURED" in (runbook / name).read_text(encoding="utf-8", errors="replace")]
    return not placeholders, placeholders


def copy_runbook_template(root: Path, target: str, force: bool) -> dict[str, Any]:
    paths = paths_for(root, target)
    paths["target_dir"].mkdir(parents=True, exist_ok=True)
    target_cfg = load_yaml(paths["target_yaml"])
    engine = str(target_cfg.get("engine") or parse_target(target)[1])
    common = root / "templates" / "runbook" / "common"
    start = root / "templates" / "runbook" / "engines" / f"{engine}.sh"
    if not start.is_file():
        start = root / "templates" / "runbook" / "engines" / "generic.sh"
    if not common.is_dir() or not start.is_file():
        raise FileNotFoundError("runbook templates are incomplete")

    destination = paths["runbook"]
    backup = None
    if force and destination.exists() and any(destination.iterdir()):
        backup_path = destination.with_name(f"runbook.backup-{timestamp()}")
        shutil.move(str(destination), str(backup_path))
        backup = str(backup_path)

    destination.mkdir(parents=True, exist_ok=True)
    actions: list[dict[str, str]] = []
    for source in sorted(path for path in common.rglob("*") if path.is_file()):
        relative = source.relative_to(common)
        target_path = destination / relative
        if target_path.exists():
            actions.append({"path": str(relative), "action": "preserved"})
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_path)
        actions.append({"path": str(relative), "action": "created"})
    start_target = destination / "start.sh"
    if start_target.exists():
        actions.append({"path": "start.sh", "action": "preserved"})
    else:
        shutil.copy2(start, start_target)
        actions.append({"path": "start.sh", "action": "created"})
    for script in destination.rglob("*.sh"):
        script.chmod(0o755)
    return {
        "status": "passed",
        "operation": "init",
        "engine": engine,
        "runbook": str(destination),
        "backup": backup,
        "actions": actions,
    }


def create_run_dir(root: Path, target: str, operation: str, suffix: str, explicit: str | None) -> Path:
    run_root = paths_for(root, target)["run_root"]
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = (root / path).resolve()
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"run directory is not empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
    else:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", suffix).strip("-")
        name = f"{timestamp()}-{operation}" + (f"-{safe}" if safe else "")
        base = run_root / name
        path = base
        index = 1
        while path.exists():
            path = Path(f"{base}-{index}")
            index += 1
        path.mkdir(parents=True)
    (path / "logs").mkdir(exist_ok=True)
    return path


def script_hashes(runbook: Path, check: str | None) -> dict[str, str]:
    names = ["env.sh", "start.sh", "ready.sh", "stop.sh"]
    if check:
        names.append(f"checks/{check}.sh")
    return {name: sha256(runbook / name) for name in names if (runbook / name).is_file()}


def script_command(path: Path) -> str:
    return shell_prefix() + f"exec bash {shlex.quote(str(path))}\n"


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
        fields = (proc / "stat").read_text(encoding="utf-8").split()
        if len(fields) > 2 and fields[2] == "Z":
            return None
        cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        return {"pid": pid, "state": fields[2], "start_ticks": fields[21], "cmdline": cmdline}
    except (OSError, IndexError):
        return None


def same_process(expected: Mapping[str, Any]) -> bool:
    current = process_identity(int(expected.get("pid", -1)))
    return bool(current and (not expected.get("start_ticks") or current.get("start_ticks") == expected.get("start_ticks")))


def spawn(root: Path, target: str, run_dir: Path) -> tuple[subprocess.Popen[bytes], dict[str, Any]]:
    runbook = paths_for(root, target)["runbook"]
    env = default_env(root, target, run_dir)
    stdout = (run_dir / "logs/server.stdout.log").open("ab", buffering=0)
    stderr = (run_dir / "logs/server.stderr.log").open("ab", buffering=0)
    try:
        proc = subprocess.Popen(
            ["bash", "-c", script_command(runbook / "start.sh")], cwd=str(runbook), env=env,
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            start_new_session=True, close_fds=True,
        )
    finally:
        stdout.close(); stderr.close()
    time.sleep(0.5)
    identity = process_identity(proc.pid) or {"pid": proc.pid}
    info = {**identity, "pid": proc.pid, "pgid": proc.pid, "started_at": now_iso(), "run_dir": str(run_dir)}
    atomic_json(run_dir / "process.json", info)
    return proc, info


def run_script(root: Path, target: str, run_dir: Path, relative: str, timeout: float, stdout: Path, stderr: Path, append: bool = False) -> subprocess.CompletedProcess[bytes]:
    runbook = paths_for(root, target)["runbook"]
    env = default_env(root, target, run_dir)
    mode = "ab" if append else "wb"
    with stdout.open(mode) as out, stderr.open(mode) as err:
        proc = subprocess.Popen(
            ["bash", "-c", script_command(runbook / relative)], cwd=str(runbook), env=env,
            stdout=out, stderr=err, start_new_session=True, close_fds=True,
        )
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait(timeout=5)
            raise
    return subprocess.CompletedProcess(["bash", relative], rc)


def as_float(env: Mapping[str, str], key: str, default: float) -> float:
    try: return float(env.get(key, default))
    except (TypeError, ValueError): return default


def check_timeout(env: Mapping[str, str], check: str, explicit: float | None) -> float:
    if explicit is not None:
        return explicit
    key = "MODEL_" + re.sub(r"[^A-Za-z0-9]+", "_", check).strip("_").upper() + "_TIMEOUT"
    return as_float(env, key, as_float(env, "MODEL_CHECK_TIMEOUT", 300))


def wait_ready(proc: subprocess.Popen[bytes], root: Path, target: str, run_dir: Path, timeout: float, interval: float, probe_timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    attempts = 0
    log = run_dir / "logs/ready.log"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return {"status": "failed", "reason": "server_exited", "exit_code": proc.returncode, "attempts": attempts}
        attempts += 1
        with log.open("ab") as handle:
            handle.write(f"\n[{now_iso()}] attempt={attempts}\n".encode())
        try:
            result = run_script(root, target, run_dir, "ready.sh", min(probe_timeout, max(deadline-time.monotonic(), 1)), log, log, True)
            if result.returncode == 0:
                return {"status": "passed", "attempts": attempts}
        except subprocess.TimeoutExpired:
            with log.open("ab") as handle: handle.write(b"probe timeout\n")
        time.sleep(max(interval, 0.1))
    return {"status": "failed", "reason": "readiness_timeout", "attempts": attempts, "timeout_seconds": timeout}


def execute_check(root: Path, target: str, run_dir: Path, check: str, timeout: float) -> dict[str, Any]:
    out = run_dir / f"logs/check-{check}.stdout.log"
    err = run_dir / f"logs/check-{check}.stderr.log"
    started = time.monotonic()
    try:
        result = run_script(root, target, run_dir, f"checks/{check}.sh", timeout, out, err)
        status = "passed" if result.returncode == 0 else ("blocked" if result.returncode == 64 else "failed")
        return {"status": status, "exit_code": result.returncode, "elapsed_seconds": round(time.monotonic()-started,3), "stdout": str(out), "stderr": str(err)}
    except subprocess.TimeoutExpired:
        return {"status": "failed", "reason": "check_timeout", "timeout_seconds": timeout, "stdout": str(out), "stderr": str(err)}


def stop_group(info: Mapping[str, Any], timeout: float, proc: subprocess.Popen[bytes] | None = None) -> dict[str, Any]:
    if proc is not None and proc.poll() is not None:
        return {"status": "passed", "reason": "already_stopped", "exit_code": proc.returncode}
    if not same_process(info):
        return {"status": "passed", "reason": "already_stopped"}
    pgid = int(info.get("pgid") or info["pid"])
    try: os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError: return {"status": "passed", "reason": "already_stopped"}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return {"status": "passed", "reason": "terminated", "exit_code": proc.returncode}
        if not same_process(info): return {"status": "passed", "reason": "terminated"}
        time.sleep(0.2)
    try: os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError: return {"status": "passed", "reason": "terminated"}
    except PermissionError:
        if proc is not None and proc.poll() is not None:
            return {"status": "passed", "reason": "terminated", "exit_code": proc.returncode}
        return {"status": "failed", "reason": "permission_denied", "pid": info.get("pid")}
    for _ in range(50):
        if proc is not None and proc.poll() is not None:
            return {"status": "passed", "reason": "killed", "exit_code": proc.returncode}
        if not same_process(info): return {"status": "passed", "reason": "killed"}
        time.sleep(0.1)
    return {"status": "failed", "reason": "process_still_alive", "pid": info.get("pid")}


def cleanup(root: Path, target: str, run_dir: Path, info: Mapping[str, Any], timeout: float, proc: subprocess.Popen[bytes] | None = None) -> dict[str, Any]:
    hook = {"status": "skipped"}
    if (paths_for(root, target)["runbook"] / "stop.sh").is_file():
        try:
            result = run_script(root, target, run_dir, "stop.sh", min(timeout, 30), run_dir/"logs/stop.stdout.log", run_dir/"logs/stop.stderr.log")
            hook = {"status": "passed" if result.returncode == 0 else "failed", "exit_code": result.returncode}
        except subprocess.TimeoutExpired:
            hook = {"status": "failed", "reason": "stop_hook_timeout"}
    terminate = stop_group(info, timeout, proc)
    return {"status": "passed" if terminate["status"] == "passed" else "failed", "stop_hook": hook, "terminate": terminate}


def active(root: Path, target: str) -> dict[str, Any] | None:
    path = paths_for(root, target)["active"]
    if not path.is_file(): return None
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return {"status": "invalid"}


def clear_active(root: Path, target: str) -> None:
    try: paths_for(root, target)["active"].unlink()
    except FileNotFoundError: pass


def emit(data: Mapping[str, Any], code: int = EXIT_OK) -> int:
    print(json.dumps(dict(data), ensure_ascii=False, indent=2))
    return code


def prepare(root: Path, target: str, operation: str, check: str | None, explicit: str | None) -> tuple[Path, dict[str,str], dict[str,Any]]:
    ok, missing = runbook_configured(root, target, check)
    if not ok: raise FileNotFoundError("Runbook missing/unconfigured: " + ", ".join(missing))
    run_dir = create_run_dir(root, target, operation, check or "", explicit)
    env = resolve_env(root, target, run_dir)
    runbook = paths_for(root, target)["runbook"]
    context = {"schema_version":1,"created_at":now_iso(),"target":target,"operation":operation,"check":check,"run_dir":str(run_dir),"runbook":str(runbook),"runbook_hashes":script_hashes(runbook,check),"runtime":public_env(env)}
    atomic_json(run_dir/"context.json",context)
    return run_dir,env,context


def run_ephemeral(args: argparse.Namespace, root: Path) -> int:
    state=active(root,args.target)
    if state and same_process(state.get("process",{})):
        return emit({"status":"blocked","reason":"managed_service_active","run_dir":state.get("run_dir")},EXIT_BUSY)
    if state: clear_active(root,args.target)
    try: run_dir,env,context=prepare(root,args.target,"run",args.check,args.run_dir)
    except Exception as exc: return emit({"status":"blocked","phase":"prepare","error":str(exc)},EXIT_BLOCKED)
    startup=as_float(env,"MODEL_STARTUP_TIMEOUT",600); interval=as_float(env,"MODEL_READY_INTERVAL",2); probe=as_float(env,"MODEL_READY_PROBE_TIMEOUT",15); shutdown=as_float(env,"MODEL_SHUTDOWN_TIMEOUT",30); ctimeout=check_timeout(env,args.check,args.check_timeout)
    result={**context,"status":"failed","phase":"launch"}; proc=None; info=None; code=EXIT_FAILED
    try:
        proc,info=spawn(root,args.target,run_dir)
        readiness=wait_ready(proc,root,args.target,run_dir,startup,interval,probe); result["readiness"]=readiness
        if readiness["status"]!="passed": result["phase"]="readiness"
        else:
            checked=execute_check(root,args.target,run_dir,args.check,ctimeout); result["check_result"]=checked; result["phase"]="check"; result["status"]=checked["status"]; code=EXIT_OK if checked["status"]=="passed" else (EXIT_BLOCKED if checked["status"]=="blocked" else EXIT_FAILED)
    except Exception as exc: result.update({"status":"failed","phase":"exception","error":f"{type(exc).__name__}: {exc}"})
    finally:
        if info:
            result["cleanup"]=cleanup(root,args.target,run_dir,info,shutdown,proc)
            if result["cleanup"]["status"]!="passed": result["status"]="failed"; result["phase"]="cleanup"; code=EXIT_CLEANUP
        result["finished_at"]=now_iso(); atomic_json(run_dir/"result.json",result)
    return emit(result,code)


def serve(args: argparse.Namespace, root: Path) -> int:
    state=active(root,args.target)
    if state and same_process(state.get("process",{})): return emit({"status":"blocked","reason":"already_running",**state},EXIT_BUSY)
    if state: clear_active(root,args.target)
    try: run_dir,env,context=prepare(root,args.target,"serve",None,args.run_dir)
    except Exception as exc: return emit({"status":"blocked","phase":"prepare","error":str(exc)},EXIT_BLOCKED)
    startup=as_float(env,"MODEL_STARTUP_TIMEOUT",600); interval=as_float(env,"MODEL_READY_INTERVAL",2); probe=as_float(env,"MODEL_READY_PROBE_TIMEOUT",15); shutdown=as_float(env,"MODEL_SHUTDOWN_TIMEOUT",30); proc=None; info=None
    try:
        proc,info=spawn(root,args.target,run_dir); ready=wait_ready(proc,root,args.target,run_dir,startup,interval,probe)
        if ready["status"]!="passed":
            result={**context,"status":"failed","phase":"readiness","readiness":ready,"cleanup":cleanup(root,args.target,run_dir,info,shutdown,proc)}; atomic_json(run_dir/"result.json",result); return emit(result,EXIT_FAILED)
        state={"schema_version":1,"target":args.target,"run_dir":str(run_dir),"process":info,"runtime":public_env(env),"runbook_hashes":context["runbook_hashes"],"started_at":now_iso()}; atomic_json(paths_for(root,args.target)["active"],state)
        result={**context,"status":"passed","phase":"serving","readiness":ready,"endpoint":env.get("MODEL_BASE_URL"),"active_state":str(paths_for(root,args.target)["active"])}; atomic_json(run_dir/"result.json",result); return emit(result)
    except Exception as exc:
        clean=cleanup(root,args.target,run_dir,info,shutdown,proc) if info else None; clear_active(root,args.target); result={**context,"status":"failed","phase":"exception","error":f"{type(exc).__name__}: {exc}","cleanup":clean}; atomic_json(run_dir/"result.json",result); return emit(result,EXIT_FAILED)


def exec_check(args: argparse.Namespace, root: Path) -> int:
    state=active(root,args.target)
    if not state or not same_process(state.get("process",{})):
        if state: clear_active(root,args.target)
        return emit({"status":"blocked","reason":"no_active_service"},EXIT_BLOCKED)
    try:
        ok,missing=runbook_configured(root,args.target,args.check)
        if not ok: raise FileNotFoundError(", ".join(missing))
        run_dir=create_run_dir(root,args.target,"exec",args.check,args.run_dir); env=resolve_env(root,args.target,run_dir); timeout=check_timeout(env,args.check,args.check_timeout); checked=execute_check(root,args.target,run_dir,args.check,timeout); result={"schema_version":1,"target":args.target,"operation":"exec","check":args.check,"status":checked["status"],"run_dir":str(run_dir),"active_run_dir":state.get("run_dir"),"check_result":checked,"finished_at":now_iso()}; atomic_json(run_dir/"result.json",result); code=EXIT_OK if checked["status"]=="passed" else (EXIT_BLOCKED if checked["status"]=="blocked" else EXIT_FAILED); return emit(result,code)
    except Exception as exc: return emit({"status":"blocked","phase":"prepare","error":str(exc)},EXIT_BLOCKED)


def status_cmd(args: argparse.Namespace, root: Path) -> int:
    state=active(root,args.target)
    if not state: return emit({"status":"stopped","target":args.target})
    alive=same_process(state.get("process",{})); result={**state,"status":"running" if alive else "stale","alive":alive}
    if not alive: clear_active(root,args.target)
    return emit(result)


def stop_cmd(args: argparse.Namespace, root: Path) -> int:
    state=active(root,args.target)
    if not state: return emit({"status":"passed","reason":"already_stopped","target":args.target})
    info=state.get("process",{})
    if not same_process(info): clear_active(root,args.target); return emit({"status":"passed","reason":"stale_state_cleared"})
    run_dir=Path(str(state["run_dir"])); timeout=args.shutdown_timeout or as_float(state.get("runtime",{}),"MODEL_SHUTDOWN_TIMEOUT",30); clean=cleanup(root,args.target,run_dir,info,timeout)
    if clean["status"]=="passed": clear_active(root,args.target)
    result={"status":clean["status"],"target":args.target,"run_dir":str(run_dir),"cleanup":clean,"finished_at":now_iso()}; atomic_json(run_dir/"stop-result.json",result); return emit(result,EXIT_OK if clean["status"]=="passed" else EXIT_CLEANUP)


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Runbook model lifecycle manager"); p.add_argument("--repo-root"); sub=p.add_subparsers(dest="command",required=True)
    init=sub.add_parser("init"); init.add_argument("target"); init.add_argument("--force",action="store_true")
    run=sub.add_parser("run"); run.add_argument("target"); run.add_argument("--check",default="smoke"); run.add_argument("--run-dir"); run.add_argument("--check-timeout",type=float)
    sv=sub.add_parser("serve"); sv.add_argument("target"); sv.add_argument("--run-dir")
    ex=sub.add_parser("exec"); ex.add_argument("target"); ex.add_argument("--check",default="smoke"); ex.add_argument("--run-dir"); ex.add_argument("--check-timeout",type=float)
    st=sub.add_parser("status"); st.add_argument("target")
    sp=sub.add_parser("stop"); sp.add_argument("target"); sp.add_argument("--shutdown-timeout",type=float)
    return p


def main(argv: Sequence[str] | None=None) -> int:
    args=parser().parse_args(argv)
    try:
        root=discover_root(args.repo_root); parse_target(args.target)
        if args.command=="init":
            result=copy_runbook_template(root,args.target,args.force)
            return emit(result, EXIT_OK if result.get("status")=="passed" else EXIT_BLOCKED)
        if args.command=="run": return run_ephemeral(args,root)
        if args.command=="serve": return serve(args,root)
        if args.command=="exec": return exec_check(args,root)
        if args.command=="status": return status_cmd(args,root)
        if args.command=="stop": return stop_cmd(args,root)
    except Exception as exc: return emit({"status":"blocked","error":f"{type(exc).__name__}: {exc}"},EXIT_BLOCKED)
    return EXIT_FAILED

if __name__=="__main__": raise SystemExit(main())
