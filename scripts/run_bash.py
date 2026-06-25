#!/usr/bin/env python3
"""Execute a command and persist command, stdout, stderr, timeout and exit facts."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def terminate_process_tree(proc: subprocess.Popen[bytes], grace_seconds: int = 10) -> dict[str, Any]:
    actions: list[str] = []
    if proc.poll() is not None:
        return {"actions": actions, "return_code": proc.returncode}

    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
            actions.append("SIGTERM process group")
        else:  # pragma: no cover - assessment targets Linux, but keep a safe fallback.
            proc.terminate()
            actions.append("terminate process")
    except ProcessLookupError:
        return {"actions": actions, "return_code": proc.poll()}

    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
                actions.append("SIGKILL process group")
            else:  # pragma: no cover
                proc.kill()
                actions.append("kill process")
        except ProcessLookupError:
            pass
        proc.wait()
    return {"actions": actions, "return_code": proc.returncode}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--cwd", default=None, help="working directory for the command")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")
    if args.timeout_seconds is not None and args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be >= 1")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    (run_dir / "command.json").write_text(
        json.dumps(
            {
                "argv": command,
                "started_at": started,
                "timeout_seconds": args.timeout_seconds,
                "working_directory": str(Path(args.cwd).expanduser().resolve()) if args.cwd else os.getcwd(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    timed_out = False
    launch_error: str | None = None
    cleanup: dict[str, Any] = {"actions": []}
    return_code = 127

    with (run_dir / "stdout.log").open("wb") as out, (run_dir / "stderr.log").open("wb") as err:
        try:
            cwd = str(Path(args.cwd).expanduser().resolve()) if args.cwd else None
            if cwd is not None and not Path(cwd).is_dir():
                raise OSError(f"working directory does not exist: {cwd}")
            proc = subprocess.Popen(
                command,
                stdout=out,
                stderr=err,
                cwd=cwd,
                start_new_session=(os.name == "posix"),
            )
        except OSError as exc:
            launch_error = f"{type(exc).__name__}: {exc}"
            err.write((launch_error + "\n").encode("utf-8", errors="replace"))
        else:
            try:
                return_code = proc.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                cleanup = terminate_process_tree(proc)
                return_code = 124
            except KeyboardInterrupt:
                cleanup = terminate_process_tree(proc)
                return_code = 130

    result = {
        "exit_code": return_code,
        "started_at": started,
        "finished_at": utc_now(),
        "timed_out": timed_out,
        "launch_error": launch_error,
        "cleanup": cleanup,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
