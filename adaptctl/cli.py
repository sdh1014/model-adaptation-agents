from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

from .paths import SCRIPTS_DIR, task_dir
from .state import create_run, init_task, read_task, write_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adaptctl")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init")
    init.add_argument("task")
    init.add_argument("--model", default="")
    init.add_argument("--target-repo", default="")
    init.add_argument("--framework", default="")
    init.add_argument("--hardware", default="P800")

    status = commands.add_parser("status")
    status.add_argument("task")

    run = commands.add_parser("run")
    run.add_argument("task")
    run.add_argument("stage", choices=["env", "model"])
    run.add_argument("--model-path", default="")

    return parser


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "init":
        path = init_task(
            args.task,
            model=args.model,
            target_repo=args.target_repo,
            framework=args.framework,
            hardware=args.hardware,
        )
        print(path)
        return 0

    if args.command == "status":
        data = read_task(args.task)
        status_path = task_dir(args.task) / "status.md"
        print_json(
            {
                "task": data,
                "status_file": str(status_path),
                "status": status_path.read_text(encoding="utf-8"),
            }
        )
        return 0

    if args.command == "run":
        task_data = read_task(args.task)
        run_dir = create_run(args.task, stage=args.stage)
        if args.stage == "env":
            command = [
                sys.executable,
                str(SCRIPTS_DIR / "env_survey.py"),
                "--output-dir",
                str(run_dir),
            ]
            if task_data.get("target_repo"):
                command.extend(["--target-repo", task_data["target_repo"]])
        elif args.stage == "model":
            command = [
                sys.executable,
                str(SCRIPTS_DIR / "model_research.py"),
                "--output-dir",
                str(run_dir),
            ]
            if args.model_path:
                command.extend(["--model-path", args.model_path])
        else:
            raise AssertionError(args.stage)

        process = subprocess.run(command, check=False)
        status = "passed" if process.returncode == 0 else "failed"
        write_status(args.task, stage=args.stage, status=status, run_dir=run_dir)
        print_json({"run": str(run_dir), "status": status})
        return process.returncode

    raise AssertionError(args.command)
