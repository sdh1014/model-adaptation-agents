from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def command_output(command: list[str]) -> dict[str, object]:
    try:
        process = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        return {
            "command": command,
            "exit_code": process.returncode,
            "stdout": process.stdout.strip(),
            "stderr": process.stderr.strip(),
        }
    except Exception as error:
        return {
            "command": command,
            "exit_code": None,
            "error": str(error),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-repo")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "target_repo": args.target_repo,
        "checks": {
            "python_version": command_output([sys.executable, "--version"]),
        },
    }

    if args.target_repo:
        data["checks"]["git_commit"] = command_output(
            ["git", "-C", args.target_repo, "rev-parse", "HEAD"]
        )
        data["checks"]["git_branch"] = command_output(
            ["git", "-C", args.target_repo, "branch", "--show-current"]
        )

    (output_dir / "environment.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "result.json").write_text(
        json.dumps({"status": "PASS", "artifacts": ["environment.json"]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

