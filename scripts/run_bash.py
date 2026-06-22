from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("command", nargs="+")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    with (logs_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
        logs_dir / "stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        try:
            process = subprocess.run(
                args.command,
                stdout=stdout,
                stderr=stderr,
                timeout=args.timeout_seconds,
                check=False,
            )
            exit_code = process.returncode
            status = "PASS" if exit_code == 0 else "FAIL"
        except subprocess.TimeoutExpired:
            exit_code = 124
            status = "FAIL"

    result = {
        "status": status,
        "command": args.command,
        "exit_code": exit_code,
        "duration_seconds": round(time.time() - started_at, 3),
        "artifacts": ["logs/stdout.log", "logs/stderr.log"],
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0 if exit_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

