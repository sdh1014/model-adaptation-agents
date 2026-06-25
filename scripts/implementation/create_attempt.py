#!/usr/bin/env python3
"""Create the next numbered attempt directory inside one adapt-implement run."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.max_attempts <= 5:
        parser.error("--max-attempts must be between 1 and 5")

    run_dir = Path(args.run_dir).expanduser().resolve()
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        parser.error(f"missing run metadata: {metadata_path}")
    try:
        run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"invalid run metadata: {exc}")
    if run_metadata.get("stage") != "adapt-implement":
        parser.error("run metadata is not for adapt-implement")

    attempts_root = run_dir / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        int(path.name)
        for path in attempts_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    next_number = (existing[-1] + 1) if existing else 1
    if next_number > args.max_attempts:
        print(
            json.dumps(
                {
                    "status": "limit_reached",
                    "max_attempts": args.max_attempts,
                    "existing_attempts": existing,
                },
                ensure_ascii=False,
            )
        )
        return 5

    attempt_dir = attempts_root / f"{next_number:02d}"
    attempt_dir.mkdir(exist_ok=False)
    attempt_metadata = {
        "schema_version": 1,
        "stage": "adapt-implement-attempt",
        "attempt": next_number,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "attempt_dir": str(attempt_dir),
        "item_id": run_metadata.get("item_id"),
    }
    (attempt_dir / "metadata.json").write_text(
        json.dumps(attempt_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(attempt_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
