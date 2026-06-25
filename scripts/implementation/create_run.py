#!/usr/bin/env python3
"""Create a unique adapt-implement run directory and write immutable run metadata."""
from __future__ import annotations

import argparse
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    value = value.strip("-.")
    if not value:
        raise ValueError("identifier becomes empty after sanitization")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--target-repo", required=True, help="editable repository path")
    parser.add_argument(
        "--repository-role",
        required=True,
        choices=["target_repo", "upstream_repo"],
        help="which target.yaml repository this work item edits",
    )
    parser.add_argument("--runs-root", default="runs")
    args = parser.parse_args()

    target_repo = Path(args.target_repo).expanduser().resolve()
    if not target_repo.is_dir():
        parser.error(f"editable repository does not exist: {target_repo}")

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d-%H%M%SZ")
    suffix = uuid.uuid4().hex[:6]
    run_dir = (
        Path(args.runs_root)
        / slug(args.model_id)
        / slug(args.target_id)
        / f"{stamp}-implement-{slug(args.item_id)}-{suffix}"
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "attempts").mkdir()

    metadata = {
        "schema_version": 1,
        "stage": "adapt-implement",
        "model_id": args.model_id,
        "target_id": args.target_id,
        "item_id": args.item_id,
        "repository_role": args.repository_role,
        "editable_repo": str(target_repo),
        "target_repo": str(target_repo),
        "created_at": now.isoformat(),
        "run_dir": str(run_dir),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
