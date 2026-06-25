#!/usr/bin/env python3
"""Summarize prior adapt-implement runs for one work item without hiding raw evidence."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def read_excerpt(path: Path, *, max_chars: int = 6000, tail_lines: int | None = None) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if tail_lines is not None:
        text = "\n".join(text.splitlines()[-tail_lines:])
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def rel_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def attempt_record(attempt_dir: Path, root: Path) -> dict[str, Any]:
    verification = attempt_dir / "verification"
    command_dir = verification / "command"
    signature = load_json(attempt_dir / "failure-signature.json")
    command = load_json(command_dir / "command.json")
    command_result = load_json(command_dir / "result.json")
    summary = load_json(verification / "summary.json")
    scope_before = load_json(verification / "scope-before.json")
    scope_after = load_json(verification / "scope-after.json")

    record: dict[str, Any] = {
        "attempt": attempt_dir.name,
        "attempt_dir": rel_or_abs(attempt_dir, root),
        "attempt_record": rel_or_abs(attempt_dir / "attempt.md", root)
        if (attempt_dir / "attempt.md").exists()
        else None,
        "attempt_excerpt": read_excerpt(attempt_dir / "attempt.md", max_chars=10000),
        "verification": summary,
        "command": command,
        "command_result": command_result,
        "scope_before": scope_before,
        "scope_after": scope_after,
        "failure_signature": signature,
        "stdout_path": rel_or_abs(command_dir / "stdout.log", root)
        if (command_dir / "stdout.log").exists()
        else None,
        "stderr_path": rel_or_abs(command_dir / "stderr.log", root)
        if (command_dir / "stderr.log").exists()
        else None,
        "stdout_tail": read_excerpt(command_dir / "stdout.log", tail_lines=40),
        "stderr_tail": read_excerpt(command_dir / "stderr.log", tail_lines=40),
    }
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-runs", type=int, default=20)
    args = parser.parse_args()
    if args.max_runs < 1:
        parser.error("--max-runs must be >= 1")

    root = Path(args.runs_root).expanduser().resolve()
    records: list[dict[str, Any]] = []
    signatures: Counter[str] = Counter()
    total_attempts = 0

    if root.exists():
        for metadata_path in sorted(root.glob("*-implement-*/metadata.json")):
            metadata = load_json(metadata_path)
            if not metadata or metadata.get("stage") != "adapt-implement":
                continue
            if metadata.get("item_id") != args.item_id:
                continue

            run_dir = metadata_path.parent
            outcome = load_json(run_dir / "outcome.json")
            attempts: list[dict[str, Any]] = []
            attempts_root = run_dir / "attempts"
            if attempts_root.exists():
                for attempt_dir in sorted(path for path in attempts_root.iterdir() if path.is_dir()):
                    record = attempt_record(attempt_dir, root)
                    attempts.append(record)
                    total_attempts += 1
                    signature = record.get("failure_signature") or {}
                    hash_value = signature.get("signature_hash")
                    if hash_value:
                        signatures[str(hash_value)] += 1

            records.append(
                {
                    "run_dir": rel_or_abs(run_dir, root),
                    "created_at": metadata.get("created_at"),
                    "status": outcome.get("status") if outcome else None,
                    "summary": outcome.get("summary") if outcome else None,
                    "latest_failure_signature": outcome.get("latest_failure_signature") if outcome else None,
                    "blocker": outcome.get("blocker") if outcome else None,
                    "next_action": outcome.get("next_action") if outcome else None,
                    "attempts": attempts,
                }
            )

    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "item_id": args.item_id,
        "runs_root": str(root),
        "invocation_count": len(records),
        "observed_attempt_count": total_attempts,
        "signature_counts": dict(signatures.most_common()),
        "runs": records[-args.max_runs :],
        "note": "Excerpts are navigation aids. Raw command, stdout and stderr files remain authoritative.",
    }
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "invocations": len(records),
                "attempts": total_attempts,
                "signatures": dict(signatures),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
