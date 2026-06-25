#!/usr/bin/env python3
"""Check files changed during one attempt against explicit allow globs."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_lines(repo: Path, *args: str) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return [line for line in proc.stdout.splitlines() if line]


def file_state(repo: Path, relative_path: str) -> dict[str, Any]:
    path = repo / relative_path
    if not path.exists():
        return {"exists": False, "size_bytes": None, "sha256": None}
    if not path.is_file():
        return {"exists": True, "size_bytes": None, "sha256": None, "kind": "non-file"}
    size = path.stat().st_size
    digest = None
    if size <= 16 * 1024 * 1024:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"exists": True, "size_bytes": size, "sha256": digest}


def normalize_snapshot_state(entry: dict[str, Any] | None) -> dict[str, Any]:
    if not entry:
        return {"exists": False, "size_bytes": None, "sha256": None}
    return {
        "exists": bool(entry.get("exists", False)),
        "size_bytes": entry.get("size_bytes"),
        "sha256": entry.get("sha256"),
    }


def matches(path: str, patterns: list[str]) -> bool:
    normalized = path.replace("\\", "/")
    for pattern in patterns:
        pattern = pattern.replace("\\", "/")
        if fnmatch.fnmatchcase(normalized, pattern):
            return True
        if pattern.endswith("/") and normalized.startswith(pattern):
            return True
        if not any(ch in pattern for ch in "*?[") and (
            normalized == pattern or normalized.startswith(pattern.rstrip("/") + "/")
        ):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-repo", required=True)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--before-snapshot", required=True)
    parser.add_argument("--allow", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    repo = Path(args.target_repo).expanduser().resolve()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not args.allow:
        result = {
            "schema_version": 1,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "status": "not_checked",
            "reason": "no --allow patterns supplied",
            "allowed_patterns": [],
            "changed_during_attempt": [],
            "violations": [],
        }
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 3

    before_path = Path(args.before_snapshot)
    try:
        before = json.loads(before_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read --before-snapshot: {exc}")

    if Path(str(before.get("repo", ""))).resolve() != repo:
        parser.error("before snapshot belongs to a different repository")

    before_map = {
        str(entry.get("relative_path")): entry
        for entry in before.get("working_tree_files", [])
        if entry.get("relative_path")
    }
    before_changed = set(before.get("all_observed_changed_files", []))

    current_changed = set(git_lines(repo, "diff", "--name-only", args.base_ref, "--"))
    current_changed.update(git_lines(repo, "diff", "--cached", "--name-only", "--"))
    current_changed.update(git_lines(repo, "ls-files", "--others", "--exclude-standard"))

    candidates = sorted(before_changed | current_changed)
    changed_during_attempt: list[str] = []
    state_changes: dict[str, Any] = {}
    for rel in candidates:
        before_state = normalize_snapshot_state(before_map.get(rel))
        current_state = file_state(repo, rel)

        # A path that was clean at the attempt snapshot but is now reported by Git
        # is always a change made during this attempt. This also catches deletion of
        # a previously clean tracked file, where both filesystem states appear absent.
        newly_dirty = rel not in before_changed and rel in current_changed
        changed_prior_state = rel in before_changed and before_state != current_state
        became_clean = rel in before_changed and rel not in current_changed
        if newly_dirty or changed_prior_state or became_clean:
            changed_during_attempt.append(rel)
            state_changes[rel] = {
                "before": before_state,
                "after": current_state,
                "newly_dirty": newly_dirty,
                "became_clean": became_clean,
            }

    violations = [path for path in changed_during_attempt if not matches(path, args.allow)]
    result = {
        "schema_version": 1,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not violations else "violated",
        "base_ref": args.base_ref,
        "before_snapshot": str(before_path.resolve()),
        "allowed_patterns": args.allow,
        "changed_during_attempt": changed_during_attempt,
        "state_changes": state_changes,
        "violations": violations,
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if violations:
        for path in violations:
            print(f"scope violation: {path}")
        return 4
    print("scope check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
