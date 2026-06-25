#!/usr/bin/env python3
"""Capture a read-only Git repository snapshot and cumulative patch evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_info(path: Path, relative_path: str | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if relative_path is not None:
        info["relative_path"] = relative_path
    if not path.exists() or not path.is_file():
        return info
    size = path.stat().st_size
    info["size_bytes"] = size
    if size <= 16 * 1024 * 1024:
        info["sha256"] = sha256_bytes(path.read_bytes())
    else:
        info["sha256"] = None
        info["hash_skipped_reason"] = "file larger than 16 MiB"
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-repo", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--phase", required=True, choices=["before", "after"])
    parser.add_argument("--base-ref", default="HEAD")
    args = parser.parse_args()

    repo = Path(args.target_repo).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if run_git(repo, "rev-parse", "--is-inside-work-tree", check=False).stdout.strip() != "true":
        parser.error(f"not a Git work tree: {repo}")

    base_check = run_git(repo, "rev-parse", "--verify", f"{args.base_ref}^{{commit}}", check=False)
    if base_check.returncode != 0:
        parser.error(f"invalid --base-ref {args.base_ref!r}: {base_check.stderr.strip()}")

    head = run_git(repo, "rev-parse", "HEAD").stdout.strip()
    branch = run_git(repo, "branch", "--show-current", check=False).stdout.strip() or None
    remote = run_git(repo, "remote", "get-url", "origin", check=False)
    remote_url = remote.stdout.strip() if remote.returncode == 0 else None
    status = run_git(repo, "status", "--porcelain=v1", "--untracked-files=all").stdout
    changed = run_git(repo, "diff", "--name-only", args.base_ref, "--").stdout.splitlines()
    staged = run_git(repo, "diff", "--cached", "--name-only", "--").stdout.splitlines()
    untracked = run_git(repo, "ls-files", "--others", "--exclude-standard").stdout.splitlines()

    patch_proc = run_git(repo, "diff", "--binary", args.base_ref, "--")
    patch_chunks = [patch_proc.stdout.encode("utf-8", errors="replace")]
    # Include reasonably sized untracked files so new source files are reviewable in patch evidence.
    for rel in untracked:
        path = repo / rel
        if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            continue
        proc = run_git(repo, "diff", "--no-index", "--binary", "--", "/dev/null", rel, check=False)
        if proc.returncode not in (0, 1):
            continue
        patch_chunks.append(proc.stdout.encode("utf-8", errors="replace"))
    patch = b"".join(patch_chunks)
    patch_path = run_dir / f"patch-{args.phase}.diff"
    patch_path.write_bytes(patch)

    status_path = run_dir / f"git-status-{args.phase}.txt"
    status_path.write_text(status, encoding="utf-8")

    changed_all = sorted(set(changed) | set(staged) | set(untracked))
    (run_dir / f"changed-files-{args.phase}.txt").write_text(
        "\n".join(changed_all) + ("\n" if changed_all else ""),
        encoding="utf-8",
    )

    snapshot = {
        "schema_version": 1,
        "phase": args.phase,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "base_ref": args.base_ref,
        "base_commit": base_check.stdout.strip(),
        "head": head,
        "branch": branch,
        "origin": remote_url,
        "dirty": bool(status.strip()),
        "status_porcelain": status.splitlines(),
        "changed_files_from_base": changed,
        "staged_files": staged,
        "untracked_files": [file_info(repo / rel, rel) for rel in untracked],
        "working_tree_files": [file_info(repo / rel, rel) for rel in changed_all],
        "all_observed_changed_files": changed_all,
        "patch_path": str(patch_path),
        "patch_sha256": sha256_bytes(patch),
    }
    (run_dir / f"repo-{args.phase}.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"head": head, "patch_sha256": snapshot["patch_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
