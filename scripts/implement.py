#!/usr/bin/env python3
"""Consolidated evidence helpers for adapt-implement."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from paths import target_run_key


def git(repo: Path, *args: str, binary: bool = False) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
        check=False,
    )


def normalize_path(value: str) -> str:
    return value.replace("\\", "/").lstrip("./").rstrip("/")


def allowed_path(path: str, rules: list[str]) -> bool:
    path = normalize_path(path)
    for raw in rules:
        rule = normalize_path(raw)
        if any(char in rule for char in "*?[") and fnmatch.fnmatch(path, rule):
            return True
        if path == rule or path.startswith(rule + "/"):
            return True
    return False


def create_run(args: argparse.Namespace) -> int:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", args.item)
    root = Path(args.runs_root) / target_run_key(args.model, args.target)
    run = root / f"{stamp}-implement-{safe}"
    index = 1
    while run.exists():
        run = root / f"{stamp}-implement-{safe}-{index}"
        index += 1
    run.mkdir(parents=True)
    metadata = {
        "schema_version": 1,
        "stage": "adapt-implement",
        "model_id": args.model,
        "target_id": args.target,
        "item_id": args.item,
        "target_repo": str(Path(args.target_repo).resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(run)
    return 0


def snapshot(args: argparse.Namespace) -> int:
    repo = Path(args.target_repo).resolve()
    output = Path(args.run_dir)
    output.mkdir(parents=True, exist_ok=True)
    root = git(repo, "rev-parse", "--show-toplevel")
    if root.returncode:
        print(root.stderr, file=sys.stderr)
        return 2
    repo = Path(root.stdout.strip())
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    branch = git(repo, "branch", "--show-current").stdout.strip()
    status = git(repo, "status", "--porcelain=v1", "--untracked-files=all", binary=True).stdout
    diff = git(repo, "diff", "--binary", args.base_ref, "--", binary=True).stdout
    untracked = git(repo, "ls-files", "--others", "--exclude-standard").stdout.splitlines()
    changed = git(repo, "diff", "--name-status", args.base_ref, "--").stdout.splitlines()
    payload = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo),
        "phase": args.phase,
        "base_ref": args.base_ref,
        "head": head,
        "branch": branch,
        "dirty": bool(status.strip()),
        "status_lines": status.decode(errors="replace").splitlines(),
        "changed_lines": changed,
        "untracked": untracked,
        "patch_sha256": hashlib.sha256(diff).hexdigest(),
    }
    (output / f"repo-{args.phase}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / f"patch-{args.phase}.diff").write_bytes(diff)
    return 0


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:  # noqa: BLE001
        return None


def history(args: argparse.Namespace) -> int:
    root = Path(args.runs_root)
    runs: list[dict[str, Any]] = []
    if root.exists():
        for metadata_path in sorted(root.glob("*-implement-*/metadata.json")):
            metadata = load_json(metadata_path)
            if not metadata or metadata.get("item_id") != args.item:
                continue
            run = metadata_path.parent
            attempts: list[dict[str, Any]] = []
            attempts_dir = run / "attempts"
            if attempts_dir.is_dir():
                for attempt_dir in sorted(path for path in attempts_dir.iterdir() if path.is_dir()):
                    attempts.append(
                        {
                            "attempt": attempt_dir.name,
                            "path": str(attempt_dir),
                            "signature": load_json(attempt_dir / "failure-signature.json"),
                            "attempt_md": str(attempt_dir / "attempt.md") if (attempt_dir / "attempt.md").exists() else None,
                        }
                    )
            runs.append({"run_dir": str(run), "metadata": metadata, "outcome": load_json(run / "outcome.json"), "attempts": attempts})
    payload = {"item_id": args.item, "run_count": len(runs), "runs": runs[-20:]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def normalize_failure(text: str) -> str:
    lines: list[str] = []
    for raw in text.splitlines()[-160:]:
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", line)
        line = re.sub(r"\bpid[=: ]+\d+\b", "pid=<PID>", line, flags=re.I)
        line = re.sub(r"/tmp/[A-Za-z0-9_./-]+", "/tmp/<PATH>", line)
        lines.append(line)
    return "\n".join(lines)


def signature(args: argparse.Namespace) -> int:
    text = ""
    for value in (args.stderr, args.stdout):
        if value and Path(value).is_file():
            text += Path(value).read_text(encoding="utf-8", errors="replace") + "\n"
    normalized = normalize_failure(text)
    payload = {
        "signature": normalized.splitlines()[-12:] if normalized else [],
        "signature_hash": hashlib.sha256(normalized.encode()).hexdigest()[:16] if normalized else None,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def changed_paths(repo: Path, base_ref: str) -> tuple[list[str], str | None]:
    diff = git(repo, "diff", "--name-only", base_ref, "--")
    untracked = git(repo, "ls-files", "--others", "--exclude-standard")
    if diff.returncode or untracked.returncode:
        return [], (diff.stderr or untracked.stderr)
    return sorted({normalize_path(item) for item in diff.stdout.splitlines() + untracked.stdout.splitlines() if item.strip()}), None


def scope(args: argparse.Namespace) -> int:
    if not args.allow:
        raise SystemExit("scope requires at least one --allow")
    repo = Path(args.target_repo).resolve()
    root = git(repo, "rev-parse", "--show-toplevel")
    if root.returncode:
        print(root.stderr, file=sys.stderr)
        return 2
    repo = Path(root.stdout.strip())
    changed, error = changed_paths(repo, args.base_ref)
    if error:
        print(error, file=sys.stderr)
        return 2
    violations = [path for path in changed if not allowed_path(path, args.allow)]
    payload = {"passed": not violations, "base_ref": args.base_ref, "allowed_rules": args.allow, "changed_paths": changed, "violations": violations}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if not violations else 4


def check(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    repo = Path(args.target_repo).resolve()
    output = Path(args.run_dir)
    output.mkdir(parents=True, exist_ok=True)
    scope_args = argparse.Namespace(target_repo=str(repo), base_ref=args.base_ref, output=str(output / "scope.json"), allow=args.allow)
    scope_code = scope(scope_args)
    if scope_code != 0:
        return scope_code
    diff_check = subprocess.run(["git", "-C", str(repo), "diff", "--check", args.base_ref, "--"], check=False)
    if diff_check.returncode != 0:
        return diff_check.returncode
    changed, error = changed_paths(repo, args.base_ref)
    if error:
        print(error, file=sys.stderr)
        return 2
    python_files = [path for path in changed if path.endswith(".py") and (repo / path).is_file()]
    if python_files:
        compile_result = subprocess.run([sys.executable, "-m", "py_compile", *python_files], cwd=str(repo), check=False)
        if compile_result.returncode != 0:
            return compile_result.returncode
    if not command:
        return 0
    runner = Path(__file__).resolve().parent / "run_bash.py"
    return subprocess.call(
        [sys.executable, str(runner), "--run-dir", str(output / "command"), "--cwd", str(repo), "--timeout", str(args.timeout), "--", *command]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Implementation evidence helpers")
    sub = parser.add_subparsers(dest="command_name", required=True)
    create = sub.add_parser("create-run")
    create.add_argument("--model", required=True)
    create.add_argument("--target", required=True)
    create.add_argument("--item", required=True)
    create.add_argument("--target-repo", required=True)
    create.add_argument("--runs-root", default="runs")
    create.set_defaults(handler=create_run)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--target-repo", required=True)
    snap.add_argument("--run-dir", required=True)
    snap.add_argument("--phase", choices=("before", "after"), required=True)
    snap.add_argument("--base-ref", required=True)
    snap.set_defaults(handler=snapshot)
    hist = sub.add_parser("history")
    hist.add_argument("--runs-root", required=True)
    hist.add_argument("--item", required=True)
    hist.add_argument("--output", required=True)
    hist.set_defaults(handler=history)
    sig = sub.add_parser("signature")
    sig.add_argument("--stdout")
    sig.add_argument("--stderr")
    sig.add_argument("--output", required=True)
    sig.set_defaults(handler=signature)
    scope_parser = sub.add_parser("scope")
    scope_parser.add_argument("--target-repo", required=True)
    scope_parser.add_argument("--base-ref", required=True)
    scope_parser.add_argument("--output", required=True)
    scope_parser.add_argument("--allow", action="append", default=[])
    scope_parser.set_defaults(handler=scope)
    check_parser = sub.add_parser("check")
    check_parser.add_argument("--target-repo", required=True)
    check_parser.add_argument("--base-ref", required=True)
    check_parser.add_argument("--run-dir", required=True)
    check_parser.add_argument("--allow", action="append", default=[])
    check_parser.add_argument("--timeout", type=float, default=600)
    check_parser.add_argument("command", nargs=argparse.REMAINDER)
    check_parser.set_defaults(handler=check)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
