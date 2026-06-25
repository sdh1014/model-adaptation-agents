#!/usr/bin/env python3
"""Structured validation case recorder.

No third-party dependencies. It stores append-only events and derives one summary.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

EXIT_PASSED = 0
EXIT_FAILED = 1
EXIT_BLOCKED = 64
EXIT_PARTIAL = 65
NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
LEVELS = {"required", "optional"}
STATUSES = {"passed", "failed", "blocked", "skipped", "not_applicable"}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"events.jsonl 第 {number} 行不是有效 JSON: {exc}") from exc
        if isinstance(value, dict):
            result.append(value)
    return result


def normalize_error(text: str) -> str:
    text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
    text = re.sub(r"\bpid[=: ]+\d+", "pid=N", text, flags=re.I)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}[T ][^ ]+", "<TIME>", text)
    text = re.sub(r"/[^\s:'\"]+", "<PATH>", text)
    text = re.sub(r"\b\d+\.\d+\b", "N.N", text)
    text = re.sub(r"\b\d+\b", "N", text)
    return " ".join(text.split())[:1200]


def tail(path: str | None, max_lines: int = 25) -> str:
    if not path:
        return ""
    target = Path(path)
    if not target.is_file():
        return ""
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def validate_name_level(name: str, level: str) -> None:
    if not NAME_RE.fullmatch(name):
        raise ValueError("case 名称只能包含字母、数字、点、下划线和连字符")
    if level not in LEVELS:
        raise ValueError("level 必须为 required 或 optional")


def cmd_init(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    events = output / "events.jsonl"
    if events.exists() and not args.force:
        print(json.dumps({"status": "blocked", "reason": "already_initialized", "output_dir": str(output)}, ensure_ascii=False))
        return EXIT_BLOCKED
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases").mkdir(exist_ok=True)
    events.write_text("", encoding="utf-8")
    for stale in (output / "summary.json", output / "failure-signature.json"):
        try:
            stale.unlink()
        except FileNotFoundError:
            pass
    atomic_json(output / "metadata.json", {"schema_version": 1, "created_at": now(), "kind": "validation"})
    print(json.dumps({"status": "passed", "output_dir": str(output)}, ensure_ascii=False))
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    validate_name_level(args.name, args.level)
    if args.status not in STATUSES:
        raise ValueError(f"不支持的状态: {args.status}")
    output = Path(args.output_dir).resolve()
    if not (output / "events.jsonl").exists():
        raise ValueError("validation 尚未初始化")
    event = {
        "schema_version": 1,
        "type": "case",
        "recorded_at": now(),
        "name": args.name,
        "level": args.level,
        "status": args.status,
        "exit_code": args.exit_code,
        "duration_seconds": args.duration_seconds,
        "stdout": args.stdout,
        "stderr": args.stderr,
        "reason": args.reason,
    }
    append_event(output / "events.jsonl", event)
    print(json.dumps(event, ensure_ascii=False))
    return 0


def derive_status(cases: list[dict[str, Any]]) -> str:
    required = [case for case in cases if case.get("level") == "required"]
    optional = [case for case in cases if case.get("level") == "optional"]
    if not cases:
        return "blocked"
    if any(case.get("status") == "failed" for case in required):
        return "failed"
    if any(case.get("status") == "blocked" for case in required):
        return "blocked"
    if not required:
        return "partial"
    if any(case.get("status") == "skipped" for case in required):
        return "partial"
    if any(case.get("status") in {"failed", "blocked", "skipped"} for case in optional):
        return "partial"
    return "passed"


def status_exit(status: str) -> int:
    return {
        "passed": EXIT_PASSED,
        "failed": EXIT_FAILED,
        "blocked": EXIT_BLOCKED,
        "partial": EXIT_PARTIAL,
    }.get(status, EXIT_FAILED)


def cmd_finalize(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    events = read_events(output / "events.jsonl")
    cases = [event for event in events if event.get("type") == "case"]
    names = [str(case.get("name")) for case in cases]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        status = "failed"
    else:
        status = derive_status(cases)
    counts = Counter(str(case.get("status")) for case in cases)
    required_total = sum(case.get("level") == "required" for case in cases)
    required_completed = sum(
        case.get("level") == "required" and case.get("status") in {"passed", "not_applicable"}
        for case in cases
    )
    signature_payload: list[dict[str, Any]] = []
    for case in cases:
        if case.get("status") not in {"failed", "blocked"}:
            continue
        error = normalize_error(tail(case.get("stderr")) or tail(case.get("stdout")) or str(case.get("reason") or ""))
        signature_payload.append(
            {
                "name": case.get("name"),
                "status": case.get("status"),
                "exit_code": case.get("exit_code"),
                "error": error,
            }
        )
    signature = None
    if signature_payload:
        encoded = json.dumps(signature_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        signature = hashlib.sha256(encoded).hexdigest()[:20]
        atomic_json(
            output / "failure-signature.json",
            {"schema_version": 1, "signature": signature, "components": signature_payload},
        )
    summary = {
        "schema_version": 1,
        "kind": "validation",
        "status": status,
        "created_at": now(),
        "case_count": len(cases),
        "required_total": required_total,
        "required_completed": required_completed,
        "coverage_complete": bool(required_total) and required_total == required_completed,
        "counts": dict(sorted(counts.items())),
        "duplicates": duplicates,
        "failure_signature": signature,
        "cases": cases,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return status_exit(status)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="记录和汇总结构化正确性验证 case")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--output-dir", required=True)
    init.add_argument("--force", action="store_true")

    record = sub.add_parser("record")
    record.add_argument("--output-dir", required=True)
    record.add_argument("--name", required=True)
    record.add_argument("--level", required=True)
    record.add_argument("--status", required=True)
    record.add_argument("--exit-code", type=int)
    record.add_argument("--duration-seconds", type=float)
    record.add_argument("--stdout")
    record.add_argument("--stderr")
    record.add_argument("--reason")

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            return cmd_init(args)
        if args.command == "record":
            return cmd_record(args)
        if args.command == "finalize":
            return cmd_finalize(args)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return EXIT_BLOCKED
    return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
