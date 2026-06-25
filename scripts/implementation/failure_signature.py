#!/usr/bin/env python3
"""Extract a stable failure signature from command evidence without claiming root cause."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXCEPTION_RE = re.compile(
    r"(?P<type>[A-Za-z_][\w.]*?(?:Error|Exception|Failure|Abort|Timeout))\s*:\s*(?P<message>.+)"
)
FAILED_TEST_RE = re.compile(r"^(?:FAILED|ERROR)\s+([^\s]+(?:::[^\s]+)*)", re.MULTILINE)


def read_text(path: str | None, max_bytes: int = 2_000_000) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    data = p.read_bytes()
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    return data.decode("utf-8", errors="replace")


def normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    text = re.sub(r"0x[0-9a-fA-F]+", "<HEX>", text)
    text = re.sub(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}\b", "<UUID>", text)
    text = re.sub(r"\b(?:pid|process)\s*[=:]?\s*\d+\b", "PID=<PID>", text, flags=re.I)
    text = re.sub(r"(?<![A-Za-z0-9_])/(?:[^\s:]+/)+([^/\s:]+\.py):\d+", r"\1:<LINE>", text)
    text = re.sub(r"\.py:\d+", ".py:<LINE>", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}[T ][0-9:.+Z-]+\b", "<TIME>", text)
    text = re.sub(r"\s+", " ", text)
    return text[:1200]


def meaningful_lines(text: str) -> list[str]:
    ignored = ("Traceback (most recent call last):", "During handling of the above exception")
    lines = []
    for raw in text.splitlines():
        line = normalize(raw)
        if not line or line in ignored:
            continue
        lines.append(line)
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-result")
    parser.add_argument("--stdout")
    parser.add_argument("--stderr")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result_data: dict[str, Any] = {}
    if args.command_result and Path(args.command_result).exists():
        try:
            result_data = json.loads(Path(args.command_result).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            result_data = {}

    stdout = read_text(args.stdout)
    stderr = read_text(args.stderr)
    combined = stderr + "\n" + stdout
    lines = meaningful_lines(combined)

    exception_type = None
    exception_message = None
    for line in reversed(lines):
        match = EXCEPTION_RE.search(line)
        if match:
            exception_type = match.group("type")
            exception_message = normalize(match.group("message"))
            break

    failed_tests = sorted(set(FAILED_TEST_RE.findall(combined)))[:20]
    last_lines = lines[-8:]
    signature_components = {
        "exit_code": result_data.get("exit_code"),
        "timed_out": bool(result_data.get("timed_out", False)),
        "launch_error": normalize(str(result_data.get("launch_error") or "")) or None,
        "exception_type": exception_type,
        "exception_message": exception_message,
        "failed_tests": failed_tests,
        "last_stable_lines": last_lines,
    }
    canonical = json.dumps(signature_components, sort_keys=True, ensure_ascii=False)
    signature_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]

    output = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "signature_hash": signature_hash,
        "signature": signature_components,
        "root_cause": None,
        "root_cause_confirmed": False,
        "note": "This is an observable failure signature, not a root-cause conclusion.",
        "sources": {
            "command_result": args.command_result,
            "stdout": args.stdout,
            "stderr": args.stderr,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(signature_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
