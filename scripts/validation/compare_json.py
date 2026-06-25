#!/usr/bin/env python3
"""Compare JSON values recursively with numeric tolerances."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence


def load(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ignored(path: str, patterns: set[str]) -> bool:
    return path in patterns or any(path.startswith(pattern + ".") for pattern in patterns)


def compare(ref: Any, actual: Any, path: str, abs_tol: float, rel_tol: float, ignores: set[str], out: list[dict[str, Any]], limit: int) -> None:
    if len(out) >= limit or ignored(path, ignores):
        return
    if isinstance(ref, bool) or isinstance(actual, bool):
        if ref != actual:
            out.append({"path": path, "reference": ref, "actual": actual, "reason": "value_mismatch"})
        return
    if isinstance(ref, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(float(ref), float(actual), rel_tol=rel_tol, abs_tol=abs_tol):
            out.append({"path": path, "reference": ref, "actual": actual, "abs_error": abs(float(ref) - float(actual)), "reason": "numeric_mismatch"})
        return
    if type(ref) is not type(actual):
        out.append({"path": path, "reference_type": type(ref).__name__, "actual_type": type(actual).__name__, "reason": "type_mismatch"})
        return
    if isinstance(ref, dict):
        keys = sorted(set(ref) | set(actual))
        for key in keys:
            child = f"{path}.{key}" if path else str(key)
            if ignored(child, ignores):
                continue
            if key not in ref:
                out.append({"path": child, "reason": "unexpected_key"})
            elif key not in actual:
                out.append({"path": child, "reason": "missing_key"})
            else:
                compare(ref[key], actual[key], child, abs_tol, rel_tol, ignores, out, limit)
            if len(out) >= limit:
                return
        return
    if isinstance(ref, list):
        if len(ref) != len(actual):
            out.append({"path": path, "reference_length": len(ref), "actual_length": len(actual), "reason": "length_mismatch"})
            return
        for index, (left, right) in enumerate(zip(ref, actual)):
            compare(left, right, f"{path}[{index}]", abs_tol, rel_tol, ignores, out, limit)
            if len(out) >= limit:
                return
        return
    if ref != actual:
        out.append({"path": path, "reference": ref, "actual": actual, "reason": "value_mismatch"})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="递归比较两个 JSON 文件")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--actual", required=True)
    parser.add_argument("--abs-tol", type=float, default=0.0)
    parser.add_argument("--rel-tol", type=float, default=0.0)
    parser.add_argument("--ignore", action="append", default=[])
    parser.add_argument("--max-mismatches", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        mismatches: list[dict[str, Any]] = []
        compare(load(args.reference), load(args.actual), "", args.abs_tol, args.rel_tol, set(args.ignore), mismatches, args.max_mismatches)
        result = {"status": "passed" if not mismatches else "failed", "mismatch_count": len(mismatches), "mismatches": mismatches, "abs_tol": args.abs_tol, "rel_tol": args.rel_tol}
        text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(text, encoding="utf-8")
        print(text, end="")
        return 0 if not mismatches else 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 64


if __name__ == "__main__":
    raise SystemExit(main())
