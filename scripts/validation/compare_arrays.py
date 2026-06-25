#!/usr/bin/env python3
"""Compare logits or other numeric arrays from JSON/NPY/NPZ files."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

EXIT_MISMATCH = 1
EXIT_BLOCKED = 64


def json_path(value: Any, path: str | None) -> Any:
    if not path:
        return value
    current = value
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise KeyError(path)
    return current


def load_array(path: Path, key: str | None, np: Any) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        try:
            names = list(archive.files)
            selected = key
            if selected is None:
                if len(names) != 1:
                    raise ValueError(f"NPZ 包含多个数组，需指定 --key: {names}")
                selected = names[0]
            if selected not in archive:
                raise KeyError(selected)
            return np.asarray(archive[selected])
        finally:
            archive.close()
    value = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(json_path(value, key))


def finite_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="比较 JSON/NPY/NPZ 数值数组")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--actual", required=True)
    parser.add_argument("--reference-key")
    parser.add_argument("--actual-key")
    parser.add_argument("--abs-tol", type=float, default=1e-4)
    parser.add_argument("--rel-tol", type=float, default=1e-4)
    parser.add_argument("--equal-nan", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    try:
        import numpy as np  # type: ignore
    except ImportError:
        print(json.dumps({"status": "blocked", "error": "compare_arrays.py 需要 NumPy"}, ensure_ascii=False), file=sys.stderr)
        return EXIT_BLOCKED

    try:
        reference = load_array(Path(args.reference), args.reference_key, np)
        actual = load_array(Path(args.actual), args.actual_key, np)
        result: dict[str, Any] = {
            "schema_version": 1,
            "reference": str(Path(args.reference).resolve()),
            "actual": str(Path(args.actual).resolve()),
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
            "reference_dtype": str(reference.dtype),
            "actual_dtype": str(actual.dtype),
            "abs_tol": args.abs_tol,
            "rel_tol": args.rel_tol,
            "equal_nan": args.equal_nan,
        }
        if reference.shape != actual.shape:
            result.update({"status": "failed", "reason": "shape_mismatch", "passed": False})
        elif reference.size == 0:
            result.update({"status": "passed", "reason": "both_empty", "passed": True, "element_count": 0})
        else:
            ref = reference.astype(np.float64, copy=False)
            act = actual.astype(np.float64, copy=False)
            close = np.isclose(ref, act, rtol=args.rel_tol, atol=args.abs_tol, equal_nan=args.equal_nan)
            difference = np.abs(act - ref)
            denominator = np.maximum(np.abs(ref), np.finfo(np.float64).tiny)
            relative = difference / denominator
            finite_diff = difference[np.isfinite(difference)]
            finite_rel = relative[np.isfinite(relative)]
            passed = bool(np.all(close))
            result.update(
                {
                    "status": "passed" if passed else "failed",
                    "passed": passed,
                    "element_count": int(reference.size),
                    "mismatched_count": int(np.size(close) - np.count_nonzero(close)),
                    "mismatched_ratio": float(1.0 - np.count_nonzero(close) / np.size(close)),
                    "max_abs_error": finite_number(np.max(finite_diff)) if finite_diff.size else None,
                    "mean_abs_error": finite_number(np.mean(finite_diff)) if finite_diff.size else None,
                    "max_rel_error": finite_number(np.max(finite_rel)) if finite_rel.size else None,
                    "reference_finite": int(np.count_nonzero(np.isfinite(ref))),
                    "actual_finite": int(np.count_nonzero(np.isfinite(act))),
                }
            )
        text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
        print(text, end="")
        return 0 if result.get("passed") else EXIT_MISMATCH
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
