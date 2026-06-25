#!/usr/bin/env python3
"""Compare two structured benchmark summaries."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence


def load(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("summary 顶层必须是对象")
    return value


def metric_map(summary: dict[str, Any], statistic: str) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for case in summary.get("cases", []):
        if not isinstance(case, dict):
            continue
        for name, metric in (case.get("metrics") or {}).items():
            if isinstance(metric, dict) and isinstance(metric.get(statistic), (int, float)):
                result[(str(case.get("name")), str(name))] = metric
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="比较两个 benchmark summary")
    parser.add_argument("--current", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--statistic", default="median")
    parser.add_argument("--tolerance-percent", type=float)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        current = load(args.current)
        baseline = load(args.baseline)
        current_metrics = metric_map(current, args.statistic)
        baseline_metrics = metric_map(baseline, args.statistic)
        rows: list[dict[str, Any]] = []
        regressions: list[dict[str, Any]] = []
        for key in sorted(set(current_metrics) & set(baseline_metrics)):
            base = float(baseline_metrics[key][args.statistic])
            cur = float(current_metrics[key][args.statistic])
            change = None if base == 0 else (cur - base) / abs(base) * 100.0
            direction = current_metrics[key].get("direction") or baseline_metrics[key].get("direction")
            regression = None
            if args.tolerance_percent is not None and change is not None and direction in {"higher", "lower"}:
                regression = change < -args.tolerance_percent if direction == "higher" else change > args.tolerance_percent
            row = {"case": key[0], "metric": key[1], "statistic": args.statistic, "baseline": base, "current": cur, "change_percent": change, "direction": direction, "regression": regression}
            rows.append(row)
            if regression:
                regressions.append(row)
        workload_equal = current.get("workload_fingerprint") == baseline.get("workload_fingerprint")
        execution_equal = current.get("execution_fingerprint") == baseline.get("execution_fingerprint")
        comparable = bool(workload_equal and execution_equal)
        incomparable_reasons: list[str] = []
        if not workload_equal:
            incomparable_reasons.append("workload_definition_changed")
        if not execution_equal:
            incomparable_reasons.append("runtime_definition_changed")
        result = {
            "schema_version": 1,
            "comparable": comparable,
            "incomparable_reasons": incomparable_reasons,
            "workload_fingerprint_current": current.get("workload_fingerprint"),
            "workload_fingerprint_baseline": baseline.get("workload_fingerprint"),
            "execution_fingerprint_equal": execution_equal,
            "statistic": args.statistic,
            "tolerance_percent": args.tolerance_percent,
            "regression": None if args.tolerance_percent is None or not comparable else bool(regressions),
            "rows": rows,
        }
        text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(text, encoding="utf-8")
        print(text, end="")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 64


if __name__ == "__main__":
    raise SystemExit(main())
