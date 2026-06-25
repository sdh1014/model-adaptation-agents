#!/usr/bin/env python3
"""Record, aggregate, and evaluate benchmark cases.

The runner accepts arbitrary benchmark tools as long as each formal iteration writes a
JSON object to BENCHMARK_SAMPLE_FILE. Numeric leaves become metrics.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

EXIT_PASSED = 0
EXIT_FAILED = 1
EXIT_BLOCKED = 64
EXIT_PARTIAL = 65
NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
LEVELS = {"required", "optional"}
MARK_STATUSES = {"blocked", "skipped", "not_applicable"}
DIRECTIONS = {"higher", "lower"}
STATISTICS = {"mean", "median", "min", "max", "p90", "p95", "p99"}


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
    if not path.is_file():
        return []
    result: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"events.jsonl 第 {number} 行无效: {exc}") from exc
        if isinstance(item, dict):
            result.append(item)
    return result


def validate_name_level(name: str, level: str) -> None:
    if not NAME_RE.fullmatch(name):
        raise ValueError("case/metric 名称只能包含字母、数字、点、下划线和连字符")
    if level not in LEVELS:
        raise ValueError("level 必须为 required 或 optional")


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def infer_direction(name: str) -> str | None:
    lower = name.lower()
    if any(token in lower for token in ("throughput", "goodput", "qps", "tokens_per_second", "requests_per_second", "completed")):
        return "higher"
    if any(token in lower for token in ("ttft", "tpot", "itl", "latency", "duration", "elapsed", "time_ms", "time_s")):
        return "lower"
    return None


def infer_unit(name: str) -> str | None:
    lower = name.lower()
    if lower.endswith("_ms") or "latency_ms" in lower:
        return "ms"
    if lower.endswith("_s"):
        return "s"
    if "throughput" in lower or "per_second" in lower:
        return "per_second"
    if "memory" in lower and (lower.endswith("_mb") or "mb" in lower):
        return "MB"
    return None


def flatten_metrics(value: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if isinstance(value, bool):
        return result
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        if prefix:
            result[prefix] = {"value": float(value), "unit": infer_unit(prefix), "direction": infer_direction(prefix), "direction_source": "inferred" if infer_direction(prefix) else None}
        return result
    if isinstance(value, dict):
        if isinstance(value.get("value"), (int, float)) and not isinstance(value.get("value"), bool):
            if prefix and math.isfinite(float(value["value"])):
                direction = value.get("direction") if value.get("direction") in DIRECTIONS else infer_direction(prefix)
                result[prefix] = {
                    "value": float(value["value"]),
                    "unit": value.get("unit") or infer_unit(prefix),
                    "direction": direction,
                    "direction_source": "explicit" if value.get("direction") in DIRECTIONS else ("inferred" if direction else None),
                }
            return result
        for key, child in value.items():
            if key in {"metadata", "args", "config", "command", "environment"}:
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_metrics(child, child_prefix))
    return result


def load_sample(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError("样本文件为空")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = None
        for line in reversed(text.splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                loaded = candidate
                break
        if loaded is None:
            raise ValueError("样本不是 JSON，也没有可解析的 JSON 行")
    if isinstance(loaded, list):
        objects = [item for item in loaded if isinstance(item, dict)]
        if not objects:
            raise ValueError("JSON 列表中没有对象")
        loaded = objects[-1]
    if not isinstance(loaded, dict):
        raise ValueError("样本顶层必须是 JSON 对象")
    metric_source = loaded.get("metrics") if isinstance(loaded.get("metrics"), dict) else loaded
    metrics = flatten_metrics(metric_source)
    if not metrics:
        raise ValueError("样本中没有有限数值指标")
    return {"raw": loaded, "metrics": metrics}


def cmd_init(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    events = output / "events.jsonl"
    if events.exists() and not args.force:
        print(json.dumps({"status": "blocked", "reason": "already_initialized", "output_dir": str(output)}, ensure_ascii=False))
        return EXIT_BLOCKED
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases").mkdir(exist_ok=True)
    events.write_text("", encoding="utf-8")
    for stale in (output / "summary.json",):
        try:
            stale.unlink()
        except FileNotFoundError:
            pass
    atomic_json(output / "metadata.json", {"schema_version": 1, "kind": "benchmark", "created_at": now()})
    return 0


def cmd_declare(args: argparse.Namespace) -> int:
    validate_name_level(args.case, args.level)
    if args.warmup < 0 or args.repeat < 1:
        raise ValueError("warmup 必须 >= 0，repeat 必须 >= 1")
    event = {"schema_version": 1, "type": "case_declaration", "recorded_at": now(), "case": args.case, "level": args.level, "warmup": args.warmup, "repeat": args.repeat}
    append_event(Path(args.output_dir).resolve() / "events.jsonl", event)
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    validate_name_level(args.case, args.level)
    output = Path(args.output_dir).resolve()
    source = Path(args.sample_file).resolve()
    sample = load_sample(source)
    case_dir = output / "cases" / args.case / "samples"
    case_dir.mkdir(parents=True, exist_ok=True)
    destination = case_dir / f"{args.iteration:03d}.json"
    shutil.copy2(source, destination)
    event = {
        "schema_version": 1,
        "type": "sample",
        "recorded_at": now(),
        "case": args.case,
        "level": args.level,
        "iteration": args.iteration,
        "duration_seconds": args.duration_seconds,
        "stdout": args.stdout,
        "stderr": args.stderr,
        "sample_file": str(destination),
        "metrics": sample["metrics"],
    }
    append_event(output / "events.jsonl", event)
    return 0


def cmd_failure(args: argparse.Namespace) -> int:
    validate_name_level(args.case, args.level)
    status = args.status
    if status not in {"failed", "blocked"}:
        raise ValueError("failure status 必须为 failed 或 blocked")
    event = {
        "schema_version": 1,
        "type": "case_failure",
        "recorded_at": now(),
        "case": args.case,
        "level": args.level,
        "iteration": args.iteration,
        "phase": args.phase,
        "status": status,
        "exit_code": args.exit_code,
        "duration_seconds": args.duration_seconds,
        "stdout": args.stdout,
        "stderr": args.stderr,
        "reason": args.reason,
    }
    append_event(Path(args.output_dir).resolve() / "events.jsonl", event)
    return 0


def cmd_mark(args: argparse.Namespace) -> int:
    validate_name_level(args.case, args.level)
    if args.status not in MARK_STATUSES:
        raise ValueError("mark status 必须为 blocked、skipped 或 not_applicable")
    event = {"schema_version": 1, "type": "case_mark", "recorded_at": now(), "case": args.case, "level": args.level, "status": args.status, "reason": args.reason}
    append_event(Path(args.output_dir).resolve() / "events.jsonl", event)
    return 0


def cmd_expect(args: argparse.Namespace) -> int:
    if not NAME_RE.fullmatch(args.case) or not NAME_RE.fullmatch(args.metric.replace(".", "-")):
        raise ValueError("case 或 metric 名称非法")
    if args.direction not in DIRECTIONS or args.statistic not in STATISTICS:
        raise ValueError("direction/statistic 不支持")
    event = {
        "schema_version": 1,
        "type": "expectation",
        "recorded_at": now(),
        "case": args.case,
        "metric": args.metric,
        "direction": args.direction,
        "threshold": args.threshold,
        "statistic": args.statistic,
        "unit": args.unit,
    }
    append_event(Path(args.output_dir).resolve() / "events.jsonl", event)
    return 0


def aggregate(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def stage_status(cases: list[dict[str, Any]]) -> str:
    required = [case for case in cases if case["level"] == "required"]
    optional = [case for case in cases if case["level"] == "optional"]
    if not cases:
        return "blocked"
    if any(case["status"] == "failed" for case in required):
        return "failed"
    if any(case["status"] == "blocked" for case in required):
        return "blocked"
    if not required or any(case["status"] == "skipped" for case in required):
        return "partial"
    if any(case["status"] in {"failed", "blocked", "skipped"} for case in optional):
        return "partial"
    return "passed"


def status_exit(status: str) -> int:
    return {"passed": 0, "failed": 1, "blocked": 64, "partial": 65}.get(status, 1)


def cmd_finalize(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    events = read_events(output / "events.jsonl")
    declarations = [e for e in events if e.get("type") == "case_declaration"]
    declaration_counts = Counter(str(e.get("case")) for e in declarations)
    duplicate_cases = sorted(name for name, count in declaration_counts.items() if count > 1)
    samples_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    marks_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expectations = [e for e in events if e.get("type") == "expectation"]
    for event in events:
        kind = event.get("type")
        case = str(event.get("case") or "")
        if kind == "sample":
            samples_by_case[case].append(event)
        elif kind == "case_failure":
            failures_by_case[case].append(event)
        elif kind == "case_mark":
            marks_by_case[case].append(event)

    cases: list[dict[str, Any]] = []
    for declaration in declarations:
        case = str(declaration["case"])
        level = str(declaration["level"])
        repeat = int(declaration["repeat"])
        samples = sorted(samples_by_case[case], key=lambda x: int(x.get("iteration") or 0))
        failures = failures_by_case[case]
        marks = marks_by_case[case]
        if marks:
            status = str(marks[-1]["status"])
            reason = marks[-1].get("reason")
        elif failures:
            status = "blocked" if all(f.get("status") == "blocked" for f in failures) and not samples else "failed"
            reason = failures[-1].get("reason")
        elif len(samples) < repeat:
            status = "failed"
            reason = f"有效样本不足: {len(samples)}/{repeat}"
        else:
            status = "passed"
            reason = None

        metric_values: dict[str, list[float]] = defaultdict(list)
        metric_meta: dict[str, dict[str, Any]] = {}
        for sample in samples:
            metrics = sample.get("metrics") if isinstance(sample.get("metrics"), dict) else {}
            for metric, item in metrics.items():
                if not isinstance(item, dict) or not isinstance(item.get("value"), (int, float)):
                    continue
                metric_values[str(metric)].append(float(item["value"]))
                metric_meta.setdefault(str(metric), {"unit": item.get("unit"), "direction": item.get("direction"), "direction_source": item.get("direction_source")})
        aggregates = {
            metric: {**aggregate(values), **metric_meta.get(metric, {})}
            for metric, values in sorted(metric_values.items())
        }
        cases.append(
            {
                "name": case,
                "level": level,
                "status": status,
                "warmup": int(declaration["warmup"]),
                "repeat": repeat,
                "valid_samples": len(samples),
                "failed_samples": len(failures),
                "reason": reason,
                "metrics": aggregates,
                "samples": samples,
                "failures": failures,
            }
        )

    status = "failed" if duplicate_cases else stage_status(cases)
    expectation_results: list[dict[str, Any]] = []
    for expectation in expectations:
        case = next((item for item in cases if item["name"] == expectation["case"]), None)
        metric = case.get("metrics", {}).get(expectation["metric"]) if case else None
        actual = metric.get(expectation["statistic"]) if isinstance(metric, dict) else None
        met = False
        if isinstance(actual, (int, float)):
            if expectation["direction"] == "higher":
                met = float(actual) >= float(expectation["threshold"])
            else:
                met = float(actual) <= float(expectation["threshold"])
        expectation_results.append({**expectation, "actual": actual, "met": met})
    target_met: bool | None = None if not expectation_results else all(item["met"] for item in expectation_results)

    context_path = output.parent / "context.json"
    context: dict[str, Any] = {}
    if context_path.is_file():
        try:
            loaded_context = json.loads(context_path.read_text(encoding="utf-8"))
            context = loaded_context if isinstance(loaded_context, dict) else {}
        except Exception:
            context = {}

    declarations_payload = [
        {"case": e["case"], "level": e["level"], "warmup": e["warmup"], "repeat": e["repeat"]}
        for e in declarations
    ]
    check_name = str(context.get("check") or "benchmark")
    runbook_hashes = context.get("runbook_hashes") if isinstance(context.get("runbook_hashes"), dict) else {}
    check_script = f"checks/{check_name}.sh"
    workload_payload = {
        "declarations": declarations_payload,
        "check": check_name,
        "check_script_hash": runbook_hashes.get(check_script),
    }
    workload_fingerprint = hashlib.sha256(json.dumps(workload_payload, sort_keys=True).encode()).hexdigest()[:20]

    runtime = context.get("runtime") if isinstance(context.get("runtime"), dict) else {}
    runtime_keys = (
        "MODEL_NAME", "MODEL_PATH", "MODEL_REVISION", "ENGINE", "HARDWARE",
        "RUNTIME_PYTHON", "TENSOR_PARALLEL_SIZE",
    )
    runtime_payload = {
        "runtime": {key: runtime.get(key) for key in runtime_keys if runtime.get(key) not in (None, "")},
        "runbook_hashes": {
            key: value
            for key, value in runbook_hashes.items()
            if key in {"env.sh", "start.sh"}
        },
    }
    execution_fingerprint = hashlib.sha256(json.dumps(runtime_payload, sort_keys=True).encode()).hexdigest()[:20]

    summary = {
        "schema_version": 1,
        "kind": "benchmark",
        "status": status,
        "target_met": target_met,
        "created_at": now(),
        "workload_fingerprint": workload_fingerprint,
        "execution_fingerprint": execution_fingerprint,
        "workload_definition": workload_payload,
        "execution_definition": runtime_payload,
        "duplicate_cases": duplicate_cases,
        "case_count": len(cases),
        "counts": dict(sorted(Counter(case["status"] for case in cases).items())),
        "cases": cases,
        "expectations": expectation_results,
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return status_exit(status)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="记录并汇总 benchmark 样本")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--output-dir", required=True)
    init.add_argument("--force", action="store_true")

    declare = sub.add_parser("declare")
    declare.add_argument("--output-dir", required=True)
    declare.add_argument("--case", required=True)
    declare.add_argument("--level", required=True)
    declare.add_argument("--warmup", type=int, required=True)
    declare.add_argument("--repeat", type=int, required=True)

    sample = sub.add_parser("sample")
    sample.add_argument("--output-dir", required=True)
    sample.add_argument("--case", required=True)
    sample.add_argument("--level", required=True)
    sample.add_argument("--iteration", type=int, required=True)
    sample.add_argument("--sample-file", required=True)
    sample.add_argument("--duration-seconds", type=float)
    sample.add_argument("--stdout")
    sample.add_argument("--stderr")

    failure = sub.add_parser("failure")
    failure.add_argument("--output-dir", required=True)
    failure.add_argument("--case", required=True)
    failure.add_argument("--level", required=True)
    failure.add_argument("--iteration", type=int)
    failure.add_argument("--phase", required=True)
    failure.add_argument("--status", required=True)
    failure.add_argument("--exit-code", type=int)
    failure.add_argument("--duration-seconds", type=float)
    failure.add_argument("--stdout")
    failure.add_argument("--stderr")
    failure.add_argument("--reason")

    mark = sub.add_parser("mark")
    mark.add_argument("--output-dir", required=True)
    mark.add_argument("--case", required=True)
    mark.add_argument("--level", required=True)
    mark.add_argument("--status", required=True)
    mark.add_argument("--reason")

    expect = sub.add_parser("expect")
    expect.add_argument("--output-dir", required=True)
    expect.add_argument("--case", required=True)
    expect.add_argument("--metric", required=True)
    expect.add_argument("--direction", required=True)
    expect.add_argument("--threshold", type=float, required=True)
    expect.add_argument("--statistic", default="median")
    expect.add_argument("--unit")

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return {
            "init": cmd_init,
            "declare": cmd_declare,
            "sample": cmd_sample,
            "failure": cmd_failure,
            "mark": cmd_mark,
            "expect": cmd_expect,
            "finalize": cmd_finalize,
        }[args.command](args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
