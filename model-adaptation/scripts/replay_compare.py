#!/usr/bin/env python3
"""Replay and compare captured samples without changing Migration Spec state."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from _lib.spec_contract import (
    SpecContractError,
    canonical_json_bytes,
    load_spec_binding,
)


class ToolError(RuntimeError):
    """The deterministic action could not form a trustworthy result."""


DEFAULT_SYNTHETIC_CASE = {
    "expected": {"spec_binding_smoke": "ready"},
    "actual": {"spec_binding_smoke": "ready"},
}


def read_synthetic_case(case_path: Optional[Path]) -> Dict[str, Any]:
    if case_path is None:
        return DEFAULT_SYNTHETIC_CASE
    try:
        case = json.loads(case_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ToolError(f"cannot read synthetic case: {error}") from error
    except json.JSONDecodeError as error:
        raise ToolError(f"synthetic case is not valid JSON: {error.msg}") from error
    if not isinstance(case, dict) or set(case) != {"expected", "actual"}:
        raise ToolError("synthetic case must contain exactly expected and actual")
    return case


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def create_run_dir(run_dir: Path) -> None:
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise ToolError(f"run directory must be fresh and creatable: {error}") from error


def run_synthetic(
    spec_path: Path, run_dir: Path, case_path: Optional[Path] = None
) -> None:
    binding = load_spec_binding(spec_path)
    case = read_synthetic_case(case_path)
    passed = canonical_json_bytes(case["expected"]) == canonical_json_bytes(case["actual"])

    create_run_dir(run_dir)

    write_json(run_dir / "synthetic-case.json", case)
    (run_dir / "replay.log").write_text(
        f"action=synthetic\npassed={str(passed).lower()}\n",
        encoding="utf-8",
    )
    result = {
        "tool": "replay_compare.py",
        "action": "synthetic",
        "spec_binding": binding.as_result_dict(),
        "passed": passed,
        "summary": (
            "Synthetic expected and actual values match."
            if passed
            else "Synthetic expected and actual values differ."
        ),
        "evidence": ["synthetic-case.json", "replay.log"],
    }
    write_json(run_dir / "result.json", result)


def run_validate_binding(spec_path: Path, run_dir: Path, result_path: Path) -> None:
    binding = load_spec_binding(spec_path)
    try:
        candidate_result = json.loads(result_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ToolError(f"cannot read result to validate: {error}") from error
    except json.JSONDecodeError as error:
        raise ToolError(f"result to validate is not valid JSON: {error.msg}") from error
    if not isinstance(candidate_result, dict):
        raise ToolError("result to validate must be one JSON object")

    expected_binding = binding.as_result_dict()
    mismatched_fields = binding.mismatched_fields(candidate_result.get("spec_binding"))
    passed = not mismatched_fields

    create_run_dir(run_dir)
    write_json(run_dir / "candidate-result.json", candidate_result)
    (run_dir / "replay.log").write_text(
        "\n".join(
            [
                "action=validate-binding",
                f"passed={str(passed).lower()}",
                "mismatched_fields=" + ",".join(mismatched_fields),
                "",
            ]
        ),
        encoding="utf-8",
    )
    result = {
        "tool": "replay_compare.py",
        "action": "validate-binding",
        "spec_binding": expected_binding,
        "passed": passed,
        "summary": (
            "Candidate result matches the current Contract Data."
            if passed
            else "Candidate result does not match the current Contract Data."
        ),
        "mismatched_fields": mismatched_fields,
        "evidence": ["candidate-result.json", "replay.log"],
    }
    write_json(run_dir / "result.json", result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--mode", required=True, choices=("synthetic", "validate-binding")
    )
    parser.add_argument("--case", type=Path)
    parser.add_argument("--result", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "synthetic":
            if args.result is not None:
                raise ToolError("--result is not valid for synthetic mode")
            run_synthetic(args.spec, args.run_dir, args.case)
        else:
            if args.result is None:
                raise ToolError("--result is required for validate-binding mode")
            if args.case is not None:
                raise ToolError("--case is not valid for validate-binding mode")
            run_validate_binding(args.spec, args.run_dir, args.result)
    except (SpecContractError, ToolError, OSError) as error:
        print(f"replay_compare.py: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
