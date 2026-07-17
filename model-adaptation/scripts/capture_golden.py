#!/usr/bin/env python3
"""Prepare and validate bounded CUDA capture for one Semantic Operator."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

from _lib.spec_contract import (
    SpecContractError,
    load_contract_data,
    load_spec_binding,
)
from model_adaptation_capture.contracts import (
    CANDIDATE_SERIALIZATION,
    CAPTURE_CONFIG_ENV,
    CONFIG_SCHEMA,
    HELPER_RELATIVE_PATH as HELPER_PATH_TEXT,
)
from model_adaptation_capture.preflight import PreflightError, run_workers


HELPER_RELATIVE_PATH = Path(HELPER_PATH_TEXT)


class ToolError(RuntimeError):
    """The capture action could not form a trustworthy result."""


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def create_run_dir(run_dir: Path) -> None:
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "samples").mkdir()
    except OSError as error:
        raise ToolError(f"run directory must be fresh and creatable: {error}") from error


def read_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ToolError(f"cannot read {label}: {error}") from error
    except json.JSONDecodeError as error:
        raise ToolError(f"{label} is not valid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ToolError(f"{label} must be one JSON object")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ToolError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def resolve_git_revision(worktree: Path) -> str:
    if not (worktree / HELPER_RELATIVE_PATH).is_file():
        raise ToolError(
            f"SGLang worktree is missing {HELPER_RELATIVE_PATH.as_posix()}"
        )
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ToolError(f"cannot resolve SGLang revision: {completed.stderr.strip()}")
    return completed.stdout.strip()


def find_operator(scan_result: Dict[str, Any], operator_id: str) -> Dict[str, Any]:
    operators = scan_result.get("operators")
    if not isinstance(operators, list):
        raise ToolError("Scan Run operators must be a list")
    matches = [item for item in operators if item.get("operator_id") == operator_id]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise ToolError(f"Scan Run must contain exactly one operator {operator_id!r}")
    return matches[0]


def find_capture_plan(
    scan_result: Dict[str, Any],
    operator_id: str,
) -> Dict[str, Any]:
    capture_plan = scan_result.get("capture_plan")
    if not isinstance(capture_plan, list):
        raise ToolError("Scan Run capture_plan must be a list")
    matches = [item for item in capture_plan if item.get("operator_id") == operator_id]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise ToolError(
            f"Scan Run capture_plan must contain exactly one entry for {operator_id!r}"
        )
    return matches[0]


def validate_scan_candidate(
    contract: Dict[str, Any],
    scan_result: Dict[str, Any],
    operator_id: str,
    expected_binding: Dict[str, Any],
) -> Dict[str, Any]:
    if scan_result.get("scan_complete") is not True:
        raise ToolError("Scan Run is not complete")
    if scan_result.get("spec_binding") != expected_binding:
        raise ToolError("Scan Run spec_binding does not match Contract Data")

    source = scan_result.get("source")
    if not isinstance(source, dict):
        raise ToolError("Scan Run source must be an object")
    expected_source = contract["source"]
    expected_checkpoint = contract["checkpoint"]
    checks = {
        "sglang_revision": expected_source["sglang_revision"],
        "sglang_kunlun_revision": expected_source["sglang_kunlun_revision"],
        "checkpoint_id": expected_checkpoint["id"],
        "checkpoint_config_sha256": expected_checkpoint["config_digest"],
    }
    for field, expected in checks.items():
        if source.get(field) != expected:
            raise ToolError(f"Scan Run source {field} does not match Contract Data")

    operator = find_operator(scan_result, operator_id)
    if operator.get("verdict") != "CAPTURE_REQUIRED":
        raise ToolError(f"operator {operator_id!r} is not CAPTURE_REQUIRED")
    if operator.get("model_path") != ["target"]:
        raise ToolError(f"operator {operator_id!r} must be target-only for this Demo")
    if operator.get("state_dependency") != "scalar limit only":
        raise ToolError(f"operator {operator_id!r} is not weight-free")
    for field in ("boundary", "cuda_impl", "kunlun_impl"):
        if not operator.get(field):
            raise ToolError(f"operator {operator_id!r} is missing {field}")

    plan = find_capture_plan(scan_result, operator_id)
    max_samples = contract["limits"]["max_samples_per_operator"]
    if plan.get("max_distinct_shapes") != max_samples:
        raise ToolError("capture plan sample limit does not match Contract Data")
    return operator


def prepare_capture_config(
    spec_path: Path,
    run_dir: Path,
    scan_result_path: Path,
    operator_id: str,
    sglang_worktree: Path,
) -> tuple[Any, Dict[str, Any]]:
    binding = load_spec_binding(spec_path)
    contract = load_contract_data(spec_path)
    scan_result = read_json_object(scan_result_path, "Scan Run result")
    operator = validate_scan_candidate(
        contract,
        scan_result,
        operator_id,
        binding.as_result_dict(),
    )

    actual_revision = resolve_git_revision(sglang_worktree)
    expected_revision = contract["source"]["sglang_revision"]
    if actual_revision != expected_revision:
        raise ToolError(
            "SGLang worktree revision does not match Contract Data: "
            f"expected {expected_revision}, got {actual_revision}"
        )

    create_run_dir(run_dir)
    config_path = run_dir / "capture-config.json"
    config = {
        "schema": CONFIG_SCHEMA,
        "spec_binding": binding.as_result_dict(),
        "operator_id": operator_id,
        "model_path": "target",
        "max_samples": contract["limits"]["max_samples_per_operator"],
        "serialization": CANDIDATE_SERIALIZATION,
        "capture_device_type": "cuda",
        "dtype": contract["runtime"]["dtype"],
        "precision_gate": contract["precision_gate"],
        "run_dir": str(run_dir.resolve()),
        "scan_result": {
            "path": str(scan_result_path.resolve()),
            "sha256": file_sha256(scan_result_path),
        },
        "source": {
            "sglang_revision": actual_revision,
            "sglang_worktree": str(sglang_worktree.resolve()),
            "helper_path": str((sglang_worktree / HELPER_RELATIVE_PATH).resolve()),
        },
        "boundary": operator["boundary"],
        "state_dependency": operator["state_dependency"],
    }
    write_json(config_path, config)
    return binding, config


def run_prepare(
    spec_path: Path,
    run_dir: Path,
    scan_result_path: Path,
    operator_id: str,
    sglang_worktree: Path,
) -> None:
    binding, config = prepare_capture_config(
        spec_path,
        run_dir,
        scan_result_path,
        operator_id,
        sglang_worktree,
    )
    config_path = run_dir / "capture-config.json"
    result = {
        "tool": "capture_golden.py",
        "action": "prepare",
        "spec_binding": binding.as_result_dict(),
        "passed": True,
        "capture_status": "PREPARED",
        "consumes_capture_session": False,
        "operator_id": operator_id,
        "launch_environment": {
            CAPTURE_CONFIG_ENV: str(config_path.resolve()),
        },
        "evidence": ["capture-config.json", "capture.log"],
        "summary": "Scan candidate and fixed SGLang revision are ready for CUDA preflight.",
    }
    write_json(run_dir / "result.json", result)
    (run_dir / "capture.log").write_text(
        "\n".join(
            [
                "action=prepare",
                f"operator_id={operator_id}",
                f"sglang_revision={config['source']['sglang_revision']}",
                f"max_samples={config['max_samples']}",
                "capture_session_consumed=false",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_preflight(
    spec_path: Path,
    run_dir: Path,
    scan_result_path: Path,
    operator_id: str,
    sglang_worktree: Path,
) -> None:
    binding, config = prepare_capture_config(
        spec_path,
        run_dir,
        scan_result_path,
        operator_id,
        sglang_worktree,
    )
    passed, evidence = run_workers(
        run_dir / "capture-config.json",
        sglang_worktree,
    )
    result = {
        "tool": "capture_golden.py",
        "action": "preflight",
        "spec_binding": binding.as_result_dict(),
        "passed": passed,
        "capture_status": "PREFLIGHT_PASSED" if passed else "PREFLIGHT_FAILED",
        "consumes_capture_session": False,
        "operator_id": operator_id,
        "serialization": config["serialization"],
        "evidence": evidence,
        "summary": (
            "The installed SGLang hook captured three CUDA BF16 candidates and a new process replayed them."
            if passed
            else "CUDA preflight did not prove capture and new-process replay; inspect the preflight logs."
        ),
    }
    write_json(run_dir / "result.json", result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("prepare", "preflight"))
    parser.add_argument("--scan-result", required=True, type=Path)
    parser.add_argument("--operator-id", required=True)
    parser.add_argument("--sglang-worktree", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        action = run_prepare if args.mode == "prepare" else run_preflight
        action(
            args.spec,
            args.run_dir,
            args.scan_result,
            args.operator_id,
            args.sglang_worktree,
        )
    except (SpecContractError, PreflightError, ToolError, OSError) as error:
        print(f"capture_golden.py: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
