#!/usr/bin/env python3
"""Prepare and validate bounded CUDA capture for one selected Kernel Call."""

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
    uses_kernel_scan_contract,
)
from model_adaptation_capture.contracts import (
    ACTIVATION_GUARD,
    CANDIDATE_SERIALIZATION,
    CAPTURE_CONFIG_ENV,
    CONFIG_SCHEMA,
    KERNEL_CALL_SERIALIZATION,
    KUNLUN_SWIGLU_TARGET,
    MLP_HOOK_TARGET,
    MODEL_RELATIVE_PATH as MODEL_PATH_TEXT,
    SWIGLU_CLAMP_OPERATOR_ID,
    SWIGLU_CLAMP_SOURCE_RELATIVE_PATH,
    checkpoint_metadata,
)
from model_adaptation_capture.preflight import (
    PreflightError,
    run_capture_preflight,
)


MODEL_RELATIVE_PATH = Path(MODEL_PATH_TEXT)


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
    if not (worktree / MODEL_RELATIVE_PATH).is_file():
        raise ToolError(
            f"SGLang worktree is missing {MODEL_RELATIVE_PATH.as_posix()}"
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
    for field in ("boundary", "cuda_impl", "kunlun_impl"):
        if not operator.get(field):
            raise ToolError(f"operator {operator_id!r} is missing {field}")

    plan = find_capture_plan(scan_result, operator_id)
    max_shapes = contract["limits"]["max_shapes_per_operator"]
    if plan.get("max_distinct_shapes") != max_shapes:
        raise ToolError("capture plan shape limit does not match Contract Data")
    if uses_kernel_scan_contract(contract):
        if scan_result.get("scan_scope") != contract["scan_scope"]:
            raise ToolError("Scan Run scope does not match Contract Data")
        selection = scan_result.get("selection")
        capture_plan = scan_result.get("capture_plan")
        first_planned_operator = (
            capture_plan[0].get("operator_id")
            if isinstance(capture_plan, list)
            and capture_plan
            and isinstance(capture_plan[0], dict)
            else None
        )
        selected_operator = (
            selection.get("active_operator")
            if isinstance(selection, dict)
            else None
        )
        if (
            not isinstance(selection, dict)
            or selection.get("evaluated_after_scan") is not True
            or selected_operator != first_planned_operator
        ):
            raise ToolError(
                "post-scan active_operator must be the first capture plan entry"
            )
        if contract["contract_revision"] <= 5 and selected_operator != operator_id:
            raise ToolError("requested operator is not the post-scan active_operator")
        sample_policy = contract["sample_policy"]
        if plan.get("tp_rank") != sample_policy["capture_tp_rank"]:
            raise ToolError("capture plan TP rank does not match Contract Data")
        if plan.get("replay_mode") != "standalone-kernel-call":
            raise ToolError("capture plan must replay the standalone kernel call")
        policy_checks = {
            "save_direct_parameter_tensors": (
                sample_policy["save_direct_parameter_tensors"]
            ),
            "save_full_checkpoint": sample_policy["save_full_checkpoint"],
            "save_module_state_dict": sample_policy["save_module_state_dict"],
        }
        for field, expected in policy_checks.items():
            if plan.get(field) is not expected:
                raise ToolError(
                    f"capture plan {field} does not match Contract Data"
                )
        boundary = operator["boundary"]
        saved_boundary_fields = {
            "saved_inputs": ("inputs", "inputs"),
            "saved_parameters": ("parameters", "direct call parameters"),
            "saved_non_tensor_args": ("non_tensor_args", "non-Tensor arguments"),
            "saved_outputs": ("outputs", "outputs"),
        }
        for plan_field, (boundary_field, label) in saved_boundary_fields.items():
            expected_names = boundary.get(boundary_field)
            if not isinstance(expected_names, list) or not all(
                isinstance(name, str) and name for name in expected_names
            ):
                raise ToolError(
                    f"operator boundary {boundary_field} must be a list of names"
                )
            if plan.get(plan_field) != expected_names:
                raise ToolError(
                    f"capture plan {plan_field.replace('_', ' ')} do not match "
                    f"the {label}"
                )
        kernel_call = operator.get("kernel_call")
        if (
            not isinstance(kernel_call, dict)
            or plan.get("hook_target") != kernel_call.get("capture_seam")
        ):
            raise ToolError("capture plan hook target has drifted")
    else:
        expected_boundary = contract["operator_boundary"]
        if operator_id != expected_boundary["id"]:
            raise ToolError("requested operator does not match Contract Data")
        if operator.get("activation_guard") != expected_boundary["activation_guard"]:
            raise ToolError(f"operator {operator_id!r} activation guard has drifted")
        if operator.get("hook_target") != MLP_HOOK_TARGET:
            raise ToolError(f"operator {operator_id!r} hook target has drifted")
        if (
            operator.get("state_dependency")
            != "same checkpoint, TP8, current-rank model weights"
        ):
            raise ToolError(
                f"operator {operator_id!r} must replay with loaded checkpoint weights"
            )
        if plan.get("tp_rank") != expected_boundary["tp_rank"]:
            raise ToolError("capture plan TP rank does not match Contract Data")
        if plan.get("replay_mode") != "loaded_model":
            raise ToolError("capture plan must replay inside a loaded model")
        if plan.get("weights_in_golden_sample") is not False:
            raise ToolError(
                "capture plan must keep checkpoint weights out of Golden Samples"
            )
    return operator


def prepare_capture_config(
    spec_path: Path,
    run_dir: Path,
    scan_result_path: Path,
    operator_id: str,
    sglang_worktree: Path,
    *,
    preflight: bool = False,
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
    kernel_scan_contract = uses_kernel_scan_contract(contract)
    if kernel_scan_contract and operator_id != SWIGLU_CLAMP_OPERATOR_ID:
        raise ToolError(f"capture/replay adapter is not implemented for {operator_id!r}")
    capture_plan = (
        find_capture_plan(scan_result, operator_id)
        if kernel_scan_contract
        else None
    )

    actual_revision = resolve_git_revision(sglang_worktree)
    expected_revision = contract["source"]["sglang_revision"]
    if actual_revision != expected_revision:
        raise ToolError(
            "SGLang worktree revision does not match Contract Data: "
            f"expected {expected_revision}, got {actual_revision}"
        )
    kernel_source_path = (
        (sglang_worktree / SWIGLU_CLAMP_SOURCE_RELATIVE_PATH).resolve()
        if kernel_scan_contract
        else None
    )
    if kernel_source_path is not None and not kernel_source_path.is_file():
        raise ToolError(
            "SGLang worktree is missing selected Kernel Call source file "
            f"{SWIGLU_CLAMP_SOURCE_RELATIVE_PATH}"
        )

    try:
        checkpoint = checkpoint_metadata(
            contract["checkpoint"]["id"],
            contract["checkpoint"]["config_digest"],
        )
    except ValueError as error:
        raise ToolError(str(error)) from error

    create_run_dir(run_dir)
    config_path = run_dir / "capture-config.json"
    config = {
        "schema": CONFIG_SCHEMA,
        "spec_binding": binding.as_result_dict(),
        "operator_id": operator_id,
        "activation_guard": (
            operator["activation_guard"]
            if kernel_scan_contract
            else contract["operator_boundary"]["activation_guard"]
        ),
        "model_path": "target",
        "max_shapes": contract["limits"]["max_shapes_per_operator"],
        "tensor_parallel_size": contract["runtime"]["tensor_parallel_size"],
        "tp_rank": (
            contract["sample_policy"]["capture_tp_rank"]
            if kernel_scan_contract
            else contract["operator_boundary"]["tp_rank"]
        ),
        "serialization": (
            KERNEL_CALL_SERIALIZATION
            if kernel_scan_contract
            else CANDIDATE_SERIALIZATION
        ),
        "capture_device_type": "cuda",
        "dtype": contract["runtime"]["dtype"],
        "checkpoint": checkpoint,
        "precision_gate": contract["precision_gate"],
        "run_dir": str(run_dir.resolve()),
        "scan_result": {
            "path": str(scan_result_path.resolve()),
            "sha256": file_sha256(scan_result_path),
        },
        "source": {
            "sglang_revision": actual_revision,
            "sglang_worktree": str(sglang_worktree.resolve()),
            "model_path": str((sglang_worktree / MODEL_RELATIVE_PATH).resolve()),
        },
        "boundary": operator["boundary"],
    }
    if kernel_scan_contract:
        if capture_plan is None:
            raise ToolError("selected kernel is missing its capture plan")
        config.update(
            {
                "hook_target": capture_plan["hook_target"],
                "sample_fields": {
                    "inputs": capture_plan["saved_inputs"],
                    "parameters": capture_plan["saved_parameters"],
                    "non_tensor_args": capture_plan["saved_non_tensor_args"],
                    "outputs": capture_plan["saved_outputs"],
                },
                "replay": {
                    "mode": "standalone-kernel-call",
                    "cuda_target": SWIGLU_CLAMP_OPERATOR_ID,
                    "p800_target": KUNLUN_SWIGLU_TARGET,
                    "weights_in_golden_sample": False,
                },
            }
        )
        config["source"]["capture_module_path"] = str(kernel_source_path)
    else:
        config.update(
            {
                "state_dependency": operator["state_dependency"],
                "replay": {
                    "mode": "loaded_model",
                    "checkpoint_id": checkpoint["id"],
                    "weights_in_golden_sample": False,
                },
            }
        )
    if preflight:
        config["preflight_tp_context"] = {
            "rank": config["tp_rank"],
            "size": config["tensor_parallel_size"],
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
        "summary": (
            "The selected existing Kernel Call and fixed source are ready for "
            "rank-0 CUDA capture in the TP8 model."
            if config["replay"]["mode"] == "standalone-kernel-call"
            else
            "The original MLP boundary and fixed source are ready for rank-0 "
            "CUDA capture in the TP8 model."
        ),
    }
    write_json(run_dir / "result.json", result)
    (run_dir / "capture.log").write_text(
        "\n".join(
            [
                "action=prepare",
                f"operator_id={operator_id}",
                f"sglang_revision={config['source']['sglang_revision']}",
                f"max_shapes={config['max_shapes']}",
                f"tensor_parallel_size={config['tensor_parallel_size']}",
                f"tp_rank={config['tp_rank']}",
                f"replay_mode={config['replay']['mode']}",
                f"hook_target={config.get('hook_target', MLP_HOOK_TARGET)}",
                "weights_in_golden_sample=false",
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
        preflight=True,
    )
    passed, evidence = run_capture_preflight(
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
            "The installed SGLang hook captured rank-0 Kernel Call records and "
            "a new process self-replayed them through the existing CUDA call; "
            "the real checkpoint capture remains a later step."
            if passed and config["replay"]["mode"] == "standalone-kernel-call"
            else
            "The installed SGLang hook captured rank-0 MLP boundary records; "
            "real checkpoint capture and loaded-model replay remain real-model steps."
            if passed
            else
            "CUDA preflight did not prove the selected existing hook, rank-0 "
            "format, and self-replay; inspect the preflight logs."
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
