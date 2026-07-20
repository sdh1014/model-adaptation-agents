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
    ATTENTION_CAPTURE_SEAM,
    ATTENTION_OPERATOR_ID,
    ATTENTION_SOURCE_RELATIVE_PATH,
    CANDIDATE_SERIALIZATION,
    CAPTURE_CONFIG_ENV,
    CONFIG_SCHEMA,
    GEMMA_FUSED_ADD_RMSNORM_CAPTURE_SEAM,
    GEMMA_FUSED_ADD_RMSNORM_OPERATOR_ID,
    GEMMA_RMSNORM_CAPTURE_SEAM,
    GEMMA_RMSNORM_OPERATOR_ID,
    KERNEL_CALL_SERIALIZATION,
    KUNLUN_SWIGLU_TARGET,
    LAYERNORM_SOURCE_RELATIVE_PATH,
    MLP_HOOK_TARGET,
    MODEL_RELATIVE_PATH as MODEL_PATH_TEXT,
    SESSION_CONFIG_SCHEMA,
    SWIGLU_CLAMP_OPERATOR_ID,
    SWIGLU_CLAMP_SOURCE_RELATIVE_PATH,
    TOPK_SIGMOID_CAPTURE_SEAM,
    TOPK_SIGMOID_OPERATOR_ID,
    TOPK_SOURCE_RELATIVE_PATH,
    checkpoint_metadata,
)
from model_adaptation_capture.preflight import (
    PreflightError,
    run_capture_preflight,
)


MODEL_RELATIVE_PATH = Path(MODEL_PATH_TEXT)
FIXED_IMAGE_RELATIVE_PATH = Path("examples/assets/example_image.png")
SUPPORTED_SOURCE_PATHS = {
    SWIGLU_CLAMP_OPERATOR_ID: Path(SWIGLU_CLAMP_SOURCE_RELATIVE_PATH),
    GEMMA_RMSNORM_OPERATOR_ID: Path(LAYERNORM_SOURCE_RELATIVE_PATH),
    GEMMA_FUSED_ADD_RMSNORM_OPERATOR_ID: Path(LAYERNORM_SOURCE_RELATIVE_PATH),
    TOPK_SIGMOID_OPERATOR_ID: Path(TOPK_SOURCE_RELATIVE_PATH),
    ATTENTION_OPERATOR_ID: Path(ATTENTION_SOURCE_RELATIVE_PATH),
}
SUPPORTED_HOOK_TARGETS = {
    SWIGLU_CLAMP_OPERATOR_ID: SWIGLU_CLAMP_OPERATOR_ID,
    GEMMA_RMSNORM_OPERATOR_ID: GEMMA_RMSNORM_CAPTURE_SEAM,
    GEMMA_FUSED_ADD_RMSNORM_OPERATOR_ID: (
        GEMMA_FUSED_ADD_RMSNORM_CAPTURE_SEAM
    ),
    TOPK_SIGMOID_OPERATOR_ID: TOPK_SIGMOID_CAPTURE_SEAM,
    ATTENTION_OPERATOR_ID: ATTENTION_CAPTURE_SEAM,
}
ADAPTER_NAMES = {
    SWIGLU_CLAMP_OPERATOR_ID: "swiglu-clamp",
    GEMMA_RMSNORM_OPERATOR_ID: "gemma-rmsnorm",
    GEMMA_FUSED_ADD_RMSNORM_OPERATOR_ID: "gemma-fused-add-rmsnorm",
    TOPK_SIGMOID_OPERATOR_ID: "topk-sigmoid",
    ATTENTION_OPERATOR_ID: "vision-prefill-attention",
}


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


def validate_capture_session_plan(
    contract: Dict[str, Any],
    scan_result: Dict[str, Any],
    expected_binding: Dict[str, Any],
) -> list[Dict[str, Any]]:
    """Validate the complete revision-6 queue before one CUDA model launch."""
    if contract.get("contract_revision", 0) < 6:
        raise ToolError("multi-operator capture requires Contract revision 6 or later")
    operators = scan_result.get("operators")
    capture_plan = scan_result.get("capture_plan")
    gap_queue = scan_result.get("gap_queue")
    if not isinstance(operators, list):
        raise ToolError("Scan Run operators must be a list")
    if not isinstance(capture_plan, list) or not capture_plan:
        raise ToolError("Scan Run capture_plan must be a non-empty list")
    if not isinstance(gap_queue, list):
        raise ToolError("Scan Run gap_queue must be a list")

    required_ids = [
        item.get("operator_id")
        for item in operators
        if isinstance(item, dict) and item.get("verdict") == "CAPTURE_REQUIRED"
    ]
    queue_ids = [
        item.get("operator_id")
        for item in gap_queue
        if isinstance(item, dict)
    ]
    planned_ids = [
        item.get("operator_id")
        for item in capture_plan
        if isinstance(item, dict)
    ]
    if (
        not all(isinstance(item, str) and item for item in required_ids)
        or len(required_ids) != len(set(required_ids))
        or queue_ids != planned_ids
        or set(planned_ids) != set(required_ids)
    ):
        raise ToolError(
            "revision-6 capture_plan must contain every CAPTURE_REQUIRED operator "
            "once and in gap_queue order"
        )
    if ATTENTION_OPERATOR_ID not in planned_ids:
        raise ToolError(
            "revision-6 capture_plan must include the Step-3.7 single-image "
            "vision attention gap"
        )

    validated = []
    for operator_id in planned_ids:
        if operator_id not in SUPPORTED_HOOK_TARGETS:
            raise ToolError(
                f"capture/replay adapter is not implemented for {operator_id!r}"
            )
        operator = validate_scan_candidate(
            contract,
            scan_result,
            operator_id,
            expected_binding,
        )
        plan = find_capture_plan(scan_result, operator_id)
        expected_hook = SUPPORTED_HOOK_TARGETS[operator_id]
        if plan.get("hook_target") != expected_hook:
            raise ToolError(
                f"capture plan hook target for {operator_id!r} must remain "
                f"the existing call {expected_hook!r}"
            )
        validated.append(operator)

    requests = scan_result.get("request_set")
    if not isinstance(requests, list):
        raise ToolError("Scan Run request_set must be a list")
    request_modes = [
        item.get("input_mode") for item in requests if isinstance(item, dict)
    ]
    expected_modes = contract["scan_scope"]["input_modes"]
    if request_modes != expected_modes:
        raise ToolError(
            "Scan Run request_set must cover Contract input modes in order"
        )
    for item in requests:
        request = item.get("request")
        if not isinstance(request, dict):
            raise ToolError("each capture request must be one object")
        sampling = request.get("sampling_params")
        if (
            not isinstance(request.get("text"), str)
            or not request["text"]
            or not isinstance(sampling, dict)
            or sampling.get("temperature") != 0
            or sampling.get("max_new_tokens") != 1
        ):
            raise ToolError(
                "capture requests must use fixed text and deterministic one-token generation"
            )
        if item["input_mode"] == "single-image":
            if request.get("image_data") != FIXED_IMAGE_RELATIVE_PATH.as_posix():
                raise ToolError(
                    "single-image request must use the fixed SGLang example image"
                )
            digest = item.get("image_sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ToolError("single-image request must bind the image SHA-256")
            if "<im_patch>" not in request["text"]:
                raise ToolError(
                    "single-image request text must contain the Step-3.7 image token"
                )
        elif "image_data" in request or "image_sha256" in item:
            raise ToolError("text-only request must not contain image data")
    return validated


def prepare_capture_session_config(
    spec_path: Path,
    run_dir: Path,
    scan_result_path: Path,
    sglang_worktree: Path,
    *,
    preflight: bool = False,
) -> tuple[Any, Dict[str, Any]]:
    """Prepare one plugin config containing the complete operator queue."""
    binding = load_spec_binding(spec_path)
    contract = load_contract_data(spec_path)
    scan_result = read_json_object(scan_result_path, "Scan Run result")
    operators = validate_capture_session_plan(
        contract,
        scan_result,
        binding.as_result_dict(),
    )

    actual_revision = resolve_git_revision(sglang_worktree)
    expected_revision = contract["source"]["sglang_revision"]
    if actual_revision != expected_revision:
        raise ToolError(
            "SGLang worktree revision does not match Contract Data: "
            f"expected {expected_revision}, got {actual_revision}"
        )
    for operator in operators:
        relative_path = SUPPORTED_SOURCE_PATHS[operator["operator_id"]]
        if not (sglang_worktree / relative_path).is_file():
            raise ToolError(
                "SGLang worktree is missing capture source file "
                f"{relative_path.as_posix()}"
            )

    image_request = next(
        item
        for item in scan_result["request_set"]
        if item["input_mode"] == "single-image"
    )
    image_path = (sglang_worktree / FIXED_IMAGE_RELATIVE_PATH).resolve()
    if (
        not image_path.is_relative_to(sglang_worktree.resolve())
        or image_path.is_symlink()
        or not image_path.is_file()
    ):
        raise ToolError("fixed single-image input is missing or escapes the worktree")
    if file_sha256(image_path) != image_request["image_sha256"]:
        raise ToolError("fixed single-image input bytes do not match the Scan Run")

    try:
        checkpoint = checkpoint_metadata(
            contract["checkpoint"]["id"],
            contract["checkpoint"]["config_digest"],
        )
    except ValueError as error:
        raise ToolError(str(error)) from error

    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        operators_dir = run_dir / "operators"
        operators_dir.mkdir()
    except OSError as error:
        raise ToolError(f"run directory must be fresh and creatable: {error}") from error

    config_path = run_dir / "capture-config.json"
    session_config_path = str(config_path.resolve())
    operator_configs = []
    for index, operator in enumerate(operators, start=1):
        operator_id = operator["operator_id"]
        plan = find_capture_plan(scan_result, operator_id)
        operator_run_dir = operators_dir / f"{index:02d}"
        operator_run_dir.mkdir()
        (operator_run_dir / "samples").mkdir()
        source_path = (
            sglang_worktree / SUPPORTED_SOURCE_PATHS[operator_id]
        ).resolve()
        p800_target = (
            KUNLUN_SWIGLU_TARGET
            if operator_id == SWIGLU_CLAMP_OPERATOR_ID
            else None
        )
        operator_configs.append(
            {
                "schema": CONFIG_SCHEMA,
                "session_config": session_config_path,
                "adapter": ADAPTER_NAMES[operator_id],
                "spec_binding": binding.as_result_dict(),
                "operator_id": operator_id,
                "activation_guard": operator["activation_guard"],
                "model_path": "target",
                "max_shapes": contract["limits"]["max_shapes_per_operator"],
                "tensor_parallel_size": contract["runtime"][
                    "tensor_parallel_size"
                ],
                "tp_rank": contract["sample_policy"]["capture_tp_rank"],
                "serialization": KERNEL_CALL_SERIALIZATION,
                "capture_device_type": "cuda",
                "dtype": contract["runtime"]["dtype"],
                "checkpoint": checkpoint,
                "precision_gate": contract["precision_gate"],
                "run_dir": str(operator_run_dir.resolve()),
                "hook_target": plan["hook_target"],
                "boundary": operator["boundary"],
                "sample_fields": {
                    "inputs": plan["saved_inputs"],
                    "parameters": plan["saved_parameters"],
                    "non_tensor_args": plan["saved_non_tensor_args"],
                    "outputs": plan["saved_outputs"],
                },
                "replay": {
                    "mode": "standalone-kernel-call",
                    "cuda_target": plan["hook_target"],
                    "p800_target": p800_target,
                    "weights_in_golden_sample": False,
                },
                "source": {
                    "capture_module_path": str(source_path),
                    "capture_module_sha256": file_sha256(source_path),
                },
            }
        )

    requests = json.loads(json.dumps(scan_result["request_set"]))
    for item in requests:
        if item["input_mode"] == "single-image":
            item["request"]["image_data"] = str(image_path)
    config = {
        "schema": SESSION_CONFIG_SCHEMA,
        "spec_binding": binding.as_result_dict(),
        "run_dir": str(run_dir.resolve()),
        "scan_result": {
            "path": str(scan_result_path.resolve()),
            "sha256": file_sha256(scan_result_path),
        },
        "source": {
            "sglang_revision": actual_revision,
            "sglang_worktree": str(sglang_worktree.resolve()),
        },
        "checkpoint": checkpoint,
        "tensor_parallel_size": contract["runtime"]["tensor_parallel_size"],
        "tp_rank": contract["sample_policy"]["capture_tp_rank"],
        "requests": requests,
        "operators": operator_configs,
    }
    if preflight:
        preflight_context = {
            "rank": config["tp_rank"],
            "size": config["tensor_parallel_size"],
        }
        config["preflight_tp_context"] = preflight_context
        for item in config["operators"]:
            item["preflight_tp_context"] = preflight_context
    write_json(config_path, config)
    return binding, config


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


def run_prepare_session(
    spec_path: Path,
    run_dir: Path,
    scan_result_path: Path,
    sglang_worktree: Path,
) -> None:
    binding, config = prepare_capture_session_config(
        spec_path,
        run_dir,
        scan_result_path,
        sglang_worktree,
    )
    config_path = run_dir / "capture-config.json"
    result = {
        "tool": "capture_golden.py",
        "action": "prepare-session",
        "spec_binding": binding.as_result_dict(),
        "passed": True,
        "capture_status": "PREPARED",
        "consumes_capture_session": False,
        "operator_ids": [
            item["operator_id"] for item in config["operators"]
        ],
        "request_modes": [
            item["input_mode"] for item in config["requests"]
        ],
        "launch_environment": {
            CAPTURE_CONFIG_ENV: str(config_path.resolve()),
        },
        "evidence": ["capture-config.json", "capture.log"],
        "summary": (
            "One TP8 model process is prepared to capture the complete gap queue "
            "from the fixed text-only and single-image requests."
        ),
    }
    write_json(run_dir / "result.json", result)
    (run_dir / "capture.log").write_text(
        "\n".join(
            [
                "action=prepare-session",
                f"sglang_revision={config['source']['sglang_revision']}",
                f"operator_count={len(config['operators'])}",
                "request_modes="
                + ",".join(item["input_mode"] for item in config["requests"]),
                "one_model_process=true",
                "capture_session_consumed=false",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_preflight_session(
    spec_path: Path,
    run_dir: Path,
    scan_result_path: Path,
    sglang_worktree: Path,
) -> None:
    binding, config = prepare_capture_session_config(
        spec_path,
        run_dir,
        scan_result_path,
        sglang_worktree,
        preflight=True,
    )
    passed, evidence = run_capture_preflight(
        run_dir / "capture-config.json",
        sglang_worktree,
    )
    result = {
        "tool": "capture_golden.py",
        "action": "preflight-session",
        "spec_binding": binding.as_result_dict(),
        "passed": passed,
        "capture_status": (
            "PREFLIGHT_PASSED" if passed else "PREFLIGHT_FAILED"
        ),
        "consumes_capture_session": False,
        "operator_ids": [
            item["operator_id"] for item in config["operators"]
        ],
        "request_modes": [
            item["input_mode"] for item in config["requests"]
        ],
        "evidence": evidence,
        "summary": (
            "All planned existing calls captured three rank-0 shapes and "
            "self-replayed in one multi-collector preflight."
            if passed
            else
            "The multi-collector preflight did not prove every planned call; "
            "inspect its per-operator logs."
        ),
    }
    write_json(run_dir / "result.json", result)


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
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "prepare",
            "prepare-session",
            "preflight",
            "preflight-session",
        ),
    )
    parser.add_argument("--scan-result", required=True, type=Path)
    parser.add_argument("--operator-id")
    parser.add_argument("--sglang-worktree", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode in {"prepare-session", "preflight-session"}:
            if args.operator_id is not None:
                raise ToolError(
                    "--operator-id is not used by session preparation"
                )
            action = (
                run_prepare_session
                if args.mode == "prepare-session"
                else run_preflight_session
            )
            action(
                args.spec,
                args.run_dir,
                args.scan_result,
                args.sglang_worktree,
            )
        else:
            if not args.operator_id:
                raise ToolError("--operator-id is required by prepare and preflight")
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
