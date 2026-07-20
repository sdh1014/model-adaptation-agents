#!/usr/bin/env python3
"""Record Golden files, then build or verify a human-copied Handoff Bundle."""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import sys
from typing import Any, Dict, Optional, Sequence

from _lib.kernel_evidence import (
    KernelEvidenceError,
    validate_kernel_worker_result,
    worker_result_sha256,
)
from _lib.spec_contract import (
    SpecContractError,
    load_contract_data,
    load_spec_binding,
)
from model_adaptation_capture.contracts import (
    CONFIG_SCHEMA,
    KERNEL_CALL_SERIALIZATION,
    KERNEL_CALL_STATE_SCHEMA,
    KERNEL_REPLAY_CONFIG_SCHEMA,
    SESSION_CONFIG_SCHEMA,
    SWIGLU_CLAMP_OPERATOR_ID,
    checkpoint_metadata,
    shape_id_for,
)


MANIFEST_SCHEMA = "handoff-manifest/v1"
GAP_MANIFEST_SCHEMA = "handoff-manifest/v2"
SAMPLE_FILES_SCHEMA = "golden-sample-files/v1"


class ToolError(RuntimeError):
    """The handoff action could not form trustworthy evidence."""


def require_gap_bundle_contract(contract: Dict[str, Any]) -> None:
    revision = contract.get("contract_revision")
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 6
    ):
        raise ToolError(
            "Scan-driven multi-operator handoff requires Contract revision 6 "
            "or later"
        )


def reject_legacy_handoff_for_current_contract(
    contract: Dict[str, Any],
) -> None:
    revision = contract.get("contract_revision")
    if isinstance(revision, int) and not isinstance(revision, bool) and revision >= 6:
        raise ToolError(
            "Contract revision 6 or later requires --scan-result and "
            "handoff-manifest/v2"
        )


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


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ToolError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def create_run_dir(run_dir: Path) -> None:
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise ToolError(f"run directory must be fresh and creatable: {error}") from error


def regular_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ToolError(f"directory does not exist: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ToolError(f"symbolic links are forbidden: {path}")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise ToolError(f"unsupported filesystem entry: {path}")
    return files


def require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ToolError(f"{label} must be one regular file")


def sample_file_records(
    golden_run: Path,
    samples: Any,
) -> list[Dict[str, Any]]:
    if not isinstance(samples, list) or not samples:
        raise ToolError("Golden Run does not contain sample metadata")
    records = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ToolError("Golden sample entry must be one object")
        shape_id = sample.get("shape_id")
        expected_file = f"samples/{shape_id}.pt"
        if not isinstance(shape_id, str) or not shape_id:
            raise ToolError("Golden sample shape_id is invalid")
        if sample.get("file") != expected_file:
            raise ToolError("Golden sample path has drifted")
        unresolved_sample_path = golden_run / expected_file
        sample_path = unresolved_sample_path.resolve()
        if (
            unresolved_sample_path.is_symlink()
            or not sample_path.is_relative_to(golden_run.resolve())
            or not sample_path.is_file()
        ):
            raise ToolError("Golden sample is missing or escapes its Run")
        records.append(
            {
                "shape_id": shape_id,
                "path": expected_file,
                "size": sample_path.stat().st_size,
                "sha256": file_sha256(sample_path),
            }
        )
    return records


def require_name_list(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not all(isinstance(name, str) and name for name in value)
        or len(value) != len(set(value))
    ):
        raise ToolError(f"{label} must be a list of unique names")
    return value


def gap_inventory(
    scan_result: Dict[str, Any],
    *,
    binding: Dict[str, Any],
    contract: Dict[str, Any],
) -> list[Dict[str, Any]]:
    """Return the ordered capture queue after validating the immutable Scan Run."""
    if scan_result.get("scan_complete") is not True:
        raise ToolError("Scan Run is not complete")
    if scan_result.get("spec_binding") != binding:
        raise ToolError("Scan Run spec_binding does not match Contract Data")
    source = scan_result.get("source")
    if not isinstance(source, dict):
        raise ToolError("Scan Run source must be one object")
    source_checks = {
        "sglang_revision": contract["source"]["sglang_revision"],
        "sglang_kunlun_revision": contract["source"][
            "sglang_kunlun_revision"
        ],
        "checkpoint_id": contract["checkpoint"]["id"],
        "checkpoint_config_sha256": contract["checkpoint"]["config_digest"],
    }
    for field, expected in source_checks.items():
        if source.get(field) != expected:
            raise ToolError(f"Scan Run source {field} does not match Contract Data")

    operators = scan_result.get("operators")
    queue = scan_result.get("gap_queue")
    plans = scan_result.get("capture_plan")
    if not isinstance(operators, list):
        raise ToolError("Scan Run operators must be a list")
    if not isinstance(queue, list) or not queue:
        raise ToolError("Scan Run gap_queue must be a non-empty list")
    if not isinstance(plans, list) or not plans:
        raise ToolError("Scan Run capture_plan must be a non-empty list")

    required: Dict[str, Dict[str, Any]] = {}
    for operator in operators:
        if not isinstance(operator, dict):
            raise ToolError("Scan Run operator entry must be one object")
        if operator.get("verdict") != "CAPTURE_REQUIRED":
            continue
        operator_id = operator.get("operator_id")
        if (
            not isinstance(operator_id, str)
            or not operator_id
            or operator_id in required
        ):
            raise ToolError(
                "Scan Run CAPTURE_REQUIRED operator ids must be unique names"
            )
        if operator.get("model_path") != ["target"]:
            raise ToolError(
                f"Scan gap operator {operator_id!r} must be target-only"
            )
        for field in ("cuda_impl", "kunlun_impl"):
            if not operator.get(field):
                raise ToolError(
                    f"Scan gap operator {operator_id!r} is missing {field}"
                )
        boundary = operator.get("boundary")
        if (
            not isinstance(boundary, dict)
            or set(boundary)
            != {"inputs", "parameters", "non_tensor_args", "outputs"}
        ):
            raise ToolError(
                f"Scan gap operator {operator_id!r} boundary is invalid"
            )
        for field, names in boundary.items():
            require_name_list(
                names,
                f"Scan gap operator {operator_id!r} boundary {field}",
            )
        if not boundary["inputs"] or not boundary["outputs"]:
            raise ToolError(
                f"Scan gap operator {operator_id!r} requires inputs and outputs"
            )
        required[operator_id] = operator

    queue_ids = []
    for item in queue:
        if (
            not isinstance(item, dict)
            or item.get("verdict") != "CAPTURE_REQUIRED"
            or not isinstance(item.get("operator_id"), str)
            or not item["operator_id"]
        ):
            raise ToolError(
                "Scan Run gap_queue must contain CAPTURE_REQUIRED operator ids"
            )
        queue_ids.append(item["operator_id"])
    if len(queue_ids) != len(set(queue_ids)):
        raise ToolError("Scan Run gap_queue contains duplicate operators")

    plan_by_id: Dict[str, Dict[str, Any]] = {}
    plan_ids = []
    for plan in plans:
        if not isinstance(plan, dict):
            raise ToolError("Scan Run capture_plan entry must be one object")
        operator_id = plan.get("operator_id")
        if (
            not isinstance(operator_id, str)
            or not operator_id
            or operator_id in plan_by_id
        ):
            raise ToolError("Scan Run capture_plan operator ids must be unique")
        plan_ids.append(operator_id)
        plan_by_id[operator_id] = plan

    if queue_ids != plan_ids or set(queue_ids) != set(required):
        raise ToolError(
            "Scan gap queue, CAPTURE_REQUIRED operators, and capture plan "
            "must contain the same operators in queue order"
        )

    boundary_to_plan = {
        "inputs": "saved_inputs",
        "parameters": "saved_parameters",
        "non_tensor_args": "saved_non_tensor_args",
        "outputs": "saved_outputs",
    }
    inventory = []
    for operator_id in queue_ids:
        operator = required[operator_id]
        plan = plan_by_id[operator_id]
        hook_target = plan.get("hook_target")
        if not isinstance(hook_target, str) or not hook_target:
            raise ToolError(
                f"capture plan hook_target for {operator_id!r} is invalid"
            )
        for boundary_field, plan_field in boundary_to_plan.items():
            expected = operator["boundary"][boundary_field]
            if plan.get(plan_field) != expected:
                raise ToolError(
                    f"capture plan {plan_field} for {operator_id!r} "
                    "does not match the scanned boundary"
                )
        plan_checks = {
            "max_distinct_shapes": contract["limits"][
                "max_shapes_per_operator"
            ],
            "tp_rank": contract["sample_policy"]["capture_tp_rank"],
            "replay_mode": "standalone-kernel-call",
            "save_direct_parameter_tensors": contract["sample_policy"][
                "save_direct_parameter_tensors"
            ],
            "save_full_checkpoint": False,
            "save_module_state_dict": False,
        }
        for field, expected in plan_checks.items():
            if plan.get(field) != expected:
                raise ToolError(
                    f"capture plan {field} for {operator_id!r} "
                    "does not match Contract Data"
                )
        inventory.append(
            {
                "operator_id": operator_id,
                "operator": operator,
                "plan": plan,
            }
        )
    return inventory


def checkpoint_from_contract(contract: Dict[str, Any]) -> Dict[str, str]:
    try:
        return checkpoint_metadata(
            contract["checkpoint"]["id"],
            contract["checkpoint"]["config_digest"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ToolError("Contract checkpoint metadata is invalid") from error


def plan_boundary(plan: Dict[str, Any]) -> Dict[str, list[str]]:
    return {
        "inputs": plan["saved_inputs"],
        "parameters": plan["saved_parameters"],
        "non_tensor_args": plan["saved_non_tensor_args"],
        "outputs": plan["saved_outputs"],
    }


def validate_tensor_metadata(
    value: Any,
    *,
    label: str,
) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"kind", "shape", "dtype", "layout", "stride"}
        or value.get("kind") != "tensor"
        or not isinstance(value.get("dtype"), str)
        or not value["dtype"]
        or value.get("layout") != "strided"
    ):
        raise ToolError(f"{label} Tensor metadata is invalid")
    shape = value.get("shape")
    stride = value.get("stride")
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in shape
        )
        or not isinstance(stride, list)
        or len(stride) != len(shape)
        or any(
            isinstance(size, bool) or not isinstance(size, int)
            for size in stride
        )
    ):
        raise ToolError(f"{label} Tensor shape or stride metadata is invalid")


def validate_signature_group(
    signature: Dict[str, Any],
    field: str,
    expected_names: Sequence[str],
) -> Dict[str, Any]:
    values = signature.get(field)
    if (
        not isinstance(values, dict)
        or len(values) != len(expected_names)
        or set(values) != set(expected_names)
    ):
        raise ToolError(f"Golden sample signature {field} differs from capture plan")
    if field != "non_tensor_args":
        for name, metadata in values.items():
            validate_tensor_metadata(
                metadata,
                label=f"Golden sample signature {field} {name}",
            )
    else:
        for name, value in values.items():
            if value is None or isinstance(value, (bool, str, int)):
                continue
            if isinstance(value, float) and math.isfinite(value):
                continue
            raise ToolError(
                f"Golden sample non-Tensor argument {name} is invalid"
            )
    return values


def validate_gap_capture_config(
    golden_run: Path,
    *,
    binding: Dict[str, Any],
    contract: Dict[str, Any],
    item: Dict[str, Any],
) -> Dict[str, Any]:
    config = read_json_object(
        golden_run / "capture-config.json",
        "Golden capture config",
    )
    operator_id = item["operator_id"]
    plan = item["plan"]
    boundary = plan_boundary(plan)
    checks = {
        "schema": CONFIG_SCHEMA,
        "spec_binding": binding,
        "operator_id": operator_id,
        "model_path": "target",
        "max_shapes": contract["limits"]["max_shapes_per_operator"],
        "tensor_parallel_size": contract["runtime"]["tensor_parallel_size"],
        "tp_rank": contract["sample_policy"]["capture_tp_rank"],
        "serialization": KERNEL_CALL_SERIALIZATION,
        "capture_device_type": "cuda",
        "dtype": contract["runtime"]["dtype"],
        "checkpoint": checkpoint_from_contract(contract),
        "precision_gate": contract["precision_gate"],
        "hook_target": plan["hook_target"],
        "boundary": boundary,
        "sample_fields": boundary,
    }
    for field, expected in checks.items():
        if config.get(field) != expected:
            raise ToolError(
                f"Golden capture config {field} for {operator_id!r} has drifted"
            )
    if not isinstance(config.get("adapter"), str) or not config["adapter"]:
        raise ToolError(
            f"Golden capture config adapter for {operator_id!r} is invalid"
        )
    if (
        not isinstance(config.get("session_config"), str)
        or not config["session_config"]
    ):
        raise ToolError(
            f"Golden capture config session_config for "
            f"{operator_id!r} is invalid"
        )
    replay = config.get("replay")
    if (
        not isinstance(replay, dict)
        or replay.get("mode") != "standalone-kernel-call"
        or replay.get("cuda_target") != plan["hook_target"]
        or replay.get("weights_in_golden_sample") is not False
    ):
        raise ToolError(
            f"Golden capture replay boundary for {operator_id!r} has drifted"
        )
    return config


def validate_gap_capture_state(
    golden_run: Path,
    *,
    binding: Dict[str, Any],
    contract: Dict[str, Any],
    item: Dict[str, Any],
    sealed: bool,
) -> tuple[Dict[str, Any], list[str]]:
    operator_id = item["operator_id"]
    plan = item["plan"]
    capture_config = validate_gap_capture_config(
        golden_run,
        binding=binding,
        contract=contract,
        item=item,
    )
    state = read_json_object(
        golden_run / "capture-state.json",
        "Golden capture state",
    )
    checkpoint = checkpoint_from_contract(contract)
    checks = {
        "schema": KERNEL_CALL_STATE_SCHEMA,
        "spec_binding": binding,
        "operator_id": operator_id,
        "tp_rank": contract["sample_policy"]["capture_tp_rank"],
        "tensor_parallel_size": contract["runtime"]["tensor_parallel_size"],
        "checkpoint": checkpoint,
        "loaded_checkpoint": {
            "model_path": checkpoint["model_path"],
            "revision": checkpoint["revision"],
        },
        "capture_session_config": capture_config["session_config"],
    }
    for field, expected in checks.items():
        if state.get(field) != expected:
            raise ToolError(
                f"Golden capture state {field} for {operator_id!r} has drifted"
            )
    process_id = state.get("capture_process_id")
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
    ):
        raise ToolError(
            f"Golden capture process id for {operator_id!r} is invalid"
        )
    if sealed:
        if state.get("status") != "SEALED" or state.get("capture_closed") is not True:
            raise ToolError(
                f"Golden Run for {operator_id!r} must be SEALED and capture_closed"
            )
    elif state.get("status") != "ACTIVE" or state.get("capture_closed") is not False:
        raise ToolError(
            f"record-samples requires an ACTIVE Golden Run for {operator_id!r}"
        )

    samples = state.get("samples")
    max_shapes = contract["limits"]["max_shapes_per_operator"]
    if not isinstance(samples, list) or not 1 <= len(samples) <= max_shapes:
        raise ToolError(
            f"Golden Run for {operator_id!r} must contain one to three samples"
        )
    if state.get("saved_shape_count") != len(samples):
        raise ToolError(
            f"Golden Run saved_shape_count for {operator_id!r} has drifted"
        )
    shape_ids = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ToolError("Golden sample entry must be one object")
        shape_id = sample.get("shape_id")
        signature = sample.get("signature")
        if (
            not isinstance(shape_id, str)
            or not shape_id
            or not isinstance(signature, dict)
        ):
            raise ToolError("Golden sample shape metadata is invalid")
        signature_checks = {
            "operator_id": operator_id,
            "model_path": "target",
            "tensor_parallel_size": contract["runtime"]["tensor_parallel_size"],
            "tp_rank": contract["sample_policy"]["capture_tp_rank"],
        }
        for field, expected in signature_checks.items():
            if signature.get(field) != expected:
                raise ToolError(
                    f"Golden sample signature {field} for "
                    f"{operator_id!r} has drifted"
                )
        inputs = validate_signature_group(
            signature,
            "inputs",
            plan["saved_inputs"],
        )
        validate_signature_group(
            signature,
            "parameters",
            plan["saved_parameters"],
        )
        validate_signature_group(
            signature,
            "non_tensor_args",
            plan["saved_non_tensor_args"],
        )
        validate_signature_group(
            signature,
            "outputs",
            plan["saved_outputs"],
        )
        try:
            expected_shape_id = shape_id_for(
                {
                    name: inputs[name]["shape"]
                    for name in plan["saved_inputs"]
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ToolError("Golden sample input shape metadata has drifted") from error
        if expected_shape_id != shape_id or shape_id in shape_ids:
            raise ToolError("Golden sample shape digest has drifted")
        expected_file = f"samples/{shape_id}.pt"
        if sample.get("file") != expected_file:
            raise ToolError("Golden sample path has drifted")
        sample_path = (golden_run / expected_file).resolve()
        if (
            not sample_path.is_relative_to(golden_run.resolve())
            or sample_path.is_symlink()
            or not sample_path.is_file()
        ):
            raise ToolError("Golden sample is missing or escapes its Run")
        shape_ids.append(shape_id)
    return state, shape_ids


def run_record_gap_samples(
    spec_path: Path,
    run_dir: Path,
    golden_run: Path,
    scan_result_path: Path,
) -> None:
    contract = load_contract_data(spec_path)
    require_gap_bundle_contract(contract)
    binding = load_spec_binding(spec_path).as_result_dict()
    scan_result = read_json_object(scan_result_path, "Scan Run result")
    inventory = gap_inventory(
        scan_result,
        binding=binding,
        contract=contract,
    )
    if golden_run.is_symlink():
        raise ToolError("Golden Run must not be a symbolic link")
    golden_run = golden_run.resolve()
    state = read_json_object(
        golden_run / "capture-state.json",
        "Golden capture state",
    )
    operator_id = state.get("operator_id")
    matches = [item for item in inventory if item["operator_id"] == operator_id]
    if len(matches) != 1:
        raise ToolError(
            "Golden operator is not present in the Scan gap queue"
        )
    if paths_overlap(run_dir, golden_run):
        raise ToolError(
            "record-samples run directory and Golden Run must not overlap"
        )
    state, _ = validate_gap_capture_state(
        golden_run,
        binding=binding,
        contract=contract,
        item=matches[0],
        sealed=False,
    )
    if "self_replay" in state or (golden_run / "self-replay").exists():
        raise ToolError("record-samples must run before CUDA self-replay")
    records = sample_file_records(golden_run, state["samples"])
    evidence_path = golden_run / "sample-files.json"
    if evidence_path.exists() or evidence_path.is_symlink():
        raise ToolError("Golden Run sample-files.json already exists")
    evidence = {
        "schema": SAMPLE_FILES_SCHEMA,
        "spec_binding": binding,
        "operator_id": operator_id,
        "files": records,
    }
    create_run_dir(run_dir)
    temporary = golden_run / ".sample-files.json.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise ToolError("temporary sample file evidence already exists")
    write_json(temporary, evidence)
    temporary.replace(evidence_path)
    evidence_digest = file_sha256(evidence_path)
    result = {
        "tool": "handoff_bundle.py",
        "action": "record-samples",
        "spec_binding": binding,
        "operator_id": operator_id,
        "passed": True,
        "actual_tensors_saved": False,
        "golden_run": str(golden_run),
        "golden_status": "ACTIVE",
        "recorded_before_self_replay": True,
        "sample_file_count": len(records),
        "sample_files_path": str(evidence_path),
        "sample_files_sha256": evidence_digest,
        "evidence": ["handoff.log"],
        "summary": (
            "Golden Sample file bytes were recorded before CUDA self-replay."
        ),
    }
    write_json(run_dir / "result.json", result)
    (run_dir / "handoff.log").write_text(
        "\n".join(
            [
                "action=record-samples",
                f"operator_id={operator_id}",
                "passed=true",
                f"sample_file_count={len(records)}",
                f"sample_files_sha256={evidence_digest}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_record_samples(
    spec_path: Path,
    run_dir: Path,
    golden_run: Path,
) -> None:
    contract = load_contract_data(spec_path)
    reject_legacy_handoff_for_current_contract(contract)
    binding = load_spec_binding(spec_path).as_result_dict()
    if golden_run.is_symlink():
        raise ToolError("Golden Run must not be a symbolic link")
    golden_run = golden_run.resolve()
    if paths_overlap(run_dir, golden_run):
        raise ToolError(
            "record-samples run directory and Golden Run must not overlap"
        )
    state = read_json_object(
        golden_run / "capture-state.json",
        "Golden capture state",
    )
    try:
        checkpoint = checkpoint_metadata(
            contract["checkpoint"]["id"],
            contract["checkpoint"]["config_digest"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ToolError("Contract checkpoint metadata is invalid") from error
    checks = {
        "schema": KERNEL_CALL_STATE_SCHEMA,
        "spec_binding": binding,
        "operator_id": SWIGLU_CLAMP_OPERATOR_ID,
        "tp_rank": contract["sample_policy"]["capture_tp_rank"],
        "tensor_parallel_size": contract["runtime"]["tensor_parallel_size"],
        "checkpoint": checkpoint,
        "loaded_checkpoint": {
            "model_path": checkpoint["model_path"],
            "revision": checkpoint["revision"],
        },
        "status": "ACTIVE",
        "capture_closed": False,
    }
    for field, expected in checks.items():
        if state.get(field) != expected:
            raise ToolError(
                f"record-samples Golden capture state {field} has drifted"
            )
    if "self_replay" in state or (golden_run / "self-replay").exists():
        raise ToolError("record-samples must run before CUDA self-replay")
    samples = state.get("samples")
    max_shapes = contract["limits"]["max_shapes_per_operator"]
    if not isinstance(samples, list) or not 1 <= len(samples) <= max_shapes:
        raise ToolError("Golden Run must contain one to three samples")
    if state.get("saved_shape_count") != len(samples):
        raise ToolError("Golden Run saved_shape_count has drifted")
    records = sample_file_records(golden_run, samples)
    evidence_path = golden_run / "sample-files.json"
    if evidence_path.exists() or evidence_path.is_symlink():
        raise ToolError("Golden Run sample-files.json already exists")

    evidence = {
        "schema": SAMPLE_FILES_SCHEMA,
        "spec_binding": binding,
        "operator_id": SWIGLU_CLAMP_OPERATOR_ID,
        "files": records,
    }
    create_run_dir(run_dir)
    temporary = golden_run / ".sample-files.json.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise ToolError("temporary sample file evidence already exists")
    write_json(temporary, evidence)
    temporary.replace(evidence_path)
    evidence_digest = file_sha256(evidence_path)

    result = {
        "tool": "handoff_bundle.py",
        "action": "record-samples",
        "spec_binding": binding,
        "operator_id": SWIGLU_CLAMP_OPERATOR_ID,
        "passed": True,
        "actual_tensors_saved": False,
        "golden_run": str(golden_run),
        "golden_status": "ACTIVE",
        "recorded_before_self_replay": True,
        "sample_file_count": len(records),
        "sample_files_path": str(evidence_path),
        "sample_files_sha256": evidence_digest,
        "evidence": ["handoff.log"],
        "summary": "Golden Sample file bytes were recorded before CUDA self-replay.",
    }
    write_json(run_dir / "result.json", result)
    (run_dir / "handoff.log").write_text(
        "\n".join(
            [
                "action=record-samples",
                "passed=true",
                f"sample_file_count={len(records)}",
                f"sample_files_sha256={evidence_digest}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def validate_golden_run(
    golden_run: Path,
    binding: Dict[str, Any],
    contract: Dict[str, Any],
) -> str:
    files = regular_files(golden_run)
    state = read_json_object(
        golden_run / "capture-state.json",
        "Golden capture state",
    )
    try:
        checkpoint = checkpoint_metadata(
            contract["checkpoint"]["id"],
            contract["checkpoint"]["config_digest"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ToolError("Contract checkpoint metadata is invalid") from error
    checks = {
        "schema": KERNEL_CALL_STATE_SCHEMA,
        "spec_binding": binding,
        "operator_id": SWIGLU_CLAMP_OPERATOR_ID,
        "tp_rank": contract["sample_policy"]["capture_tp_rank"],
        "tensor_parallel_size": contract["runtime"]["tensor_parallel_size"],
        "checkpoint": checkpoint,
        "loaded_checkpoint": {
            "model_path": checkpoint["model_path"],
            "revision": checkpoint["revision"],
        },
    }
    for field, expected in checks.items():
        if state.get(field) != expected:
            raise ToolError(f"Golden capture state {field} has drifted")
    if state.get("status") != "SEALED" or state.get("capture_closed") is not True:
        raise ToolError("Golden Run must be SEALED and capture_closed")
    self_replay = state.get("self_replay")
    if not isinstance(self_replay, dict) or self_replay.get("passed") is not True:
        raise ToolError("Golden Run self-replay has not passed")
    samples = state.get("samples")
    max_shapes = contract["limits"]["max_shapes_per_operator"]
    if not isinstance(samples, list) or not 1 <= len(samples) <= max_shapes:
        raise ToolError("Golden Run must contain one to three samples")
    if state.get("saved_shape_count") != len(samples):
        raise ToolError("Golden Run saved_shape_count has drifted")
    if self_replay.get("checked_shape_count") != len(samples):
        raise ToolError("Golden Run self-replay did not check every sample")
    recorded_worker_digest = self_replay.get("worker_result_sha256")
    if not is_sha256(recorded_worker_digest):
        raise ToolError("Golden Run self_replay worker_result_sha256 is invalid")
    shape_ids = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ToolError("Golden sample entry must be one object")
        shape_id = sample.get("shape_id")
        signature = sample.get("signature")
        expected_file = f"samples/{shape_id}.pt"
        if (
            not isinstance(shape_id, str)
            or not shape_id
            or not isinstance(signature, dict)
        ):
            raise ToolError("Golden sample shape_id is invalid")
        signature_checks = {
            "operator_id": SWIGLU_CLAMP_OPERATOR_ID,
            "model_path": "target",
            "tensor_parallel_size": contract["runtime"]["tensor_parallel_size"],
            "tp_rank": contract["sample_policy"]["capture_tp_rank"],
            "parameters": {},
        }
        for field, expected in signature_checks.items():
            if signature.get(field) != expected:
                raise ToolError(
                    f"Golden sample signature {field} has drifted"
                )
        try:
            input_shape = signature["inputs"]["x"]["shape"]
            expected_shape_id = shape_id_for(input_shape)
        except (KeyError, TypeError, ValueError) as error:
            raise ToolError("Golden sample input shape metadata has drifted") from error
        if expected_shape_id != shape_id or shape_id in shape_ids:
            raise ToolError("Golden sample shape digest has drifted")
        limit = signature.get("non_tensor_args", {}).get("gemm1_limit")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, (int, float))
            or not math.isfinite(limit)
        ):
            raise ToolError("Golden sample gemm1_limit has drifted")
        if sample.get("file") != expected_file:
            raise ToolError("Golden sample path has drifted")
        sample_path = (golden_run / expected_file).resolve()
        if (
            not sample_path.is_relative_to(golden_run.resolve())
            or sample_path.is_symlink()
            or not sample_path.is_file()
        ):
            raise ToolError("Golden sample is missing or escapes its Run")
        shape_ids.append(shape_id)

    current_sample_files = sample_file_records(golden_run, samples)
    expected_sample_file_evidence = {
        "schema": SAMPLE_FILES_SCHEMA,
        "spec_binding": binding,
        "operator_id": SWIGLU_CLAMP_OPERATOR_ID,
        "files": current_sample_files,
    }
    recorded_sample_files = read_json_object(
        golden_run / "sample-files.json",
        "Golden sample file evidence",
    )
    if recorded_sample_files != expected_sample_file_evidence:
        raise ToolError(
            "Golden sample bytes do not match sample-files.json"
        )
    sample_files_digest = file_sha256(
        golden_run / "sample-files.json"
    )
    if self_replay.get("sample_files_sha256") != sample_files_digest:
        raise ToolError(
            "Golden Run self_replay does not bind sample-files.json"
        )

    capture_config = read_json_object(
        golden_run / "capture-config.json",
        "Golden capture config",
    )
    config_checks = {
        "spec_binding": binding,
        "operator_id": SWIGLU_CLAMP_OPERATOR_ID,
        "checkpoint": checkpoint,
        "tensor_parallel_size": contract["runtime"]["tensor_parallel_size"],
        "tp_rank": contract["sample_policy"]["capture_tp_rank"],
        "precision_gate": contract["precision_gate"],
    }
    for field, expected in config_checks.items():
        if capture_config.get(field) != expected:
            raise ToolError(f"Golden capture config {field} has drifted")

    capture_result = read_json_object(
        golden_run / "result.json",
        "Golden capture result",
    )
    result_checks = {
        "tool": "capture_golden.py",
        "action": "prepare",
        "spec_binding": binding,
        "operator_id": SWIGLU_CLAMP_OPERATOR_ID,
        "passed": True,
    }
    for field, expected in result_checks.items():
        if capture_result.get(field) != expected:
            raise ToolError(f"Golden capture result {field} has drifted")
    require_regular_file(golden_run / "capture.log", "Golden capture log")

    replay_dir = golden_run / "self-replay"
    replay_config = read_json_object(
        replay_dir / "replay-config.json",
        "CUDA self-replay config",
    )
    replay_config_checks = {
        "schema": KERNEL_REPLAY_CONFIG_SCHEMA,
        "spec_binding": binding,
        "operator_id": SWIGLU_CLAMP_OPERATOR_ID,
        "execution_site": "cuda",
        "invocation_target": SWIGLU_CLAMP_OPERATOR_ID,
        "tensor_parallel_size": contract["runtime"]["tensor_parallel_size"],
        "tp_rank": contract["sample_policy"]["capture_tp_rank"],
        "precision_gate": contract["precision_gate"],
        "sample_files_sha256": sample_files_digest,
        "allow_active_capture": True,
    }
    for field, expected in replay_config_checks.items():
        if replay_config.get(field) != expected:
            raise ToolError(f"CUDA self-replay config {field} has drifted")

    worker_result = read_json_object(
        replay_dir / "worker-result.json",
        "CUDA self-replay worker result",
    )
    recomputed_worker_digest = worker_result_sha256(worker_result)
    if recomputed_worker_digest != recorded_worker_digest:
        raise ToolError(
            "Golden Run self_replay worker_result_sha256 does not match "
            "worker-result.json"
        )
    try:
        validate_kernel_worker_result(
            worker_result,
            spec_binding=binding,
            operator_id=SWIGLU_CLAMP_OPERATOR_ID,
            execution_site="cuda",
            invocation_target=SWIGLU_CLAMP_OPERATOR_ID,
            tp_rank=contract["sample_policy"]["capture_tp_rank"],
            tensor_parallel_size=contract["runtime"]["tensor_parallel_size"],
            precision_gate=contract["precision_gate"],
            expected_shape_ids=shape_ids,
            sample_files_sha256=sample_files_digest,
        )
    except KernelEvidenceError as error:
        raise ToolError(str(error)) from error
    if worker_result.get("passed") is not True:
        raise ToolError("CUDA self-replay worker did not pass")

    replay_result = read_json_object(
        replay_dir / "result.json",
        "CUDA self-replay result",
    )
    replay_result_checks = {
        "tool": "replay_compare.py",
        "action": "kernel-replay",
        "spec_binding": binding,
        "operator_id": SWIGLU_CLAMP_OPERATOR_ID,
        "execution_site": "cuda",
        "invocation_target": SWIGLU_CLAMP_OPERATOR_ID,
        "passed": True,
        "checked_shape_count": len(shape_ids),
        "failed_shape_count": 0,
        "precision_gate": contract["precision_gate"],
        "sample_files_sha256": sample_files_digest,
        "actual_tensors_saved": False,
        "evidence": [
            "replay-config.json",
            "worker-result.json",
            "replay.log",
        ],
    }
    for field, expected in replay_result_checks.items():
        if replay_result.get(field) != expected:
            raise ToolError(f"CUDA self-replay result {field} has drifted")
    require_regular_file(replay_dir / "replay.log", "CUDA self-replay log")

    allowed_files = {
        "capture-config.json",
        "capture-state.json",
        "capture.log",
        "result.json",
        "sample-files.json",
        "self-replay/replay-config.json",
        "self-replay/replay.log",
        "self-replay/result.json",
        "self-replay/worker-result.json",
        *(f"samples/{shape_id}.pt" for shape_id in shape_ids),
    }
    actual_files = {path.relative_to(golden_run).as_posix() for path in files}
    unexpected = sorted(actual_files - allowed_files)
    if unexpected:
        raise ToolError(
            "Golden Run file is not permitted: " + ", ".join(unexpected)
        )
    missing = sorted(allowed_files - actual_files)
    if missing:
        raise ToolError("Golden Run required file is missing: " + ", ".join(missing))
    return SWIGLU_CLAMP_OPERATOR_ID


def validate_gap_golden_run(
    golden_run: Path,
    *,
    binding: Dict[str, Any],
    contract: Dict[str, Any],
    item: Dict[str, Any],
) -> str:
    files = regular_files(golden_run)
    operator_id = item["operator_id"]
    plan = item["plan"]
    state, shape_ids = validate_gap_capture_state(
        golden_run,
        binding=binding,
        contract=contract,
        item=item,
        sealed=True,
    )
    self_replay = state.get("self_replay")
    if not isinstance(self_replay, dict) or self_replay.get("passed") is not True:
        raise ToolError(
            f"Golden Run self-replay for {operator_id!r} has not passed"
        )
    if self_replay.get("checked_shape_count") != len(shape_ids):
        raise ToolError(
            f"Golden Run self-replay for {operator_id!r} "
            "did not check every sample"
        )
    recorded_worker_digest = self_replay.get("worker_result_sha256")
    if not is_sha256(recorded_worker_digest):
        raise ToolError(
            f"Golden Run self-replay digest for {operator_id!r} is invalid"
        )

    current_sample_files = sample_file_records(golden_run, state["samples"])
    expected_sample_file_evidence = {
        "schema": SAMPLE_FILES_SCHEMA,
        "spec_binding": binding,
        "operator_id": operator_id,
        "files": current_sample_files,
    }
    recorded_sample_files = read_json_object(
        golden_run / "sample-files.json",
        "Golden sample file evidence",
    )
    if recorded_sample_files != expected_sample_file_evidence:
        raise ToolError(
            f"Golden sample bytes for {operator_id!r} "
            "do not match sample-files.json"
        )
    sample_files_digest = file_sha256(golden_run / "sample-files.json")
    if self_replay.get("sample_files_sha256") != sample_files_digest:
        raise ToolError(
            f"Golden Run self-replay for {operator_id!r} "
            "does not bind sample-files.json"
        )

    capture_config = validate_gap_capture_config(
        golden_run,
        binding=binding,
        contract=contract,
        item=item,
    )
    boundary = plan_boundary(plan)
    replay_dir = golden_run / "self-replay"
    replay_config = read_json_object(
        replay_dir / "replay-config.json",
        "CUDA self-replay config",
    )
    replay_config_checks = {
        "schema": KERNEL_REPLAY_CONFIG_SCHEMA,
        "spec_binding": binding,
        "operator_id": operator_id,
        "adapter": capture_config["adapter"],
        "sample_fields": boundary,
        "execution_site": "cuda",
        "invocation_target": plan["hook_target"],
        "tensor_parallel_size": contract["runtime"]["tensor_parallel_size"],
        "tp_rank": contract["sample_policy"]["capture_tp_rank"],
        "precision_gate": contract["precision_gate"],
        "sample_files_sha256": sample_files_digest,
        "allow_active_capture": True,
    }
    for field, expected in replay_config_checks.items():
        if replay_config.get(field) != expected:
            raise ToolError(
                f"CUDA self-replay config {field} for "
                f"{operator_id!r} has drifted"
            )
    for field in ("golden_run", "run_dir"):
        if (
            not isinstance(replay_config.get(field), str)
            or not replay_config[field]
        ):
            raise ToolError(
                f"CUDA self-replay config {field} for "
                f"{operator_id!r} is invalid"
            )

    worker_result = read_json_object(
        replay_dir / "worker-result.json",
        "CUDA self-replay worker result",
    )
    recomputed_worker_digest = worker_result_sha256(worker_result)
    if recomputed_worker_digest != recorded_worker_digest:
        raise ToolError(
            f"Golden Run self-replay digest for {operator_id!r} "
            "does not match worker-result.json"
        )
    try:
        validate_kernel_worker_result(
            worker_result,
            spec_binding=binding,
            operator_id=operator_id,
            execution_site="cuda",
            invocation_target=plan["hook_target"],
            tp_rank=contract["sample_policy"]["capture_tp_rank"],
            tensor_parallel_size=contract["runtime"]["tensor_parallel_size"],
            precision_gate=contract["precision_gate"],
            expected_shape_ids=shape_ids,
            sample_files_sha256=sample_files_digest,
        )
    except KernelEvidenceError as error:
        raise ToolError(str(error)) from error
    if worker_result.get("passed") is not True:
        raise ToolError(
            f"CUDA self-replay worker for {operator_id!r} did not pass"
        )

    replay_result = read_json_object(
        replay_dir / "result.json",
        "CUDA self-replay result",
    )
    replay_result_checks = {
        "tool": "replay_compare.py",
        "action": "kernel-replay",
        "spec_binding": binding,
        "operator_id": operator_id,
        "execution_site": "cuda",
        "invocation_target": plan["hook_target"],
        "passed": True,
        "checked_shape_count": len(shape_ids),
        "failed_shape_count": 0,
        "precision_gate": contract["precision_gate"],
        "sample_files_sha256": sample_files_digest,
        "actual_tensors_saved": False,
        "evidence": [
            "replay-config.json",
            "worker-result.json",
            "replay.log",
        ],
    }
    for field, expected in replay_result_checks.items():
        if replay_result.get(field) != expected:
            raise ToolError(
                f"CUDA self-replay result {field} for "
                f"{operator_id!r} has drifted"
            )
    require_regular_file(replay_dir / "replay.log", "CUDA self-replay log")

    allowed_files = {
        "capture-config.json",
        "capture-state.json",
        "sample-files.json",
        "self-replay/replay-config.json",
        "self-replay/replay.log",
        "self-replay/result.json",
        "self-replay/worker-result.json",
        *(f"samples/{shape_id}.pt" for shape_id in shape_ids),
    }
    actual_files = {path.relative_to(golden_run).as_posix() for path in files}
    unexpected = sorted(actual_files - allowed_files)
    if unexpected:
        raise ToolError(
            f"Golden Run file for {operator_id!r} is not permitted: "
            + ", ".join(unexpected)
        )
    missing = sorted(allowed_files - actual_files)
    if missing:
        raise ToolError(
            f"Golden Run required file for {operator_id!r} is missing: "
            + ", ".join(missing)
        )
    return operator_id


def validate_gap_capture_session_evidence(
    session_config: Dict[str, Any],
    *,
    binding: Dict[str, Any],
    contract: Dict[str, Any],
    inventory: Sequence[Dict[str, Any]],
    scan_result_digest: str,
    capture_configs: Dict[str, Dict[str, Any]],
    capture_states: Dict[str, Dict[str, Any]],
    live_session_path: Optional[Path] = None,
    live_scan_result_path: Optional[Path] = None,
    live_golden_runs: Optional[Dict[str, Path]] = None,
) -> int:
    """Prove that every Golden belongs to one revision-6 capture session."""
    expected_ids = [item["operator_id"] for item in inventory]
    if list(capture_configs) != expected_ids or list(capture_states) != expected_ids:
        raise ToolError(
            "capture session evidence does not follow the Scan gap queue"
        )

    session_refs = {
        config.get("session_config") for config in capture_configs.values()
    }
    if (
        len(session_refs) != 1
        or not all(isinstance(value, str) and value for value in session_refs)
    ):
        raise ToolError(
            "all Golden Runs must reference one capture session config"
        )
    session_ref = next(iter(session_refs))
    session_ref_path = Path(session_ref)
    if not session_ref_path.is_absolute() or session_ref_path.name != "capture-config.json":
        raise ToolError("capture session config path is invalid")

    process_ids = {
        state.get("capture_process_id") for state in capture_states.values()
    }
    if (
        len(process_ids) != 1
        or not all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
            for value in process_ids
        )
    ):
        raise ToolError(
            "all Golden Runs must come from one capture process"
        )
    process_id = next(iter(process_ids))

    checkpoint = checkpoint_from_contract(contract)
    checks = {
        "schema": SESSION_CONFIG_SCHEMA,
        "spec_binding": binding,
        "checkpoint": checkpoint,
        "tensor_parallel_size": contract["runtime"]["tensor_parallel_size"],
        "tp_rank": contract["sample_policy"]["capture_tp_rank"],
    }
    for field, expected in checks.items():
        if session_config.get(field) != expected:
            raise ToolError(f"capture session config {field} has drifted")

    session_run_dir = session_config.get("run_dir")
    if not isinstance(session_run_dir, str) or not session_run_dir:
        raise ToolError("capture session run_dir is invalid")
    session_run_path = Path(session_run_dir)
    if (
        not session_run_path.is_absolute()
        or session_ref_path != session_run_path / "capture-config.json"
    ):
        raise ToolError(
            "capture session config path does not match its run_dir"
        )

    scan_record = session_config.get("scan_result")
    if (
        not isinstance(scan_record, dict)
        or set(scan_record) != {"path", "sha256"}
        or not isinstance(scan_record.get("path"), str)
        or not scan_record["path"]
        or scan_record.get("sha256") != scan_result_digest
    ):
        raise ToolError("capture session Scan Run binding has drifted")
    source = session_config.get("source")
    if (
        not isinstance(source, dict)
        or source.get("sglang_revision")
        != contract["source"]["sglang_revision"]
        or not isinstance(source.get("sglang_worktree"), str)
        or not source["sglang_worktree"]
    ):
        raise ToolError("capture session source has drifted")
    if not isinstance(session_config.get("requests"), list):
        raise ToolError("capture session requests must be a list")

    session_operators = session_config.get("operators")
    expected_operator_configs = [
        capture_configs[operator_id] for operator_id in expected_ids
    ]
    if session_operators != expected_operator_configs:
        raise ToolError(
            "capture session operators do not match the Scan-ordered "
            "Golden capture configs"
        )
    for index, operator_id in enumerate(expected_ids, start=1):
        config = capture_configs[operator_id]
        operator_run_dir = config.get("run_dir")
        expected_run_dir = session_run_path / "operators" / f"{index:02d}"
        if (
            not isinstance(operator_run_dir, str)
            or Path(operator_run_dir) != expected_run_dir
        ):
            raise ToolError(
                f"Golden capture run_dir for {operator_id!r} "
                "does not match its capture session slot"
            )
        state = capture_states[operator_id]
        if state.get("capture_session_config") != session_ref:
            raise ToolError(
                f"Golden capture state for {operator_id!r} "
                "does not match the common capture session"
            )

    if live_session_path is not None:
        if live_session_path.resolve() != session_ref_path.resolve():
            raise ToolError(
                "Golden capture session reference does not match the "
                "supplied session file"
            )
        if live_session_path.parent.resolve() != session_run_path.resolve():
            raise ToolError(
                "capture session file does not live in its declared run_dir"
            )
    if live_scan_result_path is not None:
        if Path(scan_record["path"]).resolve() != live_scan_result_path.resolve():
            raise ToolError(
                "capture session Scan Run path does not match the supplied Scan Run"
            )
    if live_golden_runs is not None:
        if list(live_golden_runs) != expected_ids:
            raise ToolError(
                "live Golden Runs do not follow the Scan gap queue"
            )
        for operator_id in expected_ids:
            golden_run = live_golden_runs[operator_id]
            declared_run = Path(capture_configs[operator_id]["run_dir"])
            if golden_run.resolve() != declared_run.resolve():
                raise ToolError(
                    f"Golden Run for {operator_id!r} does not match "
                    "its capture session run_dir"
                )
            if not golden_run.resolve().is_relative_to(
                session_run_path.resolve()
            ):
                raise ToolError(
                    f"Golden Run for {operator_id!r} escapes "
                    "the capture session"
                )
    return process_id


def load_live_gap_capture_session(
    *,
    binding: Dict[str, Any],
    contract: Dict[str, Any],
    inventory: Sequence[Dict[str, Any]],
    scan_result_path: Path,
    capture_configs: Dict[str, Dict[str, Any]],
    capture_states: Dict[str, Dict[str, Any]],
    golden_runs: Dict[str, Path],
) -> tuple[Path, Dict[str, Any], int]:
    refs = {
        config.get("session_config") for config in capture_configs.values()
    }
    if (
        len(refs) != 1
        or not all(isinstance(value, str) and value for value in refs)
    ):
        raise ToolError(
            "all Golden Runs must reference one capture session config"
        )
    unresolved = Path(next(iter(refs)))
    if unresolved.is_symlink():
        raise ToolError("capture session config must not be a symbolic link")
    require_regular_file(unresolved, "capture session config")
    session_path = unresolved.resolve()
    session_config = read_json_object(
        session_path,
        "capture session config",
    )
    process_id = validate_gap_capture_session_evidence(
        session_config,
        binding=binding,
        contract=contract,
        inventory=inventory,
        scan_result_digest=file_sha256(scan_result_path),
        capture_configs=capture_configs,
        capture_states=capture_states,
        live_session_path=session_path,
        live_scan_result_path=scan_result_path,
        live_golden_runs=golden_runs,
    )
    return session_path, session_config, process_id


def validate_sample_record_result(
    result_path: Path,
    *,
    golden_run: Path,
    binding: Dict[str, Any],
) -> tuple[str, str]:
    return validate_gap_sample_record_result(
        result_path,
        golden_run=golden_run,
        binding=binding,
        operator_id=SWIGLU_CLAMP_OPERATOR_ID,
    )


def validate_gap_sample_record_result(
    result_path: Path,
    *,
    golden_run: Path,
    binding: Dict[str, Any],
    operator_id: str,
) -> tuple[str, str]:
    if result_path.is_symlink():
        raise ToolError("sample record result must not be a symbolic link")
    result_path = result_path.resolve()
    if result_path.name != "result.json":
        raise ToolError("sample record evidence must point to a Run result.json")
    require_regular_file(result_path, "sample record result")
    record_run = result_path.parent
    record_files = {
        path.relative_to(record_run).as_posix()
        for path in regular_files(record_run)
    }
    if record_files != {"result.json", "handoff.log"}:
        raise ToolError(
            "sample record Run must contain only result.json and handoff.log"
        )
    result = read_json_object(result_path, "sample record result")
    sample_files_path = golden_run / "sample-files.json"
    sample_files = read_json_object(
        sample_files_path,
        "Golden sample file evidence",
    )
    sample_file_entries = sample_files.get("files")
    checks = {
        "tool": "handoff_bundle.py",
        "action": "record-samples",
        "spec_binding": binding,
        "operator_id": operator_id,
        "passed": True,
        "actual_tensors_saved": False,
        "golden_run": str(golden_run),
        "golden_status": "ACTIVE",
        "recorded_before_self_replay": True,
        "sample_file_count": (
            len(sample_file_entries)
            if isinstance(sample_file_entries, list)
            else None
        ),
        "sample_files_path": str(sample_files_path),
        "sample_files_sha256": file_sha256(sample_files_path),
        "evidence": ["handoff.log"],
    }
    for field, expected in checks.items():
        if result.get(field) != expected:
            if field == "sample_files_sha256":
                raise ToolError(
                    "record-samples result does not match sample-files.json "
                    f"for {operator_id!r}"
                )
            raise ToolError(
                f"sample record result {field} for "
                f"{operator_id!r} has drifted"
            )
    return file_sha256(result_path), checks["sample_files_sha256"]


def manifest_entries(bundle_dir: Path) -> list[Dict[str, Any]]:
    entries = []
    for path in regular_files(bundle_dir):
        relative = path.relative_to(bundle_dir).as_posix()
        if relative == "manifest.json":
            continue
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return entries


def copy_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for source_path in regular_files(source):
        relative = source_path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)


def run_build(
    spec_path: Path,
    run_dir: Path,
    golden_run: Path,
    bundle_spec: Path,
    sample_record_result: Path,
) -> None:
    contract = load_contract_data(spec_path)
    reject_legacy_handoff_for_current_contract(contract)
    binding = load_spec_binding(spec_path).as_result_dict()
    bundle_binding = load_spec_binding(bundle_spec).as_result_dict()
    if bundle_binding != binding:
        raise ToolError("bundle Spec Contract Data does not match current Contract Data")
    if bundle_spec.is_symlink() or not bundle_spec.is_file():
        raise ToolError("bundle Spec must be one regular file")
    if golden_run.is_symlink():
        raise ToolError("Golden Run must not be a symbolic link")
    golden_run = golden_run.resolve()
    if paths_overlap(run_dir, golden_run):
        raise ToolError("build run directory and Golden Run must not overlap")
    if paths_overlap(run_dir, sample_record_result.resolve().parent):
        raise ToolError(
            "build run directory and sample record Run must not overlap"
        )
    operator_id = validate_golden_run(golden_run, binding, contract)
    (
        sample_record_result_digest,
        sample_files_digest,
    ) = validate_sample_record_result(
        sample_record_result,
        golden_run=golden_run,
        binding=binding,
    )

    create_run_dir(run_dir)
    bundle_dir = run_dir / "bundle"
    bundle_dir.mkdir()
    shutil.copyfile(bundle_spec, bundle_dir / "migration-spec.md")
    bundled_golden = bundle_dir / "runs" / golden_run.name
    bundled_golden.parent.mkdir()
    copy_tree(golden_run, bundled_golden)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "spec_binding": binding,
        "operator_id": operator_id,
        "golden_run": f"runs/{golden_run.name}",
        "sample_record_result_sha256": sample_record_result_digest,
        "sample_files_sha256": sample_files_digest,
        "files": manifest_entries(bundle_dir),
    }
    manifest_path = bundle_dir / "manifest.json"
    write_json(manifest_path, manifest)
    manifest_digest = file_sha256(manifest_path)
    result = {
        "tool": "handoff_bundle.py",
        "action": "build",
        "spec_binding": binding,
        "passed": True,
        "bundle_path": str(bundle_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_digest,
        "file_count": len(manifest["files"]),
        "golden_run": manifest["golden_run"],
        "sample_record_result_sha256": sample_record_result_digest,
        "sample_files_sha256": sample_files_digest,
        "evidence": ["bundle/manifest.json", "handoff.log"],
        "summary": "Handoff Bundle and integrity manifest were built.",
    }
    write_json(run_dir / "result.json", result)
    (run_dir / "handoff.log").write_text(
        "\n".join(
            [
                "action=build",
                f"passed=true",
                f"golden_run={manifest['golden_run']}",
                f"file_count={len(manifest['files'])}",
                f"manifest_sha256={manifest_digest}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_gap_build(
    spec_path: Path,
    run_dir: Path,
    scan_result_path: Path,
    golden_runs: Sequence[Path],
    bundle_spec: Path,
    sample_record_results: Sequence[Path],
) -> None:
    contract = load_contract_data(spec_path)
    require_gap_bundle_contract(contract)
    binding = load_spec_binding(spec_path).as_result_dict()
    bundle_binding = load_spec_binding(bundle_spec).as_result_dict()
    if bundle_binding != binding:
        raise ToolError("bundle Spec Contract Data does not match current Contract Data")
    if bundle_spec.is_symlink() or not bundle_spec.is_file():
        raise ToolError("bundle Spec must be one regular file")
    require_regular_file(scan_result_path, "Scan Run result")
    if scan_result_path.is_symlink():
        raise ToolError("Scan Run result must not be a symbolic link")
    scan_result_path = scan_result_path.resolve()
    scan_result = read_json_object(scan_result_path, "Scan Run result")
    inventory = gap_inventory(
        scan_result,
        binding=binding,
        contract=contract,
    )
    expected_ids = [item["operator_id"] for item in inventory]
    item_by_id = {item["operator_id"]: item for item in inventory}

    if paths_overlap(run_dir, scan_result_path):
        raise ToolError("build run directory and Scan Run result must not overlap")
    resolved_goldens: Dict[str, Path] = {}
    capture_configs: Dict[str, Dict[str, Any]] = {}
    capture_states: Dict[str, Dict[str, Any]] = {}
    golden_names = set()
    for supplied in golden_runs:
        if supplied.is_symlink():
            raise ToolError("Golden Run must not be a symbolic link")
        golden_run = supplied.resolve()
        if paths_overlap(run_dir, golden_run):
            raise ToolError("build run directory and Golden Run must not overlap")
        state = read_json_object(
            golden_run / "capture-state.json",
            "Golden capture state",
        )
        operator_id = state.get("operator_id")
        if operator_id not in item_by_id:
            raise ToolError(
                f"Golden operator {operator_id!r} is not present "
                "in the Scan gap queue"
            )
        if operator_id in resolved_goldens:
            raise ToolError(
                f"duplicate Golden Run for Scan gap operator {operator_id!r}"
            )
        if (
            not golden_run.name
            or golden_run.name in {".", ".."}
            or golden_run.name in golden_names
        ):
            raise ToolError("Golden Run directory names must be unique")
        validate_gap_golden_run(
            golden_run,
            binding=binding,
            contract=contract,
            item=item_by_id[operator_id],
        )
        resolved_goldens[operator_id] = golden_run
        capture_configs[operator_id] = read_json_object(
            golden_run / "capture-config.json",
            "Golden capture config",
        )
        capture_states[operator_id] = read_json_object(
            golden_run / "capture-state.json",
            "Golden capture state",
        )
        golden_names.add(golden_run.name)
    missing_goldens = [
        operator_id
        for operator_id in expected_ids
        if operator_id not in resolved_goldens
    ]
    if missing_goldens:
        raise ToolError(
            "missing Golden Run for Scan gap operator: "
            + ", ".join(missing_goldens)
        )
    capture_configs = {
        operator_id: capture_configs[operator_id]
        for operator_id in expected_ids
    }
    capture_states = {
        operator_id: capture_states[operator_id]
        for operator_id in expected_ids
    }
    resolved_goldens = {
        operator_id: resolved_goldens[operator_id]
        for operator_id in expected_ids
    }
    (
        capture_session_path,
        _capture_session_config,
        capture_process_id,
    ) = load_live_gap_capture_session(
        binding=binding,
        contract=contract,
        inventory=inventory,
        scan_result_path=scan_result_path,
        capture_configs=capture_configs,
        capture_states=capture_states,
        golden_runs=resolved_goldens,
    )

    record_results: Dict[str, Path] = {}
    for supplied in sample_record_results:
        if supplied.is_symlink():
            raise ToolError(
                "sample record result must not be a symbolic link"
            )
        result_path = supplied.resolve()
        if paths_overlap(run_dir, result_path.parent):
            raise ToolError(
                "build run directory and sample record Run must not overlap"
            )
        record = read_json_object(result_path, "sample record result")
        operator_id = record.get("operator_id")
        if operator_id not in item_by_id:
            raise ToolError(
                f"sample record operator {operator_id!r} is not present "
                "in the Scan gap queue"
            )
        if operator_id in record_results:
            raise ToolError(
                f"duplicate sample record Run for Scan gap operator "
                f"{operator_id!r}"
            )
        record_results[operator_id] = result_path
    missing_records = [
        operator_id
        for operator_id in expected_ids
        if operator_id not in record_results
    ]
    if missing_records:
        raise ToolError(
            "missing sample record Run for Scan gap operator: "
            + ", ".join(missing_records)
        )

    operator_records = []
    for operator_id in expected_ids:
        golden_run = resolved_goldens[operator_id]
        (
            record_digest,
            sample_files_digest,
        ) = validate_gap_sample_record_result(
            record_results[operator_id],
            golden_run=golden_run,
            binding=binding,
            operator_id=operator_id,
        )
        operator_records.append(
            {
                "operator_id": operator_id,
                "golden_run": f"runs/{golden_run.name}",
                "sample_record_result_sha256": record_digest,
                "sample_files_sha256": sample_files_digest,
            }
        )

    create_run_dir(run_dir)
    bundle_dir = run_dir / "bundle"
    bundle_dir.mkdir()
    shutil.copyfile(bundle_spec, bundle_dir / "migration-spec.md")
    shutil.copyfile(scan_result_path, bundle_dir / "scan-result.json")
    shutil.copyfile(capture_session_path, bundle_dir / "capture-session.json")
    runs_dir = bundle_dir / "runs"
    runs_dir.mkdir()
    for operator_id in expected_ids:
        golden_run = resolved_goldens[operator_id]
        copy_tree(golden_run, runs_dir / golden_run.name)

    manifest = {
        "schema": GAP_MANIFEST_SCHEMA,
        "spec_binding": binding,
        "scan_result": {
            "path": "scan-result.json",
            "sha256": file_sha256(scan_result_path),
        },
        "capture_session": {
            "path": "capture-session.json",
            "sha256": file_sha256(capture_session_path),
            "process_id": capture_process_id,
        },
        "operators": operator_records,
        "files": manifest_entries(bundle_dir),
    }
    manifest_path = bundle_dir / "manifest.json"
    write_json(manifest_path, manifest)
    manifest_digest = file_sha256(manifest_path)
    result = {
        "tool": "handoff_bundle.py",
        "action": "build",
        "spec_binding": binding,
        "passed": True,
        "bundle_path": str(bundle_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_digest,
        "file_count": len(manifest["files"]),
        "operator_ids": expected_ids,
        "golden_runs": [
            item["golden_run"] for item in operator_records
        ],
        "scan_result_sha256": manifest["scan_result"]["sha256"],
        "capture_session_sha256": manifest["capture_session"]["sha256"],
        "capture_process_id": capture_process_id,
        "evidence": ["bundle/manifest.json", "handoff.log"],
        "summary": (
            "The Scan gap queue, every Golden Run, and the complete integrity "
            "manifest were built into one Handoff Bundle."
        ),
    }
    write_json(run_dir / "result.json", result)
    (run_dir / "handoff.log").write_text(
        "\n".join(
            [
                "action=build",
                "schema=handoff-manifest/v2",
                "passed=true",
                f"operator_count={len(expected_ids)}",
                f"file_count={len(manifest['files'])}",
                f"manifest_sha256={manifest_digest}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def validate_manifest_entry(entry: Any) -> str:
    if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
        raise ToolError("manifest file entry must contain path, size, and sha256")
    relative = entry["path"]
    if not isinstance(relative, str) or not relative:
        raise ToolError("manifest file path must be a non-empty string")
    pure_path = PurePosixPath(relative)
    if pure_path.is_absolute() or ".." in pure_path.parts or str(pure_path) != relative:
        raise ToolError(f"manifest path is unsafe: {relative!r}")
    size = entry["size"]
    digest = entry["sha256"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ToolError(f"manifest size is invalid for {relative}")
    if not is_sha256(digest):
        raise ToolError(f"manifest sha256 is invalid for {relative}")
    return relative


def validate_gap_bundle(
    bundle_dir: Path,
    binding: Dict[str, Any],
    contract: Dict[str, Any],
    manifest: Dict[str, Any],
) -> tuple[int, str]:
    if set(manifest) != {
        "schema",
        "spec_binding",
        "scan_result",
        "capture_session",
        "operators",
        "files",
    }:
        raise ToolError("Handoff manifest v2 fields have drifted")
    if manifest.get("spec_binding") != binding:
        raise ToolError("Handoff manifest spec_binding does not match Contract Data")
    bundled_spec = bundle_dir / "migration-spec.md"
    if load_spec_binding(bundled_spec).as_result_dict() != binding:
        raise ToolError("bundled Migration Spec does not match Contract Data")

    scan_record = manifest.get("scan_result")
    if (
        not isinstance(scan_record, dict)
        or set(scan_record) != {"path", "sha256"}
        or scan_record.get("path") != "scan-result.json"
        or not is_sha256(scan_record.get("sha256"))
    ):
        raise ToolError("Handoff manifest Scan Run binding is invalid")
    bundled_scan_path = bundle_dir / "scan-result.json"
    if file_sha256(bundled_scan_path) != scan_record["sha256"]:
        raise ToolError("bundled Scan Run SHA-256 differs from manifest")
    scan_result = read_json_object(bundled_scan_path, "bundled Scan Run result")
    inventory = gap_inventory(
        scan_result,
        binding=binding,
        contract=contract,
    )
    expected_ids = [item["operator_id"] for item in inventory]
    item_by_id = {item["operator_id"]: item for item in inventory}

    capture_session_record = manifest.get("capture_session")
    if (
        not isinstance(capture_session_record, dict)
        or set(capture_session_record) != {"path", "sha256", "process_id"}
        or capture_session_record.get("path") != "capture-session.json"
        or not is_sha256(capture_session_record.get("sha256"))
        or isinstance(capture_session_record.get("process_id"), bool)
        or not isinstance(capture_session_record.get("process_id"), int)
        or capture_session_record["process_id"] <= 0
    ):
        raise ToolError("Handoff manifest capture session binding is invalid")
    bundled_session_path = bundle_dir / "capture-session.json"
    require_regular_file(bundled_session_path, "bundled capture session")
    if file_sha256(bundled_session_path) != capture_session_record["sha256"]:
        raise ToolError("bundled capture session SHA-256 differs from manifest")
    bundled_session = read_json_object(
        bundled_session_path,
        "bundled capture session",
    )

    operator_records = manifest.get("operators")
    if not isinstance(operator_records, list) or not operator_records:
        raise ToolError("Handoff manifest operators must be a non-empty list")
    actual_ids = []
    golden_paths = []
    for record in operator_records:
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "operator_id",
                "golden_run",
                "sample_record_result_sha256",
                "sample_files_sha256",
            }
        ):
            raise ToolError("Handoff manifest operator entry has drifted")
        operator_id = record.get("operator_id")
        golden_relative = record.get("golden_run")
        golden_path = (
            PurePosixPath(golden_relative)
            if isinstance(golden_relative, str)
            else None
        )
        if (
            not isinstance(operator_id, str)
            or not operator_id
            or golden_path is None
            or golden_path.is_absolute()
            or len(golden_path.parts) != 2
            or golden_path.parts[0] != "runs"
            or golden_path.parts[1] in {"", ".", ".."}
            or str(golden_path) != golden_relative
        ):
            raise ToolError("Handoff manifest Golden Run entry is invalid")
        for field in (
            "sample_record_result_sha256",
            "sample_files_sha256",
        ):
            if not is_sha256(record.get(field)):
                raise ToolError(
                    f"Handoff manifest {field} for {operator_id!r} is invalid"
                )
        actual_ids.append(operator_id)
        golden_paths.append(golden_relative)
    if actual_ids != expected_ids:
        raise ToolError(
            "Handoff manifest operator list does not match the Scan gap queue"
        )
    if len(golden_paths) != len(set(golden_paths)):
        raise ToolError("Handoff manifest Golden Run paths must be unique")

    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ToolError("Handoff manifest files must be a non-empty list")
    paths = [validate_manifest_entry(entry) for entry in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ToolError("Handoff manifest file paths must be unique and sorted")
    golden_prefixes = tuple(path + "/" for path in golden_paths)
    not_permitted = [
        path
        for path in paths
        if path
        not in {
            "migration-spec.md",
            "scan-result.json",
            "capture-session.json",
        }
        and not path.startswith(golden_prefixes)
    ]
    if not_permitted:
        raise ToolError(
            "Handoff Bundle file is not permitted: "
            + ", ".join(not_permitted)
        )
    actual_entries = manifest_entries(bundle_dir)
    actual_paths = [entry["path"] for entry in actual_entries]
    missing = sorted(set(paths) - set(actual_paths))
    if missing:
        raise ToolError("Handoff Bundle missing file: " + ", ".join(missing))
    unexpected = sorted(set(actual_paths) - set(paths))
    if unexpected:
        raise ToolError("Handoff Bundle unexpected file: " + ", ".join(unexpected))
    if actual_paths != paths:
        raise ToolError("Handoff Bundle file order does not match manifest")
    for expected, actual in zip(entries, actual_entries):
        if expected["size"] != actual["size"]:
            raise ToolError(
                f"Handoff Bundle file size differs: {expected['path']}"
            )
        if expected["sha256"] != actual["sha256"]:
            raise ToolError(
                f"Handoff Bundle file sha256 differs: {expected['path']}"
            )

    capture_configs: Dict[str, Dict[str, Any]] = {}
    capture_states: Dict[str, Dict[str, Any]] = {}
    for record in operator_records:
        operator_id = record["operator_id"]
        bundled_golden = bundle_dir / record["golden_run"]
        validate_gap_golden_run(
            bundled_golden,
            binding=binding,
            contract=contract,
            item=item_by_id[operator_id],
        )
        if file_sha256(
            bundled_golden / "sample-files.json"
        ) != record["sample_files_sha256"]:
            raise ToolError(
                f"Handoff manifest sample_files_sha256 for "
                f"{operator_id!r} has drifted"
            )
        capture_configs[operator_id] = read_json_object(
            bundled_golden / "capture-config.json",
            "bundled Golden capture config",
        )
        capture_states[operator_id] = read_json_object(
            bundled_golden / "capture-state.json",
            "bundled Golden capture state",
        )
    process_id = validate_gap_capture_session_evidence(
        bundled_session,
        binding=binding,
        contract=contract,
        inventory=inventory,
        scan_result_digest=scan_record["sha256"],
        capture_configs=capture_configs,
        capture_states=capture_states,
    )
    if process_id != capture_session_record["process_id"]:
        raise ToolError(
            "Handoff manifest capture process does not match Golden evidence"
        )
    return len(entries), file_sha256(bundle_dir / "manifest.json")


def validate_bundle(
    bundle_dir: Path,
    binding: Dict[str, Any],
    contract: Dict[str, Any],
) -> tuple[int, str]:
    manifest_path = bundle_dir / "manifest.json"
    manifest = read_json_object(manifest_path, "Handoff manifest")
    if manifest.get("schema") == GAP_MANIFEST_SCHEMA:
        require_gap_bundle_contract(contract)
        return validate_gap_bundle(
            bundle_dir,
            binding,
            contract,
            manifest,
        )
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ToolError("Handoff manifest schema has drifted")
    reject_legacy_handoff_for_current_contract(contract)
    if manifest.get("spec_binding") != binding:
        raise ToolError("Handoff manifest spec_binding does not match Contract Data")
    if manifest.get("operator_id") != SWIGLU_CLAMP_OPERATOR_ID:
        raise ToolError("Handoff manifest operator_id has drifted")
    for field in (
        "sample_record_result_sha256",
        "sample_files_sha256",
    ):
        if not is_sha256(manifest.get(field)):
            raise ToolError(f"Handoff manifest {field} is invalid")
    bundled_spec = bundle_dir / "migration-spec.md"
    if load_spec_binding(bundled_spec).as_result_dict() != binding:
        raise ToolError("bundled Migration Spec does not match Contract Data")
    golden_relative = manifest.get("golden_run")
    golden_path = (
        PurePosixPath(golden_relative)
        if isinstance(golden_relative, str)
        else None
    )
    if (
        golden_path is None
        or golden_path.is_absolute()
        or len(golden_path.parts) != 2
        or golden_path.parts[0] != "runs"
        or golden_path.parts[1] in {"", ".", ".."}
        or str(golden_path) != golden_relative
    ):
        raise ToolError("Handoff manifest golden_run is invalid")

    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ToolError("Handoff manifest files must be a non-empty list")
    paths = [validate_manifest_entry(entry) for entry in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ToolError("Handoff manifest file paths must be unique and sorted")
    golden_prefix = golden_relative + "/"
    not_permitted = [
        path
        for path in paths
        if path != "migration-spec.md" and not path.startswith(golden_prefix)
    ]
    if not_permitted:
        raise ToolError(
            "Handoff Bundle file is not permitted: "
            + ", ".join(not_permitted)
        )
    actual_entries = manifest_entries(bundle_dir)
    actual_paths = [entry["path"] for entry in actual_entries]
    missing = sorted(set(paths) - set(actual_paths))
    if missing:
        raise ToolError("Handoff Bundle missing file: " + ", ".join(missing))
    unexpected = sorted(set(actual_paths) - set(paths))
    if unexpected:
        raise ToolError("Handoff Bundle unexpected file: " + ", ".join(unexpected))
    if actual_paths != paths:
        raise ToolError("Handoff Bundle file order does not match manifest")
    for expected, actual in zip(entries, actual_entries):
        if expected["size"] != actual["size"]:
            raise ToolError(f"Handoff Bundle file size differs: {expected['path']}")
        if expected["sha256"] != actual["sha256"]:
            raise ToolError(f"Handoff Bundle file sha256 differs: {expected['path']}")
    bundled_golden = bundle_dir / golden_relative
    validate_golden_run(bundled_golden, binding, contract)
    if file_sha256(bundled_golden / "sample-files.json") != manifest.get(
        "sample_files_sha256"
    ):
        raise ToolError(
            "Handoff manifest sample_files_sha256 has drifted"
        )
    return len(entries), file_sha256(manifest_path)


def paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


def run_verify(spec_path: Path, run_dir: Path, bundle_dir: Path) -> None:
    contract = load_contract_data(spec_path)
    binding = load_spec_binding(spec_path).as_result_dict()
    bundle_dir = bundle_dir.resolve()
    if paths_overlap(run_dir, bundle_dir):
        raise ToolError("verify run directory and Handoff Bundle must not overlap")
    try:
        file_count, manifest_digest = validate_bundle(
            bundle_dir,
            binding,
            contract,
        )
    except (SpecContractError, ToolError) as error:
        create_run_dir(run_dir)
        result = {
            "tool": "handoff_bundle.py",
            "action": "verify",
            "spec_binding": binding,
            "passed": False,
            "bundle_path": str(bundle_dir),
            "manifest_path": str((bundle_dir / "manifest.json").resolve()),
            "manifest_verified": False,
            "error": str(error),
            "evidence": ["handoff.log"],
            "summary": "Handoff Bundle integrity verification failed.",
        }
        write_json(run_dir / "result.json", result)
        (run_dir / "handoff.log").write_text(
            "\n".join(
                [
                    "action=verify",
                    "passed=false",
                    f"error={error}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return
    create_run_dir(run_dir)
    result = {
        "tool": "handoff_bundle.py",
        "action": "verify",
        "spec_binding": binding,
        "passed": True,
        "bundle_path": str(bundle_dir.resolve()),
        "manifest_path": str((bundle_dir / "manifest.json").resolve()),
        "manifest_sha256": manifest_digest,
        "manifest_verified": True,
        "file_count": file_count,
        "evidence": ["handoff.log"],
        "summary": "Handoff Bundle matches its complete integrity manifest.",
    }
    manifest = read_json_object(
        bundle_dir / "manifest.json",
        "Handoff manifest",
    )
    if manifest.get("schema") == GAP_MANIFEST_SCHEMA:
        result["operator_ids"] = [
            item["operator_id"] for item in manifest["operators"]
        ]
    write_json(run_dir / "result.json", result)
    (run_dir / "handoff.log").write_text(
        "\n".join(
            [
                "action=verify",
                "passed=true",
                f"file_count={file_count}",
                f"manifest_sha256={manifest_digest}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("record-samples", "build", "verify"),
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--golden-run", action="append", type=Path)
    parser.add_argument("--bundle-spec", type=Path)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--sample-record-result", action="append", type=Path)
    parser.add_argument("--scan-result", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "record-samples":
            if args.golden_run is None or len(args.golden_run) != 1:
                raise ToolError(
                    "exactly one --golden-run is required for record-samples"
                )
            if (
                args.bundle_spec is not None
                or args.bundle_dir is not None
                or args.sample_record_result is not None
            ):
                raise ToolError(
                    "--bundle-spec, --bundle-dir, and --sample-record-result "
                    "are not valid for record-samples"
                )
            if args.scan_result is None:
                run_record_samples(
                    args.spec,
                    args.run_dir,
                    args.golden_run[0],
                )
            else:
                run_record_gap_samples(
                    args.spec,
                    args.run_dir,
                    args.golden_run[0],
                    args.scan_result,
                )
        elif args.mode == "build":
            if (
                args.golden_run is None
                or args.bundle_spec is None
                or args.sample_record_result is None
            ):
                raise ToolError(
                    "--golden-run, --bundle-spec, and --sample-record-result "
                    "are required for build"
                )
            if args.bundle_dir is not None:
                raise ToolError("--bundle-dir is not valid for build")
            if args.scan_result is None:
                if (
                    len(args.golden_run) != 1
                    or len(args.sample_record_result) != 1
                ):
                    raise ToolError(
                        "legacy build accepts exactly one --golden-run and "
                        "one --sample-record-result"
                    )
                run_build(
                    args.spec,
                    args.run_dir,
                    args.golden_run[0],
                    args.bundle_spec,
                    args.sample_record_result[0],
                )
            else:
                run_gap_build(
                    args.spec,
                    args.run_dir,
                    args.scan_result,
                    args.golden_run,
                    args.bundle_spec,
                    args.sample_record_result,
                )
        else:
            if args.bundle_dir is None:
                raise ToolError("--bundle-dir is required for verify")
            if (
                args.golden_run is not None
                or args.bundle_spec is not None
                or args.sample_record_result is not None
                or args.scan_result is not None
            ):
                raise ToolError(
                    "--golden-run, --bundle-spec, --sample-record-result, "
                    "and --scan-result "
                    "are not valid for verify"
                )
            run_verify(args.spec, args.run_dir, args.bundle_dir)
    except (SpecContractError, ToolError, OSError) as error:
        print(f"handoff_bundle.py: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
