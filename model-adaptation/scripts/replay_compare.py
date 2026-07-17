#!/usr/bin/env python3
"""Replay and compare captured samples without changing Migration Spec state."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from _lib.spec_contract import (
    SpecContractError,
    canonical_json_bytes,
    load_contract_data,
    load_spec_binding,
    uses_kernel_scan_contract,
)
from model_adaptation_capture.contracts import (
    OPERATOR_ID,
    REPLAY_CONFIG_ENV,
    REPLAY_CONFIG_SCHEMA,
    REPLAY_RESULT_SCHEMA,
    STATE_SCHEMA,
    checkpoint_metadata,
    shape_id_for,
)


class ToolError(RuntimeError):
    """The deterministic action could not form a trustworthy result."""


DEFAULT_SYNTHETIC_CASE = {
    "expected": {"spec_binding_smoke": "ready"},
    "actual": {"spec_binding_smoke": "ready"},
}


def require_loaded_model_mlp_adapter(contract: Dict[str, Any]) -> None:
    if uses_kernel_scan_contract(contract):
        raise ToolError(
            "kernel capture/replay adapter is not implemented for revision 5; "
            "do not fall back to the revision 4 loaded-model MLP replay"
        )


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


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


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


def _load_capture_state(
    golden_run: Path,
    expected_binding: Dict[str, Any],
    expected_checkpoint: Dict[str, str],
    tp_size: int,
    max_shapes: int,
    tp_rank: int,
    *,
    close_active_capture: bool,
    allow_failed: bool = False,
) -> tuple[Dict[str, Any], bool]:
    state = read_json_object(
        golden_run / "capture-state.json",
        "Golden capture state",
    )
    checks = {
        "schema": STATE_SCHEMA,
        "spec_binding": expected_binding,
        "operator_id": OPERATOR_ID,
        "tp_rank": tp_rank,
        "tensor_parallel_size": tp_size,
        "checkpoint": expected_checkpoint,
        "loaded_checkpoint": {
            "model_path": expected_checkpoint["model_path"],
            "revision": expected_checkpoint["revision"],
        },
    }
    for field, expected in checks.items():
        if state.get(field) != expected:
            raise ToolError(f"Golden capture state {field} has drifted")
    status = state.get("status")
    if status not in {"ACTIVE", "SEALED", "FAILED"}:
        raise ToolError("Golden capture state status has drifted")
    if not isinstance(state.get("capture_closed"), bool):
        raise ToolError("Golden capture state capture_closed has drifted")
    if status == "FAILED" and not allow_failed:
        raise ToolError("Golden capture state is FAILED")

    samples = state.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= max_shapes:
        raise ToolError(
            f"Golden capture state must contain one to {max_shapes} shapes"
        )
    if state.get("saved_shape_count") != len(samples):
        raise ToolError("Golden capture saved_shape_count has drifted")
    seen_shape_ids = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ToolError("Golden capture sample entry must be one object")
        shape_id = sample.get("shape_id")
        signature = sample.get("signature")
        if (
            not isinstance(shape_id, str)
            or not shape_id
            or not isinstance(signature, dict)
        ):
            raise ToolError("Golden capture sample shape metadata is invalid")
        if signature.get("operator_id") != OPERATOR_ID:
            raise ToolError("Golden capture sample operator has drifted")
        model_instance_path = signature.get("model_instance_path")
        if not isinstance(model_instance_path, str) or not model_instance_path:
            raise ToolError(
                "Golden capture sample model instance path has drifted"
            )
        try:
            expected_shape_id = shape_id_for(
                signature["inputs"]["x"]["shape"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ToolError(
                "Golden capture sample shape metadata has drifted"
            ) from error
        if shape_id != expected_shape_id:
            raise ToolError("Golden capture sample shape digest has drifted")
        if shape_id in seen_shape_ids:
            raise ToolError("Golden capture contains a duplicate shape")
        seen_shape_ids.add(shape_id)
        expected_file = f"samples/{shape_id}.pt"
        if sample.get("file") != expected_file:
            raise ToolError("Golden Sample path does not match its shape id")
        sample_path = (golden_run / sample["file"]).resolve()
        if not sample_path.is_relative_to(golden_run.resolve()):
            raise ToolError("Golden Sample path escapes the Golden Run")
        if not sample_path.is_file():
            raise ToolError(f"Golden Sample is missing: {sample['file']}")

    seal_golden_on_success = status == "ACTIVE"
    if status == "ACTIVE":
        if close_active_capture and not state["capture_closed"]:
            state["capture_closed"] = True
            atomic_write_json(golden_run / "capture-state.json", state)
        elif not close_active_capture and not state["capture_closed"]:
            raise ToolError("Golden capture must be closed before replay finalization")
    elif not state["capture_closed"]:
        raise ToolError(f"{status} Golden capture must remain closed")
    return state, seal_golden_on_success


def run_prepare_model_replay(
    spec_path: Path,
    run_dir: Path,
    golden_run: Path,
) -> None:
    binding = load_spec_binding(spec_path)
    contract = load_contract_data(spec_path)
    require_loaded_model_mlp_adapter(contract)
    tp_size = contract["runtime"]["tensor_parallel_size"]
    tp_rank = contract["operator_boundary"]["tp_rank"]
    max_shapes = contract["limits"]["max_shapes_per_operator"]
    try:
        checkpoint = checkpoint_metadata(
            contract["checkpoint"]["id"],
            contract["checkpoint"]["config_digest"],
        )
    except ValueError as error:
        raise ToolError(str(error)) from error
    state, seal_golden_on_success = _load_capture_state(
        golden_run.resolve(),
        binding.as_result_dict(),
        checkpoint,
        tp_size,
        max_shapes,
        tp_rank,
        close_active_capture=True,
    )
    shape_groups = []
    for sample in state["samples"]:
        signature = sample.get("signature")
        if not isinstance(signature, dict):
            raise ToolError("Golden capture signature must be one object")
        if signature.get("operator_id") != OPERATOR_ID:
            raise ToolError("Golden capture signature operator has drifted")
        model_instance_path = signature.get("model_instance_path")
        if not isinstance(model_instance_path, str) or not model_instance_path:
            raise ToolError("Golden capture signature is missing model instance path")
        shape_groups.append(
            {
                "shape_id": sample["shape_id"],
                "model_instance_path": model_instance_path,
            }
        )

    create_run_dir(run_dir)
    config = {
        "schema": REPLAY_CONFIG_SCHEMA,
        "spec_binding": binding.as_result_dict(),
        "operator_id": OPERATOR_ID,
        "activation_guard": contract["operator_boundary"]["activation_guard"],
        "tensor_parallel_size": tp_size,
        "tp_rank": tp_rank,
        "checkpoint": checkpoint,
        "weights_source": "loaded_checkpoint",
        "seal_golden_on_success": seal_golden_on_success,
        "golden_run": str(golden_run.resolve()),
        "run_dir": str(run_dir.resolve()),
        "shape_groups": shape_groups,
        "precision_gate": contract["precision_gate"],
    }
    config_path = run_dir / "replay-config.json"
    write_json(config_path, config)
    write_json(
        run_dir / "prepare.json",
        {
            "tool": "replay_compare.py",
            "action": "prepare-model-replay",
            "spec_binding": binding.as_result_dict(),
            "operator_id": OPERATOR_ID,
            "replay_status": "PREPARED",
            "tensor_parallel_size": tp_size,
            "tp_rank": tp_rank,
            "shape_count": len(shape_groups),
            "capture_closed": True,
            "seal_golden_on_success": seal_golden_on_success,
            "launch_environment": {
                REPLAY_CONFIG_ENV: str(config_path.resolve()),
            },
            "summary": (
                "Launch the Contract checkpoint and TP8 model; the plugin "
                "will verify the loaded model identity before replaying rank 0 "
                "x inside the matching Step3p5MLP."
            ),
        },
    )


def run_finalize_model_replay(spec_path: Path, run_dir: Path) -> None:
    binding = load_spec_binding(spec_path)
    contract = load_contract_data(spec_path)
    require_loaded_model_mlp_adapter(contract)
    try:
        checkpoint = checkpoint_metadata(
            contract["checkpoint"]["id"],
            contract["checkpoint"]["config_digest"],
        )
    except ValueError as error:
        raise ToolError(str(error)) from error
    config = read_json_object(run_dir / "replay-config.json", "replay config")
    if config.get("schema") != REPLAY_CONFIG_SCHEMA:
        raise ToolError("replay config schema has drifted")
    if config.get("spec_binding") != binding.as_result_dict():
        raise ToolError("replay config spec binding has drifted")
    if config.get("checkpoint") != checkpoint:
        raise ToolError("replay config checkpoint has drifted")
    if config.get("weights_source") != "loaded_checkpoint":
        raise ToolError("replay config weights source has drifted")
    if config.get("precision_gate") != contract["precision_gate"]:
        raise ToolError("replay config precision gate has drifted")
    seal_golden_on_success = config.get("seal_golden_on_success")
    if not isinstance(seal_golden_on_success, bool):
        raise ToolError("replay config Golden seal policy has drifted")
    tp_size = config.get("tensor_parallel_size")
    if tp_size != 8:
        raise ToolError("model replay finalization requires TP8")
    tp_rank = config.get("tp_rank")
    if tp_rank != 0:
        raise ToolError("model replay finalization requires tp_rank 0")
    shape_groups = config.get("shape_groups")
    if not isinstance(shape_groups, list) or not shape_groups:
        raise ToolError("replay config shape groups are invalid")
    configured_groups = []
    for item in shape_groups:
        if (
            not isinstance(item, dict)
            or set(item) != {"shape_id", "model_instance_path"}
            or not isinstance(item["shape_id"], str)
            or not item["shape_id"]
            or not isinstance(item["model_instance_path"], str)
            or not item["model_instance_path"]
        ):
            raise ToolError("replay config shape groups are invalid")
        configured_groups.append(
            (item["shape_id"], item["model_instance_path"])
        )
    if len(set(configured_groups)) != len(configured_groups):
        raise ToolError("replay config shape groups contain duplicate or invalid ids")
    expected_shapes = {shape_id for shape_id, _ in configured_groups}

    golden_run_value = config.get("golden_run")
    if not isinstance(golden_run_value, str) or not golden_run_value:
        raise ToolError("replay config Golden Run has drifted")
    golden_run = Path(golden_run_value).resolve()
    capture_state, state_requires_seal = _load_capture_state(
        golden_run,
        binding.as_result_dict(),
        checkpoint,
        tp_size,
        contract["limits"]["max_shapes_per_operator"],
        tp_rank,
        close_active_capture=False,
        allow_failed=True,
    )
    state_groups = [
        (
            sample["shape_id"],
            sample["signature"]["model_instance_path"],
        )
        for sample in capture_state["samples"]
    ]
    if configured_groups != state_groups:
        raise ToolError(
            "replay config shape groups do not match Golden capture state"
        )

    replay_result = read_json_object(
        run_dir / "replay-result.json",
        "model replay result",
    )
    checks = {
        "schema": REPLAY_RESULT_SCHEMA,
        "spec_binding": binding.as_result_dict(),
        "operator_id": OPERATOR_ID,
        "tp_rank": tp_rank,
        "tensor_parallel_size": tp_size,
        "checkpoint": checkpoint,
        "loaded_checkpoint": {
            "model_path": checkpoint["model_path"],
            "revision": checkpoint["revision"],
        },
    }
    for field, expected in checks.items():
        if replay_result.get(field) != expected:
            raise ToolError(f"model replay result {field} has drifted")
    checked_shapes = replay_result.get("checked_shapes")
    if not isinstance(checked_shapes, list):
        raise ToolError("model replay checked_shapes is invalid")
    checked_groups = []
    per_shape_passed = []
    for item in checked_shapes:
        if (
            not isinstance(item, dict)
            or set(item)
            != {"shape_id", "model_instance_path", "passed"}
            or not isinstance(item["shape_id"], str)
            or not item["shape_id"]
            or not isinstance(item["model_instance_path"], str)
            or not item["model_instance_path"]
            or not isinstance(item["passed"], bool)
        ):
            raise ToolError("model replay checked_shapes entry is invalid")
        checked_groups.append(
            (item["shape_id"], item["model_instance_path"])
        )
        per_shape_passed.append(item["passed"])
    if (
        set(checked_groups) != set(configured_groups)
        or len(checked_groups) != len(configured_groups)
    ):
        raise ToolError("model replay checked shape groups have drifted")

    passed = replay_result.get("passed")
    if not isinstance(passed, bool) or passed != all(per_shape_passed):
        raise ToolError("model replay per-shape pass results are inconsistent")
    result = {
        "tool": "replay_compare.py",
        "action": "finalize-model-replay",
        "spec_binding": binding.as_result_dict(),
        "operator_id": OPERATOR_ID,
        "passed": passed,
        "checked_shape_count": len(expected_shapes),
        "tp_rank": tp_rank,
        "checkpoint": checkpoint,
        "loaded_checkpoint": replay_result["loaded_checkpoint"],
        "replay_result_file": "replay-result.json",
        "summary": (
            "Every rank-0 MLP shape passed inside the loaded TP8 model."
            if passed
            else "At least one rank-0 MLP shape failed inside the loaded TP8 model."
        ),
    }
    result_path = run_dir / "result.json"
    if result_path.exists():
        existing_result = read_json_object(result_path, "model replay result")
        if existing_result != result:
            raise ToolError("existing model replay result has drifted")

    if seal_golden_on_success:
        target_status = "SEALED" if passed else "FAILED"
        self_replay = {
            "passed": passed,
            "checked_shape_count": len(expected_shapes),
            "checkpoint_id": checkpoint["id"],
            "loaded_checkpoint": replay_result["loaded_checkpoint"],
            "replay_result_sha256": hashlib.sha256(
                canonical_json_bytes(replay_result)
            ).hexdigest(),
        }
        if capture_state["status"] == "ACTIVE":
            if not state_requires_seal:
                raise ToolError("ACTIVE Golden capture state cannot be sealed")
            capture_state["status"] = target_status
            capture_state["self_replay"] = self_replay
            atomic_write_json(
                golden_run / "capture-state.json",
                capture_state,
            )
        elif (
            capture_state["status"] != target_status
            or capture_state.get("self_replay") != self_replay
        ):
            raise ToolError(
                "Golden capture terminal self-replay evidence has drifted"
            )
    elif capture_state["status"] != "SEALED" or state_requires_seal:
        raise ToolError("P800 replay requires an already SEALED Golden Run")

    if not result_path.exists():
        atomic_write_json(result_path, result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "synthetic",
            "validate-binding",
            "prepare-model-replay",
            "finalize-model-replay",
        ),
    )
    parser.add_argument("--case", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--golden-run", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "synthetic":
            if args.result is not None:
                raise ToolError("--result is not valid for synthetic mode")
            if args.golden_run is not None:
                raise ToolError("--golden-run is not valid for synthetic mode")
            run_synthetic(args.spec, args.run_dir, args.case)
        elif args.mode == "validate-binding":
            if args.result is None:
                raise ToolError("--result is required for validate-binding mode")
            if args.case is not None or args.golden_run is not None:
                raise ToolError(
                    "--case and --golden-run are not valid for validate-binding mode"
                )
            run_validate_binding(args.spec, args.run_dir, args.result)
        elif args.mode == "prepare-model-replay":
            if args.golden_run is None:
                raise ToolError(
                    "--golden-run is required for prepare-model-replay mode"
                )
            if args.case is not None or args.result is not None:
                raise ToolError(
                    "--case and --result are not valid for prepare-model-replay mode"
                )
            run_prepare_model_replay(args.spec, args.run_dir, args.golden_run)
        else:
            if any(
                value is not None
                for value in (args.case, args.result, args.golden_run)
            ):
                raise ToolError(
                    "--case, --result, and --golden-run are not valid for "
                    "finalize-model-replay mode"
                )
            run_finalize_model_replay(args.spec, args.run_dir)
    except (SpecContractError, ToolError, OSError) as error:
        print(f"replay_compare.py: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
