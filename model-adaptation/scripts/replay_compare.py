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
from model_adaptation_capture.contracts import (
    OPERATOR_ID,
    REPLAY_CONFIG_ENV,
    REPLAY_CONFIG_SCHEMA,
    REPLAY_RESULT_SCHEMA,
    STATE_SCHEMA,
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
    tp_size: int,
    max_shapes: int,
    tp_rank: int,
) -> Dict[str, Any]:
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
        "status": "SEALED",
    }
    for field, expected in checks.items():
        if state.get(field) != expected:
            raise ToolError(f"Golden capture state {field} has drifted")
    samples = state.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= max_shapes:
        raise ToolError(
            f"Golden capture state must contain one to {max_shapes} shapes"
        )
    for sample in samples:
        sample_path = (golden_run / sample["file"]).resolve()
        if not sample_path.is_relative_to(golden_run.resolve()):
            raise ToolError("Golden Sample path escapes the Golden Run")
        if not sample_path.is_file():
            raise ToolError(f"Golden Sample is missing: {sample['file']}")
    return state


def run_prepare_model_replay(
    spec_path: Path,
    run_dir: Path,
    golden_run: Path,
) -> None:
    binding = load_spec_binding(spec_path)
    contract = read_json_object_from_contract(spec_path)
    tp_size = contract["runtime"]["tensor_parallel_size"]
    tp_rank = contract["operator_boundary"]["tp_rank"]
    max_shapes = contract["limits"]["max_shapes_per_operator"]
    state = _load_capture_state(
        golden_run.resolve(),
        binding.as_result_dict(),
        tp_size,
        max_shapes,
        tp_rank,
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
                "shape_id": sample["signature_id"],
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
        "checkpoint_id": contract["checkpoint"]["id"],
        "weights_source": "loaded_checkpoint",
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
            "launch_environment": {
                REPLAY_CONFIG_ENV: str(config_path.resolve()),
            },
            "summary": (
                "Launch the same checkpoint and TP8 model on P800; the plugin "
                "will replay rank 0 x inside the matching Step3p5MLP."
            ),
        },
    )


def read_json_object_from_contract(spec_path: Path) -> Dict[str, Any]:
    from _lib.spec_contract import load_contract_data

    return load_contract_data(spec_path)


def run_finalize_model_replay(spec_path: Path, run_dir: Path) -> None:
    if (run_dir / "result.json").exists():
        raise ToolError("model replay result already exists")
    binding = load_spec_binding(spec_path)
    config = read_json_object(run_dir / "replay-config.json", "replay config")
    if config.get("schema") != REPLAY_CONFIG_SCHEMA:
        raise ToolError("replay config schema has drifted")
    if config.get("spec_binding") != binding.as_result_dict():
        raise ToolError("replay config spec binding has drifted")
    tp_size = config.get("tensor_parallel_size")
    if tp_size != 8:
        raise ToolError("model replay finalization requires TP8")
    tp_rank = config.get("tp_rank")
    if tp_rank != 0:
        raise ToolError("model replay finalization requires tp_rank 0")
    expected_shapes = {
        item["shape_id"] for item in config.get("shape_groups", [])
    }

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
    }
    for field, expected in checks.items():
        if replay_result.get(field) != expected:
            raise ToolError(f"model replay result {field} has drifted")
    checked_shapes = replay_result.get("checked_shapes")
    if not isinstance(checked_shapes, list):
        raise ToolError("model replay checked_shapes is invalid")
    if {item.get("shape_id") for item in checked_shapes} != expected_shapes:
        raise ToolError("model replay did not check every shape")

    passed = replay_result.get("passed") is True
    write_json(
        run_dir / "result.json",
        {
            "tool": "replay_compare.py",
            "action": "finalize-model-replay",
            "spec_binding": binding.as_result_dict(),
            "operator_id": OPERATOR_ID,
            "passed": passed,
            "checked_shape_count": len(expected_shapes),
            "tp_rank": tp_rank,
            "replay_result_file": "replay-result.json",
            "summary": (
                "Every rank-0 MLP shape passed inside the loaded TP8 model."
                if passed
                else "At least one rank-0 MLP shape failed inside the loaded TP8 model."
            ),
        },
    )


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
