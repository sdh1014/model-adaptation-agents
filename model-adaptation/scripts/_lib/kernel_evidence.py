"""Validate metadata produced by the existing Kernel Call replay path."""

import hashlib
from typing import Any, Dict, Iterable

from _lib.spec_contract import canonical_json_bytes
from model_adaptation_capture.contracts import KERNEL_REPLAY_RESULT_SCHEMA


class KernelEvidenceError(ValueError):
    """Kernel replay metadata contradicts the expected invocation."""


def worker_result_sha256(result: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(result)).hexdigest()


def validate_kernel_worker_result(
    result: Dict[str, Any],
    *,
    spec_binding: Dict[str, Any],
    operator_id: str,
    execution_site: str,
    invocation_target: str,
    tp_rank: int,
    tensor_parallel_size: int,
    precision_gate: Dict[str, Any],
    expected_shape_ids: Iterable[str],
    sample_files_sha256: str,
) -> None:
    checks = {
        "schema": KERNEL_REPLAY_RESULT_SCHEMA,
        "spec_binding": spec_binding,
        "operator_id": operator_id,
        "execution_site": execution_site,
        "invocation_target": invocation_target,
        "tp_rank": tp_rank,
        "tensor_parallel_size": tensor_parallel_size,
        "precision_gate": precision_gate,
        "sample_files_sha256": sample_files_sha256,
        "actual_tensors_saved": False,
    }
    for field, expected in checks.items():
        if result.get(field) != expected:
            raise KernelEvidenceError(
                f"kernel replay result {field} has drifted"
            )

    checked_shapes = result.get("checked_shapes")
    if not isinstance(checked_shapes, list) or not checked_shapes:
        raise KernelEvidenceError("kernel replay checked_shapes is invalid")
    if result.get("checked_shape_count") != len(checked_shapes):
        raise KernelEvidenceError(
            "kernel replay checked shape count has drifted"
        )

    failed = 0
    checked_shape_ids = []
    seen = set()
    for item in checked_shapes:
        if (
            not isinstance(item, dict)
            or set(item) != {"shape_id", "passed"}
            or not isinstance(item["shape_id"], str)
            or not item["shape_id"]
            or not isinstance(item["passed"], bool)
            or item["shape_id"] in seen
        ):
            raise KernelEvidenceError(
                "kernel replay checked shape entry is invalid"
            )
        seen.add(item["shape_id"])
        checked_shape_ids.append(item["shape_id"])
        failed += not item["passed"]

    expected = list(expected_shape_ids)
    if checked_shape_ids != expected:
        raise KernelEvidenceError(
            "kernel replay checked shapes do not match the Golden Run"
        )
    if result.get("failed_shape_count") != failed:
        raise KernelEvidenceError(
            "kernel replay failed shape count has drifted"
        )
    passed = result.get("passed")
    if not isinstance(passed, bool) or passed != (failed == 0):
        raise KernelEvidenceError(
            "kernel replay pass result is inconsistent"
        )
    errors = result.get("errors")
    if not isinstance(errors, list) or len(errors) != failed:
        raise KernelEvidenceError(
            "kernel replay error evidence is inconsistent"
        )
