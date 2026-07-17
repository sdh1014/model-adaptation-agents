"""Read and bind the machine-readable Contract Data block in a Migration Spec."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict


CONTRACT_DATA_BEGIN = "<!-- CONTRACT-DATA: BEGIN -->"
CONTRACT_DATA_END = "<!-- CONTRACT-DATA: END -->"
COMMON_REQUIRED_PATHS = (
    "schema",
    "spec_id",
    "contract_revision",
    "model",
    "source.sglang_revision",
    "source.sglang_kunlun_revision",
    "checkpoint.id",
    "checkpoint.config_digest",
    "model_path.target_entry",
    "runtime.tensor_parallel_size",
    "runtime.dtype",
    "runtime.cuda_graph_backend_decode",
    "runtime.cuda_graph_backend_prefill",
    "limits.max_shapes_per_operator",
    "limits.max_repair_attempts",
    "precision_gate.comparator",
    "precision_gate.atol",
    "precision_gate.rtol",
    "precision_gate.require_exact_structure",
    "precision_gate.require_same_dtype",
    "precision_gate.require_finite",
    "precision_gate.check_stride",
)
LEGACY_OPERATOR_REQUIRED_PATHS = (
    "operator_boundary.id",
    "operator_boundary.activation_guard",
    "operator_boundary.tp_rank",
    "demo_input_mode",
)
KERNEL_SCAN_REQUIRED_PATHS = (
    "scan_scope.model_paths",
    "scan_scope.granularity",
    "scan_scope.input_modes",
    "sample_policy.capture_tp_rank",
    "sample_policy.save_inputs",
    "sample_policy.save_expected_outputs",
    "sample_policy.save_direct_parameter_tensors",
    "sample_policy.parameter_scope",
    "sample_policy.save_full_checkpoint",
    "sample_policy.save_module_state_dict",
)
MISSING = object()


class SpecContractError(ValueError):
    """The Migration Spec cannot produce a trustworthy Contract binding."""


@dataclass(frozen=True)
class SpecBinding:
    spec_id: str
    contract_revision: int
    contract_data_sha256: str

    def as_result_dict(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "contract_revision": self.contract_revision,
            "contract_data_sha256": self.contract_data_sha256,
        }

    def mismatched_fields(self, candidate: Any) -> list:
        candidate_binding = candidate if isinstance(candidate, dict) else {}
        return [
            field
            for field, expected in self.as_result_dict().items()
            if candidate_binding.get(field) != expected
        ]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def load_contract_data(spec_path: Path) -> Dict[str, Any]:
    try:
        spec_text = spec_path.read_text(encoding="utf-8")
    except OSError as error:
        raise SpecContractError(f"cannot read Migration Spec: {error}") from error

    if spec_text.count(CONTRACT_DATA_BEGIN) != 1 or spec_text.count(CONTRACT_DATA_END) != 1:
        raise SpecContractError("Migration Spec must contain exactly one Contract Data block")

    begin_marker = spec_text.index(CONTRACT_DATA_BEGIN)
    end_marker = spec_text.index(CONTRACT_DATA_END)
    if end_marker < begin_marker:
        raise SpecContractError("Contract Data markers are out of order")
    begin = begin_marker + len(CONTRACT_DATA_BEGIN)
    end = end_marker
    raw_contract = spec_text[begin:end].strip()
    try:
        contract_data = json.loads(raw_contract)
    except json.JSONDecodeError as error:
        raise SpecContractError(f"Contract Data is not valid JSON: {error.msg}") from error
    if not isinstance(contract_data, dict):
        raise SpecContractError("Contract Data must be one JSON object")
    return contract_data


def value_at_path(data: Dict[str, Any], dotted_path: str) -> Any:
    value: Any = data
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return MISSING
        value = value[part]
    return value


def is_placeholder(value: Any) -> bool:
    if value is MISSING or value is None:
        return True
    if not isinstance(value, str):
        return False
    normalized = value.strip()
    return (
        not normalized
        or normalized.upper() in {"TBD", "TODO", "PENDING"}
        or (normalized.startswith("<") and normalized.endswith(">"))
    )


def uses_kernel_scan_contract(contract_data: Dict[str, Any]) -> bool:
    return "scan_scope" in contract_data or "sample_policy" in contract_data


def require_approved_contract_data(contract_data: Dict[str, Any]) -> None:
    required_paths = COMMON_REQUIRED_PATHS + (
        KERNEL_SCAN_REQUIRED_PATHS
        if uses_kernel_scan_contract(contract_data)
        else LEGACY_OPERATOR_REQUIRED_PATHS
    )
    unresolved = [
        path
        for path in required_paths
        if is_placeholder(value_at_path(contract_data, path))
    ]
    if unresolved:
        raise SpecContractError(
            "Contract Data is not approved; missing required values: "
            + ", ".join(unresolved)
        )


def require_fixed_contract_data(contract_data: Dict[str, Any]) -> None:
    fixed_values = {
        "schema": "migration-spec/v0",
        "model": "Step-3.7-Flash",
        "model_path.target_entry": "Step3p7ForConditionalGeneration.forward",
        "model_path.draft_entry": None,
        "runtime.tensor_parallel_size": 8,
        "runtime.dtype": "bfloat16",
        "runtime.quantization": None,
        "runtime.speculative_algorithm": None,
        "runtime.cuda_graph_backend_decode": "disabled",
        "runtime.cuda_graph_backend_prefill": "disabled",
        "runtime.attention_backend": None,
        "limits.max_shapes_per_operator": 3,
        "limits.max_repair_attempts": 5,
        "precision_gate.comparator": "torch.testing.assert_close",
        "precision_gate.require_exact_structure": True,
        "precision_gate.require_same_dtype": True,
        "precision_gate.require_finite": True,
        "precision_gate.check_stride": False,
    }
    if uses_kernel_scan_contract(contract_data):
        for forbidden in ("operator_boundary", "demo_input_mode", "active_operator"):
            if forbidden in contract_data:
                raise SpecContractError(
                    f"Contract Data {forbidden} is forbidden in the kernel-scan "
                    "Contract; select active_operator after Scan"
                )
        fixed_values.update(
            {
                "scan_scope.model_paths": ["target"],
                "scan_scope.granularity": "kernel-call",
                "scan_scope.input_modes": ["text-only", "single-image"],
                "sample_policy.capture_tp_rank": 0,
                "sample_policy.save_inputs": True,
                "sample_policy.save_expected_outputs": True,
                "sample_policy.save_direct_parameter_tensors": True,
                "sample_policy.parameter_scope": (
                    "selected-kernel-call-current-rank"
                ),
                "sample_policy.save_full_checkpoint": False,
                "sample_policy.save_module_state_dict": False,
            }
        )
    else:
        fixed_values.update(
            {
                "operator_boundary.id": "Step3p5MLP.forward",
                "operator_boundary.activation_guard": "self.limit is not None",
                "operator_boundary.tp_rank": 0,
            }
        )
    for path, expected in fixed_values.items():
        actual = value_at_path(contract_data, path)
        if type(actual) is not type(expected) or actual != expected:
            raise SpecContractError(f"Contract Data {path} must be {expected!r}")

    for path in (
        "source.sglang_revision",
        "source.sglang_kunlun_revision",
        "checkpoint.id",
        "checkpoint.config_digest",
    ):
        value = value_at_path(contract_data, path)
        if not isinstance(value, str) or not value.strip():
            raise SpecContractError(f"Contract Data {path} must be a non-empty string")
    if not uses_kernel_scan_contract(contract_data):
        value = value_at_path(contract_data, "demo_input_mode")
        if not isinstance(value, str) or not value.strip():
            raise SpecContractError(
                "Contract Data demo_input_mode must be a non-empty string"
            )

    for path in ("precision_gate.atol", "precision_gate.rtol"):
        value = value_at_path(contract_data, path)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise SpecContractError(
                f"Contract Data {path} must be a finite non-negative number"
            )

    for path, expected in (
        ("precision_gate.atol", 0.01),
        ("precision_gate.rtol", 0.02),
    ):
        actual = value_at_path(contract_data, path)
        if actual != expected:
            raise SpecContractError(f"Contract Data {path} must be {expected!r}")


def load_spec_binding(spec_path: Path) -> SpecBinding:
    contract_data = load_contract_data(spec_path)
    require_approved_contract_data(contract_data)
    require_fixed_contract_data(contract_data)
    spec_id = contract_data.get("spec_id")
    contract_revision = contract_data.get("contract_revision")
    if not isinstance(spec_id, str) or not spec_id:
        raise SpecContractError("Contract Data spec_id must be a non-empty string")
    if (
        not isinstance(contract_revision, int)
        or isinstance(contract_revision, bool)
        or contract_revision < 1
    ):
        raise SpecContractError("Contract Data contract_revision must be a positive integer")

    digest = hashlib.sha256(canonical_json_bytes(contract_data)).hexdigest()
    return SpecBinding(
        spec_id=spec_id,
        contract_revision=contract_revision,
        contract_data_sha256=digest,
    )
