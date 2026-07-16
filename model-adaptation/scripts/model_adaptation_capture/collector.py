"""Collect no more than three portable, weight-free operator samples."""

import hashlib
import json
import math
import os
from pathlib import Path
import threading
from typing import Any, Callable, Dict

import torch

from .contracts import (
    CANDIDATE_SAMPLE_SCHEMA,
    CANDIDATE_SERIALIZATION,
    CONFIG_SCHEMA,
    OPERATOR_ID,
    STATE_SCHEMA,
)


class CaptureError(RuntimeError):
    """The current call cannot be saved as a trustworthy capture candidate."""


def _read_json_object(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CaptureError(f"cannot read capture config {path}: {error}") from error
    if not isinstance(value, dict):
        raise CaptureError("capture config must be one JSON object")
    return value


def _atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def signature_id(value: Dict[str, Any]) -> str:
    """Return the stable ID for a call signature without tensor values."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _is_capture_rank() -> bool:
    distributed = torch.distributed
    if not distributed.is_available() or not distributed.is_initialized():
        return True
    return distributed.get_rank() == 0


class CandidateCollector:
    """Capture distinct calls while preserving the wrapped function's output."""

    def __init__(self, config: Dict[str, Any]):
        self._validate_config(config)
        self.config = config
        self.run_dir = Path(config["run_dir"]).resolve()
        self.samples_dir = self.run_dir / "samples"
        self.state_path = self.run_dir / "capture-state.json"
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._state = self._load_or_create_state()

    @classmethod
    def from_config_path(cls, path: Path) -> "CandidateCollector":
        return cls(_read_json_object(path.resolve()))

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        if config.get("schema") != CONFIG_SCHEMA:
            raise CaptureError(f"capture config schema must be {CONFIG_SCHEMA!r}")
        if config.get("operator_id") != OPERATOR_ID:
            raise CaptureError("capture config operator is not the Demo operator")
        if config.get("model_path") != "target":
            raise CaptureError("capture config model_path must be target")
        if config.get("max_samples") != 3:
            raise CaptureError("capture config max_samples must be 3")
        if config.get("serialization") != CANDIDATE_SERIALIZATION:
            raise CaptureError(
                "capture config serialization must be "
                f"{CANDIDATE_SERIALIZATION!r}"
            )
        if config.get("capture_device_type") not in {"cpu", "cuda"}:
            raise CaptureError("capture_device_type must be cpu or cuda")
        if config.get("dtype") != "bfloat16":
            raise CaptureError("capture dtype must be bfloat16")
        if not isinstance(config.get("spec_binding"), dict):
            raise CaptureError("capture config is missing spec_binding")
        if not isinstance(config.get("run_dir"), str):
            raise CaptureError("capture config is missing run_dir")

    def _load_or_create_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            state = _read_json_object(self.state_path)
            if (
                state.get("schema") != STATE_SCHEMA
                or state.get("operator_id") != self.config["operator_id"]
            ):
                raise CaptureError("existing capture state does not match config")
            return state
        state = {
            "schema": STATE_SCHEMA,
            "operator_id": self.config["operator_id"],
            "status": "CAPTURING",
            "saved_sample_count": 0,
            "repeated_call_count": 0,
            "skipped_call_count": 0,
            "samples": [],
            "skipped_signatures": [],
        }
        _atomic_write_json(self.state_path, state)
        return state

    def _signature(
        self,
        gate_up: torch.Tensor,
        limit: float,
        execution_phase: str,
    ) -> Dict[str, Any]:
        return {
            "operator_id": self.config["operator_id"],
            "model_path": self.config["model_path"],
            "execution_phase": execution_phase,
            "inputs": {
                "gate_up": {
                    "kind": "tensor",
                    "shape": list(gate_up.shape),
                    "dtype": _dtype_name(gate_up.dtype),
                    "layout": str(gate_up.layout).removeprefix("torch."),
                    "stride": list(gate_up.stride()),
                }
            },
            "parameters": {"limit": float(limit)},
        }

    def _validate_new_sample(
        self,
        gate_up: torch.Tensor,
        limit: float,
        expected: torch.Tensor,
    ) -> None:
        if isinstance(gate_up, torch.nn.Parameter):
            raise CaptureError("nn.Parameter inputs are forbidden")
        if not isinstance(expected, torch.Tensor):
            raise CaptureError("operator output must be a Tensor")
        if isinstance(expected, torch.nn.Parameter):
            raise CaptureError("nn.Parameter outputs are forbidden")
        expected_device = self.config["capture_device_type"]
        if gate_up.device.type != expected_device or expected.device.type != expected_device:
            raise CaptureError(
                f"capture requires {expected_device} tensors, got "
                f"{gate_up.device.type} input and {expected.device.type} output"
            )
        if gate_up.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16:
            raise CaptureError("capture requires BF16 input and output tensors")
        if gate_up.layout != torch.strided or expected.layout != torch.strided:
            raise CaptureError("capture requires dense strided tensors")
        if isinstance(limit, bool) or not isinstance(limit, (int, float)):
            raise CaptureError("limit must be a numeric scalar")
        if not math.isfinite(float(limit)):
            raise CaptureError("limit must be finite")
        if not bool(torch.isfinite(gate_up).all().item()):
            raise CaptureError("capture input contains non-finite values")
        if not bool(torch.isfinite(expected).all().item()):
            raise CaptureError("capture output contains non-finite values")

    def capture(
        self,
        original: Callable[..., Any],
        gate_up: torch.Tensor,
        limit: float,
        *,
        execution_phase: str,
    ) -> Any:
        expected = original(gate_up, limit)
        if not _is_capture_rank():
            return expected
        if not isinstance(gate_up, torch.Tensor):
            raise CaptureError("gate_up must be a Tensor")
        if not isinstance(execution_phase, str) or not execution_phase:
            raise CaptureError("execution_phase must be a non-empty string")

        signature = self._signature(gate_up, limit, execution_phase)
        current_signature_id = signature_id(signature)
        with self._lock:
            for sample in self._state["samples"]:
                if sample["signature_id"] == current_signature_id:
                    sample["repeat_count"] += 1
                    self._state["repeated_call_count"] += 1
                    _atomic_write_json(self.state_path, self._state)
                    return expected

            if self._state["saved_sample_count"] >= self.config["max_samples"]:
                self._state["skipped_call_count"] += 1
                for skipped in self._state["skipped_signatures"]:
                    if skipped["signature_id"] == current_signature_id:
                        skipped["call_count"] += 1
                        break
                else:
                    self._state["skipped_signatures"].append(
                        {
                            "signature_id": current_signature_id,
                            "signature": signature,
                            "call_count": 1,
                        }
                    )
                _atomic_write_json(self.state_path, self._state)
                return expected

            self._validate_new_sample(gate_up, limit, expected)
            sample_name = f"{current_signature_id}.pt"
            sample_path = self.samples_dir / sample_name
            if sample_path.exists():
                raise CaptureError(f"sample already exists outside capture state: {sample_name}")
            payload = {
                "schema": CANDIDATE_SAMPLE_SCHEMA,
                "spec_binding": self.config["spec_binding"],
                "operator_id": self.config["operator_id"],
                "signature": signature,
                "inputs": {
                    "gate_up": gate_up.detach()
                    .to("cpu")
                    .clone(memory_format=torch.preserve_format)
                },
                "parameters": {"limit": float(limit)},
                "expected": expected.detach()
                .to("cpu")
                .clone(memory_format=torch.preserve_format),
            }
            temporary = sample_path.with_name(f".{sample_name}.{os.getpid()}.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, sample_path)
            self._state["samples"].append(
                {
                    "signature_id": current_signature_id,
                    "file": f"samples/{sample_name}",
                    "signature": signature,
                    "repeat_count": 0,
                }
            )
            self._state["saved_sample_count"] += 1
            _atomic_write_json(self.state_path, self._state)
        return expected
