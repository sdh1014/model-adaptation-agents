"""Capture the selected rank-0 Kernel Call without model weights."""

import json
import math
import os
from pathlib import Path
import threading
from typing import Any, Dict, Optional

import torch

from .contracts import (
    CONFIG_SCHEMA,
    FIXED_PRECISION_GATE,
    KERNEL_CALL_SAMPLE_SCHEMA,
    KERNEL_CALL_SERIALIZATION,
    KERNEL_CALL_STATE_SCHEMA,
    KUNLUN_SWIGLU_TARGET,
    SWIGLU_CLAMP_HOOK_TARGET,
    SWIGLU_CLAMP_OPERATOR_ID,
    VALIDATION_TP_RANK,
    shape_id_for,
)


class KernelCallError(RuntimeError):
    """A Kernel Call sample cannot be trusted or replayed."""


def _read_json_object(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise KernelCallError(f"cannot read capture config {path}: {error}") from error
    if not isinstance(value, dict):
        raise KernelCallError("capture config must be one JSON object")
    return value


def _atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


class KernelCallCollector:
    """Save at most three distinct rank-0 input shapes."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        tp_rank: int,
        tp_size: int,
        loaded_checkpoint: Optional[Dict[str, str]] = None,
    ):
        self._validate_config(config)
        if tp_size != config["tensor_parallel_size"]:
            raise KernelCallError(
                "runtime tensor parallel size does not match capture config"
            )
        if isinstance(tp_rank, bool) or not isinstance(tp_rank, int):
            raise KernelCallError("tp_rank must be an integer")
        if tp_rank < 0 or tp_rank >= tp_size:
            raise KernelCallError("tp_rank is outside the configured TP group")

        self.config = config
        self.loaded_checkpoint = self._validate_loaded_checkpoint(
            config,
            loaded_checkpoint,
        )
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.is_capture_rank = tp_rank == config["tp_rank"]
        self.run_dir = Path(config["run_dir"]).resolve()
        self.samples_dir = self.run_dir / "samples"
        self.state_path = self.run_dir / "capture-state.json"
        self._lock = threading.Lock()
        self._state: Optional[Dict[str, Any]] = None
        if self.is_capture_rank:
            self.samples_dir.mkdir(parents=True, exist_ok=True)
            self._state = self._load_or_create_state()

    @classmethod
    def from_config_path(
        cls,
        path: Path,
        *,
        tp_rank: int,
        tp_size: int,
        loaded_checkpoint: Optional[Dict[str, str]] = None,
    ) -> "KernelCallCollector":
        return cls(
            _read_json_object(path.resolve()),
            tp_rank=tp_rank,
            tp_size=tp_size,
            loaded_checkpoint=loaded_checkpoint,
        )

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        checks = {
            "schema": CONFIG_SCHEMA,
            "operator_id": SWIGLU_CLAMP_OPERATOR_ID,
            "hook_target": SWIGLU_CLAMP_HOOK_TARGET,
            "model_path": "target",
            "max_shapes": 3,
            "tensor_parallel_size": 8,
            "tp_rank": VALIDATION_TP_RANK,
            "serialization": KERNEL_CALL_SERIALIZATION,
            "dtype": "bfloat16",
            "precision_gate": FIXED_PRECISION_GATE,
            "boundary": {
                "inputs": ["x"],
                "parameters": [],
                "non_tensor_args": ["gemm1_limit"],
                "outputs": ["output"],
            },
            "sample_fields": {
                "inputs": ["x"],
                "parameters": [],
                "non_tensor_args": ["gemm1_limit"],
                "outputs": ["output"],
            },
            "replay": {
                "mode": "standalone-kernel-call",
                "cuda_target": SWIGLU_CLAMP_OPERATOR_ID,
                "p800_target": KUNLUN_SWIGLU_TARGET,
                "weights_in_golden_sample": False,
            },
        }
        for field, expected in checks.items():
            if config.get(field) != expected:
                raise KernelCallError(f"capture config {field} has drifted")
        if config.get("capture_device_type") not in {"cpu", "cuda"}:
            raise KernelCallError("capture_device_type must be cpu or cuda")
        if not isinstance(config.get("spec_binding"), dict):
            raise KernelCallError("capture config is missing spec_binding")
        if not isinstance(config.get("run_dir"), str):
            raise KernelCallError("capture config is missing run_dir")
        checkpoint = config.get("checkpoint")
        if (
            not isinstance(checkpoint, dict)
            or set(checkpoint)
            != {"id", "model_path", "revision", "config_digest"}
            or any(
                not isinstance(checkpoint[field], str) or not checkpoint[field]
                for field in checkpoint
            )
            or checkpoint["id"]
            != f"{checkpoint['model_path']}@{checkpoint['revision']}"
        ):
            raise KernelCallError("capture config checkpoint identity is invalid")

    @staticmethod
    def _validate_loaded_checkpoint(
        config: Dict[str, Any],
        loaded_checkpoint: Optional[Dict[str, str]],
    ) -> Optional[Dict[str, str]]:
        if "preflight_tp_context" in config:
            if loaded_checkpoint is not None:
                raise KernelCallError(
                    "preflight capture must not claim a loaded checkpoint"
                )
            return None
        if (
            not isinstance(loaded_checkpoint, dict)
            or set(loaded_checkpoint) != {"model_path", "revision"}
            or any(
                not isinstance(loaded_checkpoint[field], str)
                or not loaded_checkpoint[field]
                for field in loaded_checkpoint
            )
        ):
            raise KernelCallError(
                "loaded checkpoint identity is required for formal capture"
            )
        for field in ("model_path", "revision"):
            if loaded_checkpoint[field] != config["checkpoint"][field]:
                raise KernelCallError(
                    f"loaded checkpoint {field} does not match capture config"
                )
        return loaded_checkpoint

    def _load_or_create_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            state = _read_json_object(self.state_path)
            checks = {
                "schema": KERNEL_CALL_STATE_SCHEMA,
                "spec_binding": self.config["spec_binding"],
                "operator_id": self.config["operator_id"],
                "tp_rank": self.config["tp_rank"],
                "tensor_parallel_size": self.tp_size,
                "checkpoint": self.config["checkpoint"],
                "loaded_checkpoint": self.loaded_checkpoint,
            }
            if any(state.get(field) != expected for field, expected in checks.items()):
                raise KernelCallError("existing capture state does not match config")
            if (
                state.get("status") not in {"ACTIVE", "SEALED", "FAILED"}
                or not isinstance(state.get("capture_closed"), bool)
            ):
                raise KernelCallError("existing capture state has drifted")
            return state

        state = {
            "schema": KERNEL_CALL_STATE_SCHEMA,
            "spec_binding": self.config["spec_binding"],
            "operator_id": self.config["operator_id"],
            "tp_rank": self.config["tp_rank"],
            "tensor_parallel_size": self.tp_size,
            "checkpoint": self.config["checkpoint"],
            "loaded_checkpoint": self.loaded_checkpoint,
            "status": "ACTIVE",
            "capture_closed": False,
            "saved_shape_count": 0,
            "repeated_call_count": 0,
            "skipped_call_count": 0,
            "samples": [],
            "skipped_signatures": [],
        }
        _atomic_write_json(self.state_path, state)
        return state

    def _signature(
        self,
        x: torch.Tensor,
        gemm1_limit: float,
    ) -> Dict[str, Any]:
        return {
            "operator_id": self.config["operator_id"],
            "model_path": self.config["model_path"],
            "tensor_parallel_size": self.tp_size,
            "tp_rank": self.config["tp_rank"],
            "inputs": {
                "x": {
                    "kind": "tensor",
                    "shape": list(x.shape),
                    "dtype": _dtype_name(x.dtype),
                    "layout": str(x.layout).removeprefix("torch."),
                    "stride": list(x.stride()),
                }
            },
            "parameters": {},
            "non_tensor_args": {"gemm1_limit": float(gemm1_limit)},
        }

    def _validate_call(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        gemm1_limit: float,
    ) -> None:
        if not isinstance(x, torch.Tensor):
            raise KernelCallError("x must be a Tensor")
        if isinstance(x, torch.nn.Parameter):
            raise KernelCallError("Parameter inputs are forbidden")
        if not isinstance(output, torch.Tensor):
            raise KernelCallError("operator output must be a Tensor")
        if isinstance(output, torch.nn.Parameter):
            raise KernelCallError("Parameter outputs are forbidden")
        expected_device = self.config["capture_device_type"]
        if x.device.type != expected_device or output.device.type != expected_device:
            raise KernelCallError(
                f"capture requires {expected_device} tensors, got "
                f"{x.device.type} input and {output.device.type} output"
            )
        if x.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
            raise KernelCallError("capture requires BF16 input and output tensors")
        if x.layout != torch.strided or output.layout != torch.strided:
            raise KernelCallError("capture requires dense strided tensors")
        if (
            x.ndim < 1
            or x.shape[-1] % 2 != 0
            or output.shape != x.shape[:-1] + (x.shape[-1] // 2,)
        ):
            raise KernelCallError("SwiGLU output shape does not match input")
        if isinstance(gemm1_limit, bool) or not isinstance(
            gemm1_limit,
            (int, float),
        ):
            raise KernelCallError("gemm1_limit must be a numeric scalar")
        if not math.isfinite(float(gemm1_limit)):
            raise KernelCallError("gemm1_limit must be finite")
        if not bool(torch.isfinite(x).all().item()):
            raise KernelCallError("capture input contains non-finite values")
        if not bool(torch.isfinite(output).all().item()):
            raise KernelCallError("capture output contains non-finite values")

    def record(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        *,
        gemm1_limit: float,
    ) -> None:
        if not self.is_capture_rank:
            return
        self._validate_call(x, output, gemm1_limit)
        signature = self._signature(x, gemm1_limit)
        current_shape_id = shape_id_for(x.shape)
        if self._state is None:
            raise KernelCallError("capture state was not initialized for rank 0")

        with self._lock:
            self._state = self._load_or_create_state()
            if self._state["status"] != "ACTIVE" or self._state["capture_closed"]:
                raise KernelCallError("capture is closed and cannot accept more samples")

            for sample in self._state["samples"]:
                if sample["shape_id"] == current_shape_id:
                    sample["repeat_count"] += 1
                    self._state["repeated_call_count"] += 1
                    _atomic_write_json(self.state_path, self._state)
                    return

            if self._state["saved_shape_count"] >= self.config["max_shapes"]:
                self._state["skipped_call_count"] += 1
                for skipped in self._state["skipped_signatures"]:
                    if skipped["shape_id"] == current_shape_id:
                        skipped["call_count"] += 1
                        break
                else:
                    self._state["skipped_signatures"].append(
                        {
                            "shape_id": current_shape_id,
                            "signature": signature,
                            "call_count": 1,
                        }
                    )
                _atomic_write_json(self.state_path, self._state)
                return

            sample_name = f"{current_shape_id}.pt"
            sample_path = self.samples_dir / sample_name
            if sample_path.exists():
                raise KernelCallError(
                    f"sample already exists outside capture state: {sample_name}"
                )
            payload = {
                "schema": KERNEL_CALL_SAMPLE_SCHEMA,
                "spec_binding": self.config["spec_binding"],
                "operator_id": self.config["operator_id"],
                "tp_rank": self.tp_rank,
                "tensor_parallel_size": self.tp_size,
                "signature": signature,
                "inputs": {
                    "x": x.detach()
                    .to("cpu")
                    .clone(memory_format=torch.preserve_format)
                },
                "parameters": {},
                "non_tensor_args": {"gemm1_limit": float(gemm1_limit)},
                "outputs": {
                    "output": output.detach()
                    .to("cpu")
                    .clone(memory_format=torch.preserve_format)
                },
            }
            temporary = sample_path.with_name(f".{sample_name}.{os.getpid()}.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, sample_path)
            self._state["samples"].append(
                {
                    "shape_id": current_shape_id,
                    "file": str(sample_path.relative_to(self.run_dir)),
                    "signature": signature,
                    "repeat_count": 0,
                }
            )
            self._state["saved_shape_count"] += 1
            _atomic_write_json(self.state_path, self._state)


class PlannedKernelCallCollector:
    """Capture one entry from a validated multi-operator session plan."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        tp_rank: int,
        tp_size: int,
        loaded_checkpoint: Optional[Dict[str, str]] = None,
    ):
        self._validate_config(config)
        if tp_size != config["tensor_parallel_size"]:
            raise KernelCallError(
                "runtime tensor parallel size does not match capture config"
            )
        if (
            isinstance(tp_rank, bool)
            or not isinstance(tp_rank, int)
            or tp_rank < 0
            or tp_rank >= tp_size
        ):
            raise KernelCallError("tp_rank is outside the configured TP group")
        self.config = config
        self.loaded_checkpoint = KernelCallCollector._validate_loaded_checkpoint(
            config,
            loaded_checkpoint,
        )
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.is_capture_rank = tp_rank == config["tp_rank"]
        self.run_dir = Path(config["run_dir"]).resolve()
        self.samples_dir = self.run_dir / "samples"
        self.state_path = self.run_dir / "capture-state.json"
        self._lock = threading.Lock()
        self._state: Optional[Dict[str, Any]] = None
        if self.is_capture_rank:
            self.samples_dir.mkdir(parents=True, exist_ok=True)
            self._state = self._load_or_create_state()

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        if config.get("schema") != CONFIG_SCHEMA:
            raise KernelCallError(f"capture config schema must be {CONFIG_SCHEMA!r}")
        checks = {
            "model_path": "target",
            "max_shapes": 3,
            "tensor_parallel_size": 8,
            "tp_rank": VALIDATION_TP_RANK,
            "serialization": KERNEL_CALL_SERIALIZATION,
            "dtype": "bfloat16",
            "precision_gate": FIXED_PRECISION_GATE,
        }
        for field, expected in checks.items():
            if config.get(field) != expected:
                raise KernelCallError(f"capture config {field} has drifted")
        for field in (
            "session_config",
            "adapter",
            "operator_id",
            "activation_guard",
            "run_dir",
            "hook_target",
        ):
            if not isinstance(config.get(field), str) or not config[field]:
                raise KernelCallError(f"capture config is missing {field}")
        if config.get("capture_device_type") not in {"cpu", "cuda"}:
            raise KernelCallError("capture_device_type must be cpu or cuda")
        if not isinstance(config.get("spec_binding"), dict):
            raise KernelCallError("capture config is missing spec_binding")
        boundary = config.get("boundary")
        sample_fields = config.get("sample_fields")
        if (
            not isinstance(boundary, dict)
            or not isinstance(sample_fields, dict)
            or boundary != sample_fields
            or set(boundary)
            != {"inputs", "parameters", "non_tensor_args", "outputs"}
        ):
            raise KernelCallError(
                "capture boundary and saved sample fields must match"
            )
        for names in boundary.values():
            if (
                not isinstance(names, list)
                or not all(isinstance(name, str) and name for name in names)
                or len(names) != len(set(names))
            ):
                raise KernelCallError("capture field names are invalid")
        if not boundary["inputs"] or not boundary["outputs"]:
            raise KernelCallError("capture requires inputs and outputs")
        replay = config.get("replay")
        p800_target = replay.get("p800_target") if isinstance(replay, dict) else None
        if (
            not isinstance(replay, dict)
            or replay.get("mode") != "standalone-kernel-call"
            or replay.get("cuda_target") != config["hook_target"]
            or (
                p800_target is not None
                and (not isinstance(p800_target, str) or not p800_target)
            )
            or replay.get("weights_in_golden_sample") is not False
        ):
            raise KernelCallError("capture replay boundary has drifted")
        checkpoint = config.get("checkpoint")
        if (
            not isinstance(checkpoint, dict)
            or set(checkpoint)
            != {"id", "model_path", "revision", "config_digest"}
            or any(
                not isinstance(checkpoint[field], str) or not checkpoint[field]
                for field in checkpoint
            )
            or checkpoint["id"]
            != f"{checkpoint['model_path']}@{checkpoint['revision']}"
        ):
            raise KernelCallError("capture config checkpoint identity is invalid")

    def _load_or_create_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            state = _read_json_object(self.state_path)
            checks = {
                "schema": KERNEL_CALL_STATE_SCHEMA,
                "spec_binding": self.config["spec_binding"],
                "operator_id": self.config["operator_id"],
                "tp_rank": self.config["tp_rank"],
                "tensor_parallel_size": self.tp_size,
                "checkpoint": self.config["checkpoint"],
                "loaded_checkpoint": self.loaded_checkpoint,
                "capture_session_config": self.config["session_config"],
                "capture_process_id": os.getpid(),
            }
            if any(state.get(field) != expected for field, expected in checks.items()):
                raise KernelCallError("existing capture state does not match config")
            if (
                state.get("status") not in {"ACTIVE", "SEALED", "FAILED"}
                or not isinstance(state.get("capture_closed"), bool)
            ):
                raise KernelCallError("existing capture state has drifted")
            return state
        state = {
            "schema": KERNEL_CALL_STATE_SCHEMA,
            "spec_binding": self.config["spec_binding"],
            "operator_id": self.config["operator_id"],
            "tp_rank": self.config["tp_rank"],
            "tensor_parallel_size": self.tp_size,
            "checkpoint": self.config["checkpoint"],
            "loaded_checkpoint": self.loaded_checkpoint,
            "capture_session_config": self.config["session_config"],
            "capture_process_id": os.getpid(),
            "status": "ACTIVE",
            "capture_closed": False,
            "saved_shape_count": 0,
            "repeated_call_count": 0,
            "skipped_call_count": 0,
            "samples": [],
            "skipped_signatures": [],
        }
        _atomic_write_json(self.state_path, state)
        return state

    @staticmethod
    def _tensor_metadata(value: torch.Tensor) -> Dict[str, Any]:
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": _dtype_name(value.dtype),
            "layout": str(value.layout).removeprefix("torch."),
            "stride": list(value.stride()),
        }

    def _validate_tensors(
        self,
        values: Dict[str, Any],
        *,
        allow_parameter: bool,
        label: str,
    ) -> None:
        expected_device = self.config["capture_device_type"]
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise KernelCallError(f"{label} {name} must be a Tensor")
            if not allow_parameter and isinstance(value, torch.nn.Parameter):
                raise KernelCallError(f"{label} {name} must not be an nn.Parameter")
            if value.device.type != expected_device:
                raise KernelCallError(
                    f"{label} {name} must be on {expected_device}, "
                    f"got {value.device.type}"
                )
            if value.layout != torch.strided:
                raise KernelCallError(f"{label} {name} must be a strided Tensor")
            if not bool(torch.isfinite(value).all().item()):
                raise KernelCallError(f"{label} {name} contains non-finite values")

    @staticmethod
    def _validate_non_tensor_args(values: Dict[str, Any]) -> None:
        for name, value in values.items():
            if value is None or isinstance(value, (bool, str, int)):
                continue
            if isinstance(value, float) and math.isfinite(value):
                continue
            raise KernelCallError(
                f"non-Tensor argument {name} must be a finite JSON scalar or null"
            )

    def record_call(
        self,
        *,
        inputs: Dict[str, Any],
        parameters: Dict[str, Any],
        non_tensor_args: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> None:
        if not self.is_capture_rank:
            return
        groups = {
            "inputs": inputs,
            "parameters": parameters,
            "non_tensor_args": non_tensor_args,
            "outputs": outputs,
        }
        for field, values in groups.items():
            if (
                not isinstance(values, dict)
                or list(values) != self.config["sample_fields"][field]
            ):
                raise KernelCallError(
                    f"recorded {field} do not match the capture plan"
                )
        self._validate_tensors(inputs, allow_parameter=False, label="input")
        self._validate_tensors(
            parameters,
            allow_parameter=True,
            label="parameter",
        )
        self._validate_tensors(outputs, allow_parameter=False, label="output")
        self._validate_non_tensor_args(non_tensor_args)

        signature = {
            "operator_id": self.config["operator_id"],
            "model_path": self.config["model_path"],
            "tensor_parallel_size": self.tp_size,
            "tp_rank": self.config["tp_rank"],
            "inputs": {
                name: self._tensor_metadata(value)
                for name, value in inputs.items()
            },
            "parameters": {
                name: self._tensor_metadata(value)
                for name, value in parameters.items()
            },
            "non_tensor_args": dict(non_tensor_args),
            "outputs": {
                name: self._tensor_metadata(value)
                for name, value in outputs.items()
            },
        }
        current_shape_id = shape_id_for(
            {name: list(value.shape) for name, value in inputs.items()}
        )
        if self._state is None:
            raise KernelCallError("capture state was not initialized for rank 0")
        with self._lock:
            self._state = self._load_or_create_state()
            if self._state["status"] != "ACTIVE" or self._state["capture_closed"]:
                raise KernelCallError("capture is closed and cannot accept more samples")
            for sample in self._state["samples"]:
                if sample["shape_id"] == current_shape_id:
                    sample["repeat_count"] += 1
                    self._state["repeated_call_count"] += 1
                    _atomic_write_json(self.state_path, self._state)
                    return
            if self._state["saved_shape_count"] >= self.config["max_shapes"]:
                self._state["skipped_call_count"] += 1
                for skipped in self._state["skipped_signatures"]:
                    if skipped["shape_id"] == current_shape_id:
                        skipped["call_count"] += 1
                        break
                else:
                    self._state["skipped_signatures"].append(
                        {
                            "shape_id": current_shape_id,
                            "signature": signature,
                            "call_count": 1,
                        }
                    )
                _atomic_write_json(self.state_path, self._state)
                return

            sample_name = f"{current_shape_id}.pt"
            sample_path = self.samples_dir / sample_name
            if sample_path.exists():
                raise KernelCallError(
                    f"sample already exists outside capture state: {sample_name}"
                )
            payload = {
                "schema": KERNEL_CALL_SAMPLE_SCHEMA,
                "spec_binding": self.config["spec_binding"],
                "operator_id": self.config["operator_id"],
                "tp_rank": self.tp_rank,
                "tensor_parallel_size": self.tp_size,
                "signature": signature,
                "inputs": {
                    name: value.detach()
                    .to("cpu")
                    .clone(memory_format=torch.preserve_format)
                    for name, value in inputs.items()
                },
                "parameters": {
                    name: value.detach()
                    .to("cpu")
                    .clone(memory_format=torch.preserve_format)
                    for name, value in parameters.items()
                },
                "non_tensor_args": dict(non_tensor_args),
                "outputs": {
                    name: value.detach()
                    .to("cpu")
                    .clone(memory_format=torch.preserve_format)
                    for name, value in outputs.items()
                },
            }
            temporary = sample_path.with_name(f".{sample_name}.{os.getpid()}.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, sample_path)
            self._state["samples"].append(
                {
                    "shape_id": current_shape_id,
                    "file": str(sample_path.relative_to(self.run_dir)),
                    "signature": signature,
                    "repeat_count": 0,
                }
            )
            self._state["saved_shape_count"] += 1
            _atomic_write_json(self.state_path, self._state)
