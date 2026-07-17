"""Collect at most three rank-0 shapes at the original MLP boundary."""

import json
import math
import os
from pathlib import Path
import threading
from typing import Any, Dict

import torch

from .contracts import (
    CANDIDATE_SAMPLE_SCHEMA,
    CANDIDATE_SERIALIZATION,
    CONFIG_SCHEMA,
    ACTIVATION_GUARD,
    OPERATOR_ID,
    STATE_SCHEMA,
    VALIDATION_TP_RANK,
    shape_id_for,
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


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


class CandidateCollector:
    """Save rank 0 calls while every other TP rank remains read-only."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        tp_rank: int,
        tp_size: int,
        loaded_checkpoint: Dict[str, str] | None = None,
    ):
        self._validate_config(config)
        if tp_size != config["tensor_parallel_size"]:
            raise CaptureError(
                "runtime tensor parallel size does not match capture config"
            )
        if isinstance(tp_rank, bool) or not isinstance(tp_rank, int):
            raise CaptureError("tp_rank must be an integer")
        if tp_rank < 0 or tp_rank >= tp_size:
            raise CaptureError("tp_rank is outside the configured TP group")
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
        self._state = None
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
        loaded_checkpoint: Dict[str, str] | None = None,
    ) -> "CandidateCollector":
        return cls(
            _read_json_object(path.resolve()),
            tp_rank=tp_rank,
            tp_size=tp_size,
            loaded_checkpoint=loaded_checkpoint,
        )

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        if config.get("schema") != CONFIG_SCHEMA:
            raise CaptureError(f"capture config schema must be {CONFIG_SCHEMA!r}")
        if config.get("operator_id") != OPERATOR_ID:
            raise CaptureError("capture config operator is not the Demo operator")
        if config.get("activation_guard") != ACTIVATION_GUARD:
            raise CaptureError("capture config activation guard has drifted")
        if config.get("model_path") != "target":
            raise CaptureError("capture config model_path must be target")
        if config.get("max_shapes") != 3:
            raise CaptureError("capture config max_shapes must be 3")
        if config.get("tensor_parallel_size") != 8:
            raise CaptureError("capture config tensor_parallel_size must be 8")
        if config.get("tp_rank") != VALIDATION_TP_RANK:
            raise CaptureError(
                f"capture config tp_rank must be {VALIDATION_TP_RANK}"
            )
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
        checkpoint = config.get("checkpoint")
        if (
            not isinstance(checkpoint, dict)
            or set(checkpoint)
            != {"id", "model_path", "revision", "config_digest"}
            or any(
                not isinstance(checkpoint[field], str)
                or not checkpoint[field]
                for field in checkpoint
            )
            or checkpoint["id"]
            != f"{checkpoint['model_path']}@{checkpoint['revision']}"
        ):
            raise CaptureError("capture config checkpoint identity is invalid")

    @staticmethod
    def _validate_loaded_checkpoint(
        config: Dict[str, Any],
        loaded_checkpoint: Dict[str, str] | None,
    ) -> Dict[str, str] | None:
        if "preflight_tp_context" in config:
            if loaded_checkpoint is not None:
                raise CaptureError(
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
            raise CaptureError(
                "loaded checkpoint identity is required for formal capture"
            )
        for field in ("model_path", "revision"):
            if loaded_checkpoint[field] != config["checkpoint"][field]:
                raise CaptureError(
                    f"loaded checkpoint {field} does not match capture config"
                )
        return loaded_checkpoint

    def _load_or_create_state(self) -> Dict[str, Any]:
        if self.state_path.exists():
            state = _read_json_object(self.state_path)
            if (
                state.get("schema") != STATE_SCHEMA
                or state.get("operator_id") != self.config["operator_id"]
                or state.get("tp_rank") != self.config["tp_rank"]
                or state.get("tensor_parallel_size") != self.tp_size
                or state.get("spec_binding") != self.config["spec_binding"]
                or state.get("checkpoint") != self.config["checkpoint"]
                or state.get("loaded_checkpoint") != self.loaded_checkpoint
                or state.get("status") not in {"ACTIVE", "SEALED", "FAILED"}
                or not isinstance(state.get("capture_closed"), bool)
            ):
                raise CaptureError("existing capture state does not match config")
            return state
        state = {
            "schema": STATE_SCHEMA,
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
        model_instance_path: str,
        limit: float,
        execution_phase: str,
    ) -> Dict[str, Any]:
        return {
            "operator_id": self.config["operator_id"],
            "model_path": self.config["model_path"],
            "model_instance_path": model_instance_path,
            "execution_phase": execution_phase,
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
            "parameters": {"limit": float(limit)},
        }

    def _validate_new_sample(
        self,
        x: torch.Tensor,
        limit: float,
        output: torch.Tensor,
    ) -> None:
        if isinstance(x, torch.nn.Parameter):
            raise CaptureError("nn.Parameter inputs are forbidden")
        if not isinstance(output, torch.Tensor):
            raise CaptureError("operator output must be a Tensor")
        if isinstance(output, torch.nn.Parameter):
            raise CaptureError("nn.Parameter outputs are forbidden")
        expected_device = self.config["capture_device_type"]
        if x.device.type != expected_device or output.device.type != expected_device:
            raise CaptureError(
                f"capture requires {expected_device} tensors, got "
                f"{x.device.type} input and {output.device.type} output"
            )
        if x.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
            raise CaptureError("capture requires BF16 input and output tensors")
        if x.layout != torch.strided or output.layout != torch.strided:
            raise CaptureError("capture requires dense strided tensors")
        if isinstance(limit, bool) or not isinstance(limit, (int, float)):
            raise CaptureError("limit must be a numeric scalar")
        if not math.isfinite(float(limit)):
            raise CaptureError("limit must be finite")
        if not bool(torch.isfinite(x).all().item()):
            raise CaptureError("capture input contains non-finite values")
        if not bool(torch.isfinite(output).all().item()):
            raise CaptureError("capture output contains non-finite values")

    def record(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        *,
        model_instance_path: str,
        limit: float,
        execution_phase: str,
    ) -> None:
        if not self.is_capture_rank:
            return
        if not isinstance(x, torch.Tensor):
            raise CaptureError("x must be a Tensor")
        if not isinstance(model_instance_path, str) or not model_instance_path:
            raise CaptureError("model_instance_path must be a non-empty string")
        if not isinstance(execution_phase, str) or not execution_phase:
            raise CaptureError("execution_phase must be a non-empty string")

        signature = self._signature(
            x,
            model_instance_path,
            limit,
            execution_phase,
        )
        current_shape_id = shape_id_for(x.shape)
        if self._state is None:
            raise CaptureError("capture state was not initialized for rank 0")
        with self._lock:
            self._state = self._load_or_create_state()
            if (
                self._state["status"] != "ACTIVE"
                or self._state["capture_closed"]
            ):
                raise CaptureError("capture is closed and cannot accept more samples")

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

            self._validate_new_sample(x, limit, output)
            sample_name = f"{current_shape_id}.pt"
            sample_path = self.samples_dir / sample_name
            if sample_path.exists():
                raise CaptureError(f"sample already exists outside capture state: {sample_name}")
            payload = {
                "schema": CANDIDATE_SAMPLE_SCHEMA,
                "spec_binding": self.config["spec_binding"],
                "operator_id": self.config["operator_id"],
                "model_instance_path": model_instance_path,
                "tp_rank": self.tp_rank,
                "tensor_parallel_size": self.tp_size,
                "signature": signature,
                "inputs": {
                    "x": x.detach()
                    .to("cpu")
                    .clone(memory_format=torch.preserve_format)
                },
                "parameters": {"limit": float(limit)},
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
