"""Replay rank-0 Golden Samples inside an already loaded TP8 model."""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict

import torch

from .contracts import (
    CANDIDATE_SAMPLE_SCHEMA,
    FIXED_PRECISION_GATE,
    OPERATOR_ID,
    REPLAY_CONFIG_SCHEMA,
    REPLAY_RESULT_SCHEMA,
    STATE_SCHEMA,
    VALIDATION_TP_RANK,
    shape_id_for,
)


class ModelReplayError(RuntimeError):
    """The loaded model replay cannot produce trustworthy rank evidence."""


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelReplayError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ModelReplayError(f"{label} must be one JSON object")
    return value


def _atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _shape_groups_from_state(state: Dict[str, Any]) -> list[tuple[str, str]]:
    samples = state.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 3:
        raise ModelReplayError(
            "Golden capture state requires one to three shapes"
        )
    groups = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ModelReplayError("Golden capture sample entry is invalid")
        signature = sample.get("signature")
        shape_id = sample.get("shape_id")
        model_instance_path = (
            signature.get("model_instance_path")
            if isinstance(signature, dict)
            else None
        )
        if (
            not isinstance(shape_id, str)
            or not shape_id
            or not isinstance(model_instance_path, str)
            or not model_instance_path
        ):
            raise ModelReplayError(
                "Golden capture sample shape groups are invalid"
            )
        groups.append((shape_id, model_instance_path))
    if len(set(groups)) != len(groups):
        raise ModelReplayError("Golden capture shape groups contain duplicates")
    return groups


class LoadedModelReplay:
    """Compare Golden Samples using the loaded rank-0 MLP weights."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        tp_rank: int,
        tp_size: int,
        loaded_checkpoint: Dict[str, str],
    ):
        if config.get("schema") != REPLAY_CONFIG_SCHEMA:
            raise ModelReplayError("loaded-model replay config schema has drifted")
        if config.get("operator_id") != OPERATOR_ID:
            raise ModelReplayError("loaded-model replay operator has drifted")
        if config.get("tensor_parallel_size") != 8 or tp_size != 8:
            raise ModelReplayError("loaded-model replay requires TP8")
        if config.get("tp_rank") != VALIDATION_TP_RANK:
            raise ModelReplayError(
                f"loaded-model replay tp_rank must be {VALIDATION_TP_RANK}"
            )
        if isinstance(tp_rank, bool) or not isinstance(tp_rank, int):
            raise ModelReplayError("tp_rank must be an integer")
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ModelReplayError("tp_rank is outside the TP group")
        if not isinstance(config.get("spec_binding"), dict):
            raise ModelReplayError("loaded-model replay is missing spec_binding")
        if config.get("weights_source") != "loaded_checkpoint":
            raise ModelReplayError(
                "loaded-model replay must use the loaded checkpoint weights"
            )
        checkpoint = config.get("checkpoint")
        if (
            not isinstance(checkpoint, dict)
            or set(checkpoint)
            != {"id", "model_path", "revision", "config_digest"}
        ):
            raise ModelReplayError(
                "loaded-model replay checkpoint identity is invalid"
            )
        if (
            not isinstance(loaded_checkpoint, dict)
            or set(loaded_checkpoint) != {"model_path", "revision"}
            or any(
                not isinstance(loaded_checkpoint[field], str)
                or not loaded_checkpoint[field]
                for field in loaded_checkpoint
            )
        ):
            raise ModelReplayError("loaded checkpoint identity is unavailable")
        for field in ("model_path", "revision"):
            if loaded_checkpoint[field] != checkpoint.get(field):
                raise ModelReplayError(
                    f"loaded checkpoint {field} does not match Contract Data"
                )
        if config.get("precision_gate") != FIXED_PRECISION_GATE:
            raise ModelReplayError(
                "loaded-model replay precision gate has drifted"
            )
        shape_groups = config.get("shape_groups")
        if not isinstance(shape_groups, list) or not 1 <= len(shape_groups) <= 3:
            raise ModelReplayError("loaded-model replay requires one to three shapes")
        for group in shape_groups:
            if (
                not isinstance(group, dict)
                or set(group) != {"shape_id", "model_instance_path"}
                or not isinstance(group["shape_id"], str)
                or not group["shape_id"]
                or not isinstance(group["model_instance_path"], str)
                or not group["model_instance_path"]
            ):
                raise ModelReplayError(
                    "loaded-model replay shape group is invalid"
                )
        configured_groups = [
            (group["shape_id"], group["model_instance_path"])
            for group in shape_groups
        ]
        if len(set(configured_groups)) != len(configured_groups):
            raise ModelReplayError(
                "loaded-model replay shape groups contain duplicates"
            )

        golden_run_value = config.get("golden_run")
        if not isinstance(golden_run_value, str) or not golden_run_value:
            raise ModelReplayError("loaded-model replay is missing Golden Run")
        golden_run = Path(golden_run_value).resolve()
        state = _read_json_object(
            golden_run / "capture-state.json",
            "Golden capture state",
        )
        state_checks = {
            "schema": STATE_SCHEMA,
            "spec_binding": config["spec_binding"],
            "operator_id": config["operator_id"],
            "tp_rank": config["tp_rank"],
            "tensor_parallel_size": tp_size,
            "checkpoint": checkpoint,
            "loaded_checkpoint": loaded_checkpoint,
            "capture_closed": True,
        }
        for field, expected in state_checks.items():
            if state.get(field) != expected:
                raise ModelReplayError(
                    f"Golden capture state {field} has drifted"
                )
        if state.get("status") not in {"ACTIVE", "SEALED"}:
            raise ModelReplayError(
                "Golden capture state is not replayable"
            )
        if configured_groups != _shape_groups_from_state(state):
            raise ModelReplayError(
                "loaded-model replay shape groups do not match Golden capture"
            )

        self.config = config
        self.loaded_checkpoint = loaded_checkpoint
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.run_dir = Path(config["run_dir"]).resolve()
        self.golden_run = golden_run
        self.result_path = self.run_dir / "replay-result.json"
        self._processed_paths: set[str] = set()
        self._result = {
            "schema": REPLAY_RESULT_SCHEMA,
            "spec_binding": config["spec_binding"],
            "operator_id": config["operator_id"],
            "tp_rank": tp_rank,
            "tensor_parallel_size": tp_size,
            "checkpoint": checkpoint,
            "loaded_checkpoint": loaded_checkpoint,
            "passed": True,
            "checked_shapes": [],
            "errors": [],
        }

    @classmethod
    def from_config_path(
        cls,
        path: Path,
        *,
        tp_rank: int,
        tp_size: int,
        loaded_checkpoint: Dict[str, str],
    ) -> "LoadedModelReplay":
        return cls(
            _read_json_object(path.resolve(), "loaded-model replay config"),
            tp_rank=tp_rank,
            tp_size=tp_size,
            loaded_checkpoint=loaded_checkpoint,
        )

    def _load_payload(
        self,
        shape_id: str,
        model_instance_path: str,
    ) -> Dict[str, Any]:
        sample_path = (
            self.golden_run
            / "samples"
            / f"{shape_id}.pt"
        ).resolve()
        if not sample_path.is_relative_to(self.golden_run):
            raise ModelReplayError("Golden Sample path escapes the Golden Run")
        try:
            payload = torch.load(
                sample_path,
                map_location="cpu",
                weights_only=True,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ModelReplayError(
                f"cannot load rank-0 Golden Sample {sample_path}: {error}"
            ) from error
        expected_keys = {
            "schema",
            "spec_binding",
            "operator_id",
            "model_instance_path",
            "tp_rank",
            "tensor_parallel_size",
            "signature",
            "inputs",
            "parameters",
            "outputs",
        }
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            raise ModelReplayError("rank-0 Golden Sample structure has drifted")
        expected_values = {
            "schema": CANDIDATE_SAMPLE_SCHEMA,
            "spec_binding": self.config["spec_binding"],
            "operator_id": self.config["operator_id"],
            "model_instance_path": model_instance_path,
            "tp_rank": self.config["tp_rank"],
            "tensor_parallel_size": self.tp_size,
        }
        for field, expected in expected_values.items():
            if payload.get(field) != expected:
                raise ModelReplayError(
                    f"rank-0 Golden Sample {field} does not match replay config"
                )
        try:
            actual_shape_id = shape_id_for(
                payload["signature"]["inputs"]["x"]["shape"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ModelReplayError(
                "rank-0 Golden Sample shape metadata has drifted"
            ) from error
        if actual_shape_id != shape_id:
            raise ModelReplayError("rank-0 Golden Sample shape digest has drifted")
        if set(payload["inputs"]) != {"x"} or set(payload["outputs"]) != {"output"}:
            raise ModelReplayError("rank-0 Golden Sample must contain x and output")
        if set(payload["parameters"]) != {"limit"}:
            raise ModelReplayError("rank-0 Golden Sample must contain only limit")
        x = payload["inputs"]["x"]
        output = payload["outputs"]["output"]
        if not isinstance(x, torch.Tensor) or not isinstance(output, torch.Tensor):
            raise ModelReplayError("rank-0 Golden input and output must be tensors")
        if x.device.type != "cpu" or output.device.type != "cpu":
            raise ModelReplayError("serialized Golden tensors must be on CPU")
        if x.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
            raise ModelReplayError("rank-0 Golden tensors must be BF16")
        if not bool(torch.isfinite(x).all().item()) or not bool(
            torch.isfinite(output).all().item()
        ):
            raise ModelReplayError("rank-0 Golden tensors must be finite")
        return payload

    def run_for_instance(
        self,
        original: Callable[..., Any],
        module: Any,
        *,
        model_instance_path: str,
        device: torch.device,
    ) -> None:
        if self.tp_rank != self.config["tp_rank"]:
            return
        if model_instance_path in self._processed_paths:
            return
        self._processed_paths.add(model_instance_path)
        groups = [
            item
            for item in self.config["shape_groups"]
            if item["model_instance_path"] == model_instance_path
        ]
        if not groups:
            return

        precision = self.config["precision_gate"]
        for group in groups:
            shape_id = group["shape_id"]
            try:
                payload = self._load_payload(shape_id, model_instance_path)
                if payload["parameters"]["limit"] != float(module.limit):
                    raise ModelReplayError(
                        "loaded Step3p5MLP limit does not match Golden Sample"
                    )
                actual = original(
                    module,
                    payload["inputs"]["x"].to(device),
                )
                if not isinstance(actual, torch.Tensor):
                    raise ModelReplayError("loaded Step3p5MLP output must be a Tensor")
                actual = actual.detach().to("cpu")
                if not bool(torch.isfinite(actual).all().item()):
                    raise ModelReplayError("loaded Step3p5MLP output is non-finite")
                torch.testing.assert_close(
                    actual,
                    payload["outputs"]["output"],
                    atol=precision["atol"],
                    rtol=precision["rtol"],
                    equal_nan=False,
                    check_device=True,
                    check_dtype=precision["require_same_dtype"],
                    check_layout=True,
                    check_stride=precision["check_stride"],
                )
                self._result["checked_shapes"].append(
                    {
                        "shape_id": shape_id,
                        "model_instance_path": model_instance_path,
                        "passed": True,
                    }
                )
            except (AssertionError, ModelReplayError, RuntimeError, TypeError) as error:
                self._result["passed"] = False
                self._result["checked_shapes"].append(
                    {
                        "shape_id": shape_id,
                        "model_instance_path": model_instance_path,
                        "passed": False,
                    }
                )
                self._result["errors"].append(
                    {
                        "shape_id": shape_id,
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
        _atomic_write_json(self.result_path, self._result)
