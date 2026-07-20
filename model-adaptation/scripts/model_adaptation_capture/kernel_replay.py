"""Replay the selected Kernel Call through the existing CUDA or Kunlun seam."""

import argparse
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Optional

import torch

from .contracts import (
    FIXED_PRECISION_GATE,
    KERNEL_CALL_SAMPLE_SCHEMA,
    KERNEL_CALL_STATE_SCHEMA,
    KERNEL_REPLAY_CONFIG_SCHEMA,
    KERNEL_REPLAY_RESULT_SCHEMA,
    KUNLUN_SWIGLU_TARGET,
    SWIGLU_CLAMP_OPERATOR_ID,
    VALIDATION_TP_RANK,
    shape_id_for,
)


class KernelReplayError(RuntimeError):
    """The standalone replay cannot form trustworthy evidence."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise KernelReplayError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise KernelReplayError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise KernelReplayError(f"{label} must be one JSON object")
    return value


def _atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _default_cuda_call(x: torch.Tensor, gemm1_limit: float) -> torch.Tensor:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _swiglu_silu_clamp_mul,
    )

    return _swiglu_silu_clamp_mul(x, gemm1_limit)


def _default_p800_call(
    x: torch.Tensor,
    gemm1_limit: float,
    *,
    pass_limit: bool = False,
) -> torch.Tensor:
    import kunlun_ops

    actual = torch.empty(
        x.shape[:-1] + (x.shape[-1] // 2,),
        dtype=x.dtype,
        device=x.device,
    )
    if pass_limit:
        kunlun_ops.swiglu(x=x, y=actual, limit=float(gemm1_limit))
    else:
        kunlun_ops.swiglu(x=x, y=actual)
    return actual


def _is_git_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_candidate(candidate: Any) -> None:
    if not isinstance(candidate, dict):
        raise KernelReplayError("candidate replay metadata must be one object")
    if set(candidate) != {
        "candidate_operator_id",
        "modified_paths",
        "original_call_arguments",
        "original_call_none_fallback",
        "original_call_source",
        "original_call_source_sha256",
        "result",
        "result_sha256",
        "patch_sha256",
        "sglang_kunlun_worktree",
        "sglang_kunlun_revision",
    }:
        raise KernelReplayError("candidate replay metadata has drifted")
    if not isinstance(candidate["result"], str) or not candidate["result"]:
        raise KernelReplayError("candidate replay result path is invalid")
    if not _is_sha256(candidate["result_sha256"]):
        raise KernelReplayError("candidate result SHA-256 is invalid")
    if not _is_sha256(candidate["patch_sha256"]):
        raise KernelReplayError("candidate patch SHA-256 is invalid")
    if (
        not isinstance(candidate["candidate_operator_id"], str)
        or not candidate["candidate_operator_id"]
    ):
        raise KernelReplayError("candidate operator id is invalid")
    modified_paths = candidate["modified_paths"]
    if (
        not isinstance(modified_paths, list)
        or not modified_paths
        or not all(isinstance(path, str) and path for path in modified_paths)
        or len(modified_paths) != len(set(modified_paths))
    ):
        raise KernelReplayError("candidate modified paths are invalid")
    if (
        not isinstance(candidate["original_call_source"], str)
        or not candidate["original_call_source"]
    ):
        raise KernelReplayError("candidate original call source is invalid")
    if not _is_sha256(candidate["original_call_source_sha256"]):
        raise KernelReplayError("candidate original call source SHA-256 is invalid")
    if candidate["original_call_arguments"] != {
        "x": "y",
        "y": "out1",
        "limit": "float(layer.moe_runner_config.gemm1_clamp_limit)",
    }:
        raise KernelReplayError(
            "candidate does not preserve the expected original Kernel Call arguments"
        )
    if candidate["original_call_none_fallback"] is not True:
        raise KernelReplayError(
            "candidate does not preserve the original no-limit fallback"
        )
    if (
        not isinstance(candidate["sglang_kunlun_worktree"], str)
        or not candidate["sglang_kunlun_worktree"]
    ):
        raise KernelReplayError("candidate worktree is invalid")
    if not _is_git_revision(candidate["sglang_kunlun_revision"]):
        raise KernelReplayError("candidate Kunlun revision is invalid")


def _validate_config(config: Dict[str, Any]) -> None:
    legacy_fields = {
        "repair_entry",
        "sglang_kunlun_worktree",
        "sglang_kunlun_revision",
    }
    if legacy_fields.intersection(config):
        raise KernelReplayError(
            "kernel replay config contains the retired repair callable protocol"
        )
    if config.get("schema") != KERNEL_REPLAY_CONFIG_SCHEMA:
        raise KernelReplayError("kernel replay config schema has drifted")
    if config.get("operator_id") != SWIGLU_CLAMP_OPERATOR_ID:
        raise KernelReplayError("kernel replay operator has drifted")
    if not isinstance(config.get("spec_binding"), dict):
        raise KernelReplayError("kernel replay config is missing spec_binding")
    if config.get("tensor_parallel_size") != 8:
        raise KernelReplayError("kernel replay requires TP8 capture metadata")
    if config.get("tp_rank") != VALIDATION_TP_RANK:
        raise KernelReplayError(
            f"kernel replay tp_rank must be {VALIDATION_TP_RANK}"
        )
    if config.get("precision_gate") != FIXED_PRECISION_GATE:
        raise KernelReplayError("kernel replay precision gate has drifted")
    execution_site = config.get("execution_site")
    if execution_site not in {"cuda", "p800"}:
        raise KernelReplayError("execution_site must be cuda or p800")
    expected_target = (
        SWIGLU_CLAMP_OPERATOR_ID
        if execution_site == "cuda"
        else KUNLUN_SWIGLU_TARGET
    )
    if config.get("invocation_target") != expected_target:
        raise KernelReplayError("kernel replay invocation target has drifted")
    candidate = config.get("candidate")
    if execution_site == "cuda" and candidate is not None:
        raise KernelReplayError("CUDA self-replay must not carry a candidate")
    if candidate is not None:
        if execution_site != "p800":
            raise KernelReplayError("candidate replay is only valid on P800")
        if config.get("allow_active_capture") is not False:
            raise KernelReplayError(
                "candidate replay must not allow active capture"
            )
        _validate_candidate(candidate)
    if not isinstance(config.get("allow_active_capture"), bool):
        raise KernelReplayError("kernel replay active-capture policy has drifted")
    if not _is_sha256(config.get("sample_files_sha256")):
        raise KernelReplayError(
            "kernel replay sample_files_sha256 is invalid"
        )
    for field in ("golden_run", "run_dir"):
        if not isinstance(config.get(field), str) or not config[field]:
            raise KernelReplayError(f"kernel replay config is missing {field}")


def _load_state(config: Dict[str, Any]) -> Dict[str, Any]:
    golden_run = Path(config["golden_run"]).resolve()
    if _file_sha256(golden_run / "sample-files.json") != config.get(
        "sample_files_sha256"
    ):
        raise KernelReplayError(
            "kernel replay sample file evidence has drifted"
        )
    state = _read_json_object(
        golden_run / "capture-state.json",
        "Kernel Call capture state",
    )
    checks = {
        "schema": KERNEL_CALL_STATE_SCHEMA,
        "spec_binding": config["spec_binding"],
        "operator_id": config["operator_id"],
        "tp_rank": config["tp_rank"],
        "tensor_parallel_size": config["tensor_parallel_size"],
    }
    for field, expected in checks.items():
        if state.get(field) != expected:
            raise KernelReplayError(f"Kernel Call capture state {field} has drifted")
    if config["allow_active_capture"]:
        if state.get("status") not in {"ACTIVE", "SEALED"}:
            raise KernelReplayError("preflight capture state is not replayable")
    elif state.get("status") != "SEALED" or state.get("capture_closed") is not True:
        raise KernelReplayError("kernel replay requires a SEALED Golden Run")
    samples = state.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 3:
        raise KernelReplayError("Kernel Call capture requires one to three shapes")
    if state.get("saved_shape_count") != len(samples):
        raise KernelReplayError("Kernel Call saved_shape_count has drifted")
    return state


def _load_sample(
    config: Dict[str, Any],
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(sample, dict):
        raise KernelReplayError("Kernel Call sample entry must be one object")
    shape_id = sample.get("shape_id")
    expected_file = f"samples/{shape_id}.pt"
    if not isinstance(shape_id, str) or not shape_id:
        raise KernelReplayError("Kernel Call sample shape id is invalid")
    if sample.get("file") != expected_file:
        raise KernelReplayError("Kernel Call sample path has drifted")
    golden_run = Path(config["golden_run"]).resolve()
    sample_path = (golden_run / expected_file).resolve()
    if not sample_path.is_relative_to(golden_run):
        raise KernelReplayError("Kernel Call sample path escapes the Golden Run")
    try:
        payload = torch.load(
            sample_path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise KernelReplayError(f"cannot load Kernel Call sample: {error}") from error
    expected_keys = {
        "schema",
        "spec_binding",
        "operator_id",
        "tp_rank",
        "tensor_parallel_size",
        "signature",
        "inputs",
        "parameters",
        "non_tensor_args",
        "outputs",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise KernelReplayError("Kernel Call sample structure has drifted")
    checks = {
        "schema": KERNEL_CALL_SAMPLE_SCHEMA,
        "spec_binding": config["spec_binding"],
        "operator_id": config["operator_id"],
        "tp_rank": config["tp_rank"],
        "tensor_parallel_size": config["tensor_parallel_size"],
    }
    for field, expected in checks.items():
        if payload.get(field) != expected:
            raise KernelReplayError(f"Kernel Call sample {field} has drifted")
    if not isinstance(payload["signature"], dict):
        raise KernelReplayError("Kernel Call sample signature must be one object")
    if payload["signature"] != sample.get("signature"):
        raise KernelReplayError(
            "Kernel Call sample signature differs from capture state"
        )
    if not isinstance(payload["inputs"], dict) or set(payload["inputs"]) != {"x"}:
        raise KernelReplayError("Kernel Call sample must contain only input x")
    if payload["parameters"] != {}:
        raise KernelReplayError("Kernel Call sample must not contain parameters")
    if (
        not isinstance(payload["non_tensor_args"], dict)
        or set(payload["non_tensor_args"]) != {"gemm1_limit"}
    ):
        raise KernelReplayError(
            "Kernel Call sample must contain only gemm1_limit"
        )
    if (
        not isinstance(payload["outputs"], dict)
        or set(payload["outputs"]) != {"output"}
    ):
        raise KernelReplayError("Kernel Call sample must contain only output")
    x = payload["inputs"]["x"]
    expected_output = payload["outputs"]["output"]
    if not isinstance(x, torch.Tensor) or not isinstance(
        expected_output,
        torch.Tensor,
    ):
        raise KernelReplayError("Kernel Call input and output must be Tensors")
    if x.device.type != "cpu" or expected_output.device.type != "cpu":
        raise KernelReplayError("serialized Kernel Call tensors must be on CPU")
    if x.dtype != torch.bfloat16 or expected_output.dtype != torch.bfloat16:
        raise KernelReplayError("serialized Kernel Call tensors must be BF16")
    if (
        x.ndim < 1
        or x.shape[-1] % 2 != 0
        or expected_output.shape != x.shape[:-1] + (x.shape[-1] // 2,)
    ):
        raise KernelReplayError("Kernel Call sample shape has drifted")
    if not bool(torch.isfinite(x).all().item()) or not bool(
        torch.isfinite(expected_output).all().item()
    ):
        raise KernelReplayError("serialized Kernel Call tensors must be finite")
    gemm1_limit = payload["non_tensor_args"]["gemm1_limit"]
    if isinstance(gemm1_limit, bool) or not isinstance(
        gemm1_limit,
        (int, float),
    ):
        raise KernelReplayError("gemm1_limit must be a numeric scalar")
    if not math.isfinite(float(gemm1_limit)):
        raise KernelReplayError("gemm1_limit must be finite")
    expected_signature = {
        "operator_id": config["operator_id"],
        "model_path": "target",
        "tensor_parallel_size": config["tensor_parallel_size"],
        "tp_rank": config["tp_rank"],
        "inputs": {
            "x": {
                "kind": "tensor",
                "shape": list(x.shape),
                "dtype": str(x.dtype).removeprefix("torch."),
                "layout": str(x.layout).removeprefix("torch."),
                "stride": list(x.stride()),
            }
        },
        "parameters": {},
        "non_tensor_args": {"gemm1_limit": float(gemm1_limit)},
    }
    if payload["signature"] != expected_signature:
        raise KernelReplayError("Kernel Call sample signature has drifted")
    try:
        actual_shape_id = shape_id_for(
            payload["signature"]["inputs"]["x"]["shape"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise KernelReplayError("Kernel Call shape metadata has drifted") from error
    if actual_shape_id != shape_id:
        raise KernelReplayError("Kernel Call shape digest has drifted")
    return payload


def run_kernel_replay_worker(
    config: Dict[str, Any],
    *,
    cuda_call: Optional[Callable[[torch.Tensor, float], torch.Tensor]] = None,
    p800_call: Optional[Callable[[torch.Tensor, float], torch.Tensor]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Replay every Golden Sample and write metadata-only comparison evidence."""
    _validate_config(config)
    state = _load_state(config)
    run_dir = Path(config["run_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    selected_device = device or torch.device(
        "cuda",
        torch.cuda.current_device(),
    )
    if config["execution_site"] == "cuda":
        invoke = cuda_call or _default_cuda_call
    else:
        candidate = config.get("candidate")
        invoke = p800_call or partial(
            _default_p800_call,
            pass_limit=isinstance(candidate, dict),
        )
    precision = config["precision_gate"]
    checked_shapes = []
    errors = []

    for sample in state["samples"]:
        payload = _load_sample(config, sample)
        shape_id = sample["shape_id"]
        x = payload["inputs"]["x"].to(selected_device)
        try:
            actual = invoke(
                x,
                float(payload["non_tensor_args"]["gemm1_limit"]),
            )
        except (RuntimeError, TypeError) as error:
            checked_shapes.append({"shape_id": shape_id, "passed": False})
            errors.append(
                {
                    "shape_id": shape_id,
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
            continue

        try:
            if not isinstance(actual, torch.Tensor):
                raise KernelReplayError("existing Kernel Call did not return a Tensor")
            if not bool(torch.isfinite(actual).all().item()):
                raise KernelReplayError("existing Kernel Call returned non-finite values")
            actual_cpu = actual.detach().to("cpu")
            expected = payload["outputs"]["output"]
            if actual_cpu.shape != expected.shape:
                raise KernelReplayError("Kernel Call output shape differs from CUDA")
            if (
                precision["require_same_dtype"]
                and actual_cpu.dtype != expected.dtype
            ):
                raise KernelReplayError("Kernel Call output dtype differs from CUDA")
            torch.testing.assert_close(
                actual_cpu,
                expected,
                atol=precision["atol"],
                rtol=precision["rtol"],
                equal_nan=False,
                check_device=True,
                check_dtype=precision["require_same_dtype"],
                check_layout=True,
                check_stride=precision["check_stride"],
            )
            checked_shapes.append({"shape_id": shape_id, "passed": True})
        except (AssertionError, KernelReplayError) as error:
            checked_shapes.append({"shape_id": shape_id, "passed": False})
            errors.append(
                {
                    "shape_id": shape_id,
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )

    failed_shape_count = sum(not item["passed"] for item in checked_shapes)
    result = {
        "schema": KERNEL_REPLAY_RESULT_SCHEMA,
        "spec_binding": config["spec_binding"],
        "operator_id": config["operator_id"],
        "execution_site": config["execution_site"],
        "invocation_target": config["invocation_target"],
        "tp_rank": config["tp_rank"],
        "tensor_parallel_size": config["tensor_parallel_size"],
        "precision_gate": precision,
        "sample_files_sha256": config["sample_files_sha256"],
        "passed": failed_shape_count == 0,
        "checked_shape_count": len(checked_shapes),
        "failed_shape_count": failed_shape_count,
        "checked_shapes": checked_shapes,
        "errors": errors,
        "actual_tensors_saved": False,
    }
    _atomic_write_json(run_dir / "worker-result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = _read_json_object(args.config.resolve(), "kernel replay config")
        result = run_kernel_replay_worker(config)
    except (
        KernelReplayError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"kernel replay worker: {error}", file=sys.stderr)
        return 2
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
