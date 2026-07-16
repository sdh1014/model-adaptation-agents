"""Run the CUDA Hook smoke and new-process candidate replay."""

import argparse
from importlib.metadata import entry_points
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

from .contracts import (
    CANDIDATE_SAMPLE_SCHEMA,
    CAPTURE_CONFIG_ENV,
    FORWARD_HOOK_TARGET,
    PLUGIN_ENTRY_POINT,
    SWIGLU_HOOK_TARGET,
)


class PreflightError(RuntimeError):
    """The CUDA preflight could not form trustworthy evidence."""


def read_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise PreflightError(f"cannot read {label}: {error}") from error
    except json.JSONDecodeError as error:
        raise PreflightError(f"{label} is not valid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise PreflightError(f"{label} must be one JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _worker_environment(
    config_path: Path,
    sglang_worktree: Path,
    capture: bool,
) -> Dict[str, str]:
    environment = os.environ.copy()
    scripts_root = str(Path(__file__).resolve().parents[1])
    sglang_python = str((sglang_worktree / "python").resolve())
    python_paths = [scripts_root, sglang_python]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    if capture:
        environment[CAPTURE_CONFIG_ENV] = str(config_path.resolve())
        environment["SGLANG_PLUGINS"] = PLUGIN_ENTRY_POINT
    else:
        environment.pop(CAPTURE_CONFIG_ENV, None)
    return environment


def _run_worker(
    mode: str,
    config_path: Path,
    sglang_worktree: Path,
    log_path: Path,
    *,
    capture: bool,
) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "model_adaptation_capture.preflight",
            "--mode",
            mode,
            "--config",
            str(config_path.resolve()),
        ],
        cwd=sglang_worktree,
        env=_worker_environment(config_path, sglang_worktree, capture),
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.write_text(
        "\n".join(
            [
                f"worker_mode={mode}",
                f"returncode={completed.returncode}",
                "--- stdout ---",
                completed.stdout,
                "--- stderr ---",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )
    return completed


def run_workers(
    config_path: Path,
    sglang_worktree: Path,
) -> tuple[bool, list[str]]:
    run_dir = config_path.resolve().parent
    capture_worker = _run_worker(
        "capture",
        config_path,
        sglang_worktree,
        run_dir / "preflight-capture.log",
        capture=True,
    )
    verify_worker = None
    if capture_worker.returncode == 0:
        verify_worker = _run_worker(
            "verify",
            config_path,
            sglang_worktree,
            run_dir / "preflight-verify.log",
            capture=False,
        )

    passed = capture_worker.returncode == 0 and (
        verify_worker is not None and verify_worker.returncode == 0
    )
    evidence = ["capture-config.json", "preflight-capture.log"]
    for candidate in (
        "capture-state.json",
        "preflight-capture-summary.json",
        "preflight-verify.log",
        "preflight-verify-summary.json",
    ):
        if (run_dir / candidate).exists():
            evidence.append(candidate)
    return passed, evidence


def _require_cuda_torch():
    try:
        import torch
    except ImportError as error:
        raise PreflightError(f"preflight requires Torch: {error}") from error
    if not torch.cuda.is_available():
        raise PreflightError("preflight requires an available CUDA device")
    return torch


def _require_capture_entry_point() -> None:
    matches = [
        item
        for item in entry_points(group="sglang.srt.plugins")
        if item.name == PLUGIN_ENTRY_POINT
    ]
    if len(matches) != 1:
        raise PreflightError(
            "installed environment must contain exactly one "
            f"sglang.srt.plugins entry point named {PLUGIN_ENTRY_POINT!r}"
        )


def run_capture_worker(config_path: Path) -> None:
    config = read_json_object(config_path, "capture config")
    torch = _require_cuda_torch()
    _require_capture_entry_point()

    from sglang.srt.plugins import load_plugins
    from sglang.srt.plugins.hook_registry import HookRegistry

    load_plugins()
    missing_hooks = sorted(
        {FORWARD_HOOK_TARGET, SWIGLU_HOOK_TARGET} - HookRegistry._patched
    )
    if missing_hooks:
        raise PreflightError(
            "SGLang did not apply capture hooks: " + ", ".join(missing_hooks)
        )

    from sglang.srt.models import step3p5_ops

    loaded_helper = Path(step3p5_ops.__file__).resolve()
    expected_helper = Path(config["source"]["helper_path"]).resolve()
    if loaded_helper != expected_helper:
        raise PreflightError(
            f"loaded helper {loaded_helper} does not match fixed source {expected_helper}"
        )

    device = torch.device("cuda", torch.cuda.current_device())
    for rows in (1, 2, 4):
        gate_up = torch.linspace(-4, 4, steps=rows * 8, device=device).reshape(
            rows, 8
        )
        output = step3p5_ops.step_swiglu_with_limit(
            gate_up.to(torch.bfloat16),
            16.0,
        )
        if output.dtype != torch.bfloat16 or not bool(
            torch.isfinite(output).all().item()
        ):
            raise PreflightError(
                "actual SGLang helper did not return finite BF16 output"
            )
    step3p5_ops.step_swiglu_with_limit(
        torch.full((2, 8), 2.0, dtype=torch.bfloat16, device=device),
        16.0,
    )
    step3p5_ops.step_swiglu_with_limit(
        torch.ones((8, 8), dtype=torch.bfloat16, device=device),
        16.0,
    )
    torch.cuda.synchronize(device)

    state = read_json_object(
        Path(config["run_dir"]) / "capture-state.json",
        "capture state",
    )
    expected_counts = {
        "saved_sample_count": 3,
        "repeated_call_count": 1,
        "skipped_call_count": 1,
    }
    for field, expected in expected_counts.items():
        if state.get(field) != expected:
            raise PreflightError(
                f"capture state {field} must be {expected}, got {state.get(field)!r}"
            )
    skipped = state.get("skipped_signatures")
    if not isinstance(skipped, list) or len(skipped) != 1:
        raise PreflightError("capture state must retain the one skipped signature")

    write_json(
        Path(config["run_dir"]) / "preflight-capture-summary.json",
        {
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
            "loaded_helper": str(loaded_helper),
            "applied_hooks": [FORWARD_HOOK_TARGET, SWIGLU_HOOK_TARGET],
            "skipped_signature_count": len(skipped),
            **expected_counts,
        },
    )


def _validate_candidate_payload(
    torch,
    payload: Any,
    config: Dict[str, Any],
    state_sample: Dict[str, Any],
) -> None:
    from .collector import signature_id

    expected_keys = {
        "schema",
        "spec_binding",
        "operator_id",
        "signature",
        "inputs",
        "parameters",
        "expected",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise PreflightError("capture candidate has unexpected structure")
    if payload["schema"] != CANDIDATE_SAMPLE_SCHEMA:
        raise PreflightError(
            f"capture candidate schema is not {CANDIDATE_SAMPLE_SCHEMA}"
        )
    if payload["spec_binding"] != config["spec_binding"]:
        raise PreflightError("candidate spec binding does not match capture config")
    if payload["operator_id"] != config["operator_id"]:
        raise PreflightError("candidate operator does not match capture config")
    if not isinstance(payload["inputs"], dict) or set(payload["inputs"]) != {
        "gate_up"
    }:
        raise PreflightError("candidate inputs must contain only gate_up")
    if not isinstance(payload["parameters"], dict) or set(payload["parameters"]) != {
        "limit"
    }:
        raise PreflightError("candidate parameters must contain only limit")

    gate_up = payload["inputs"]["gate_up"]
    expected = payload["expected"]
    if not isinstance(gate_up, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise PreflightError("candidate input and expected values must be tensors")
    if gate_up.device.type != "cpu" or expected.device.type != "cpu":
        raise PreflightError("serialized candidate tensors must be on CPU")
    if gate_up.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16:
        raise PreflightError("candidate tensors must be BF16")
    if not bool(torch.isfinite(gate_up).all().item()) or not bool(
        torch.isfinite(expected).all().item()
    ):
        raise PreflightError("candidate tensors must be finite")
    if gate_up.ndim < 1 or gate_up.shape[-1] % 2 != 0:
        raise PreflightError("candidate gate_up last dimension must be even")
    if expected.shape != gate_up.shape[:-1] + (gate_up.shape[-1] // 2,):
        raise PreflightError("candidate expected shape does not match SwiGLU output")

    signature = payload["signature"]
    if not isinstance(signature, dict) or signature != state_sample.get("signature"):
        raise PreflightError("candidate signature does not match capture state")
    if signature_id(signature) != state_sample.get("signature_id"):
        raise PreflightError("candidate signature digest does not match capture state")
    input_signature = signature.get("inputs", {}).get("gate_up", {})
    if input_signature != {
        "kind": "tensor",
        "shape": list(gate_up.shape),
        "dtype": "bfloat16",
        "layout": "strided",
        "stride": list(gate_up.stride()),
    }:
        raise PreflightError("candidate tensor metadata does not match gate_up")
    if signature.get("parameters") != payload["parameters"]:
        raise PreflightError("candidate scalar metadata does not match parameters")


def run_verify_worker(config_path: Path) -> None:
    config = read_json_object(config_path, "capture config")
    torch = _require_cuda_torch()
    from sglang.srt.models.step3p5_ops import step_swiglu_with_limit

    loaded_helper = Path(sys.modules[step_swiglu_with_limit.__module__].__file__).resolve()
    expected_helper = Path(config["source"]["helper_path"]).resolve()
    if loaded_helper != expected_helper:
        raise PreflightError(
            f"loaded helper {loaded_helper} does not match fixed source {expected_helper}"
        )

    run_dir = Path(config["run_dir"]).resolve()
    state = read_json_object(run_dir / "capture-state.json", "capture state")
    samples = state.get("samples")
    if not isinstance(samples, list) or len(samples) != 3:
        raise PreflightError("preflight verification requires exactly three candidates")
    precision = config["precision_gate"]
    if (
        precision.get("comparator") != "torch.testing.assert_close"
        or precision.get("atol") != 0.01
        or precision.get("rtol") != 0.02
    ):
        raise PreflightError("capture config Precision Gate has drifted")

    device = torch.device("cuda", torch.cuda.current_device())
    checked = []
    for state_sample in samples:
        sample_path = (run_dir / state_sample["file"]).resolve()
        if not sample_path.is_relative_to(run_dir):
            raise PreflightError("candidate path escapes the Run directory")
        payload = torch.load(sample_path, map_location="cpu", weights_only=True)
        _validate_candidate_payload(torch, payload, config, state_sample)
        actual = step_swiglu_with_limit(
            payload["inputs"]["gate_up"].to(device),
            payload["parameters"]["limit"],
        ).to("cpu")
        torch.testing.assert_close(
            actual,
            payload["expected"],
            atol=precision["atol"],
            rtol=precision["rtol"],
            equal_nan=False,
            check_device=True,
            check_dtype=True,
            check_layout=True,
            check_stride=precision["check_stride"],
        )
        checked.append(state_sample["file"])
    write_json(
        run_dir / "preflight-verify-summary.json",
        {
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
            "new_process": True,
            "serialization": config["serialization"],
            "checked_candidates": checked,
            "comparator": precision["comparator"],
            "atol": precision["atol"],
            "rtol": precision["rtol"],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("capture", "verify"))
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        action = run_capture_worker if args.mode == "capture" else run_verify_worker
        action(args.config)
    except (
        PreflightError,
        AssertionError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"capture preflight worker: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
