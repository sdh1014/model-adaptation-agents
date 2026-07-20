"""Validate CUDA software and hardware readiness without touching operators."""

import argparse
from importlib.metadata import entry_points
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

from .contracts import (
    CAPTURE_CONFIG_ENV,
    PLUGIN_ENTRY_POINT,
    REPLAY_CONFIG_ENV,
)


ENVIRONMENT_CONFIG_SCHEMA = "cuda-environment-preflight-config/v1"
ENVIRONMENT_RESULT_SCHEMA = "cuda-environment-preflight-result/v1"


class EnvironmentPreflightError(RuntimeError):
    """The CUDA environment could not form trustworthy readiness evidence."""


def read_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise EnvironmentPreflightError(
            f"cannot read {label}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise EnvironmentPreflightError(
            f"{label} is not valid JSON: {error.msg}"
        ) from error
    if not isinstance(value, dict):
        raise EnvironmentPreflightError(f"{label} must be one JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _worker_environment(sglang_worktree: Path) -> Dict[str, str]:
    environment = os.environ.copy()
    scripts_root = str(Path(__file__).resolve().parents[1])
    sglang_python = str((sglang_worktree / "python").resolve())
    python_paths = [scripts_root, sglang_python]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["SGLANG_PLUGINS"] = PLUGIN_ENTRY_POINT
    environment.pop(CAPTURE_CONFIG_ENV, None)
    environment.pop(REPLAY_CONFIG_ENV, None)
    return environment


def run_environment_preflight(
    config_path: Path,
    sglang_worktree: Path,
) -> tuple[bool, Dict[str, Any], list[str]]:
    run_dir = config_path.resolve().parent
    config = read_json_object(config_path, "environment preflight config")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "model_adaptation_capture.preflight",
            "--config",
            str(config_path.resolve()),
        ],
        cwd=sglang_worktree,
        env=_worker_environment(sglang_worktree),
        text=True,
        capture_output=True,
        check=False,
    )
    (run_dir / "environment.log").write_text(
        "\n".join(
            [
                "worker_mode=environment",
                f"returncode={completed.returncode}",
                "--- stdout ---",
                completed.stdout,
                "--- stderr ---",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )
    evidence = ["environment-config.json", "environment.log"]
    result_path = run_dir / "environment-result.json"
    if not result_path.is_file():
        return False, {}, evidence
    evidence.append("environment-result.json")
    result = read_json_object(result_path, "environment preflight result")
    expected = {
        "schema": ENVIRONMENT_RESULT_SCHEMA,
        "spec_binding": config["spec_binding"],
        "sglang_revision": config["source"]["sglang_revision"],
        "tensor_parallel_size": config["runtime"]["tensor_parallel_size"],
        "dtype": config["runtime"]["dtype"],
        "passed": True,
    }
    environment = result.get("environment")
    if not isinstance(environment, dict):
        return False, {}, evidence
    required_environment = {
        "cuda_available": True,
        "cuda_device_count": config["runtime"]["tensor_parallel_size"],
        "bfloat16_supported": True,
        "plugin_entry_point": PLUGIN_ENTRY_POINT,
    }
    passed = (
        completed.returncode == 0
        and all(result.get(field) == value for field, value in expected.items())
        and environment.get("cuda_device_count", 0)
        >= required_environment["cuda_device_count"]
        and all(
            environment.get(field) == value
            for field, value in required_environment.items()
            if field != "cuda_device_count"
        )
    )
    return passed, environment, evidence


def _require_cuda_torch():
    try:
        import torch
    except ImportError as error:
        raise EnvironmentPreflightError(
            f"environment preflight requires Torch: {error}"
        ) from error
    if not torch.cuda.is_available():
        raise EnvironmentPreflightError(
            "environment preflight requires an available CUDA device"
        )
    return torch


def _require_capture_entry_point() -> None:
    matches = [
        item
        for item in entry_points(group="sglang.srt.plugins")
        if item.name == PLUGIN_ENTRY_POINT
    ]
    if len(matches) != 1:
        raise EnvironmentPreflightError(
            "installed environment must contain exactly one "
            f"sglang.srt.plugins entry point named {PLUGIN_ENTRY_POINT!r}"
        )


def run_environment_worker(config_path: Path) -> None:
    config = read_json_object(config_path, "environment preflight config")
    if config.get("schema") != ENVIRONMENT_CONFIG_SCHEMA:
        raise EnvironmentPreflightError(
            "environment preflight config schema is invalid"
        )
    p800_only_environment = {}
    if os.environ.get("SGLANG_PLATFORM", "").lower() == "kunlun":
        p800_only_environment["SGLANG_PLATFORM"] = os.environ[
            "SGLANG_PLATFORM"
        ]
    if os.environ.get(
        "SGLANG_IS_FLASHINFER_AVAILABLE",
        "",
    ).lower() in {"0", "false", "no"}:
        p800_only_environment["SGLANG_IS_FLASHINFER_AVAILABLE"] = (
            os.environ["SGLANG_IS_FLASHINFER_AVAILABLE"]
        )
    if p800_only_environment:
        raise EnvironmentPreflightError(
            "CUDA environment contains P800-only launch flags: "
            + ", ".join(sorted(p800_only_environment))
        )

    runtime = config.get("runtime")
    source = config.get("source")
    if not isinstance(runtime, dict) or not isinstance(source, dict):
        raise EnvironmentPreflightError(
            "environment preflight config is missing runtime or source"
        )
    tensor_parallel_size = runtime.get("tensor_parallel_size")
    if (
        isinstance(tensor_parallel_size, bool)
        or not isinstance(tensor_parallel_size, int)
        or tensor_parallel_size < 1
    ):
        raise EnvironmentPreflightError(
            "tensor_parallel_size must be a positive integer"
        )
    if runtime.get("dtype") != "bfloat16":
        raise EnvironmentPreflightError(
            "environment preflight requires bfloat16"
        )

    torch = _require_cuda_torch()
    device_count = torch.cuda.device_count()
    if device_count < tensor_parallel_size:
        raise EnvironmentPreflightError(
            "CUDA environment exposes fewer devices than tensor_parallel_size: "
            f"required {tensor_parallel_size}, got {device_count}"
        )
    if not torch.cuda.is_bf16_supported():
        raise EnvironmentPreflightError(
            "CUDA environment does not support bfloat16"
        )

    _require_capture_entry_point()
    import sglang
    from sglang.srt.plugins import load_plugins

    module_file = getattr(sglang, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise EnvironmentPreflightError(
            "loaded SGLang module does not have a source path"
        )
    loaded_module = Path(module_file).resolve()
    expected_package = (
        Path(source["sglang_worktree"]).resolve() / "python" / "sglang"
    )
    if not loaded_module.is_relative_to(expected_package):
        raise EnvironmentPreflightError(
            f"loaded SGLang module {loaded_module} is outside "
            f"{expected_package}"
        )

    load_plugins()
    environment = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": True,
        "cuda_device_count": device_count,
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(tensor_parallel_size)
        ],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "bfloat16_supported": True,
        "sglang_module_path": str(loaded_module),
        "plugin_entry_point": PLUGIN_ENTRY_POINT,
    }
    write_json(
        Path(config["run_dir"]) / "environment-result.json",
        {
            "schema": ENVIRONMENT_RESULT_SCHEMA,
            "spec_binding": config["spec_binding"],
            "sglang_revision": source["sglang_revision"],
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": runtime["dtype"],
            "passed": True,
            "environment": environment,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_environment_worker(args.config)
    except (
        EnvironmentPreflightError,
        AssertionError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"environment preflight worker: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
