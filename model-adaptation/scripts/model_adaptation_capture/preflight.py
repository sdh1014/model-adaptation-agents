"""Validate the original MLP hook and rank-0 format without a checkpoint."""

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
    FORWARD_HOOK_TARGET,
    MLP_HOOK_TARGET,
    PLUGIN_ENTRY_POINT,
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
) -> Dict[str, str]:
    environment = os.environ.copy()
    scripts_root = str(Path(__file__).resolve().parents[1])
    sglang_python = str((sglang_worktree / "python").resolve())
    python_paths = [scripts_root, sglang_python]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment[CAPTURE_CONFIG_ENV] = str(config_path.resolve())
    environment["SGLANG_PLUGINS"] = PLUGIN_ENTRY_POINT
    return environment


def run_capture_preflight(
    config_path: Path,
    sglang_worktree: Path,
) -> tuple[bool, list[str]]:
    run_dir = config_path.resolve().parent
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "model_adaptation_capture.preflight",
            "--mode",
            "capture",
            "--config",
            str(config_path.resolve()),
        ],
        cwd=sglang_worktree,
        env=_worker_environment(config_path, sglang_worktree),
        text=True,
        capture_output=True,
        check=False,
    )
    (run_dir / "preflight-capture.log").write_text(
        "\n".join(
            [
                "worker_mode=capture",
                f"returncode={completed.returncode}",
                "--- stdout ---",
                completed.stdout,
                "--- stderr ---",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )
    evidence = ["capture-config.json", "preflight-capture.log"]
    for candidate in (
        "capture-state.json",
        "preflight-capture-summary.json",
    ):
        if (run_dir / candidate).exists():
            evidence.append(candidate)
    return completed.returncode == 0, evidence


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
        {FORWARD_HOOK_TARGET, MLP_HOOK_TARGET} - HookRegistry._patched
    )
    if missing_hooks:
        raise PreflightError(
            "SGLang did not apply capture hooks: " + ", ".join(missing_hooks)
        )

    from sglang.srt.models.step3p5 import Step3p5MLP

    loaded_model_source = Path(
        sys.modules[Step3p5MLP.__module__].__file__
    ).resolve()
    expected_model_source = Path(config["source"]["model_path"]).resolve()
    if loaded_model_source != expected_model_source:
        raise PreflightError(
            f"loaded model source {loaded_model_source} does not match "
            f"fixed source {expected_model_source}"
        )

    class GateUpProjection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.prefix = "model.layers.43.share_expert.gate_up_proj"

        def forward(self, x):
            return torch.cat((x, x), dim=-1), None

    class DownProjection(torch.nn.Module):
        def forward(self, x):
            return x, None

    mlp = Step3p5MLP.__new__(Step3p5MLP)
    torch.nn.Module.__init__(mlp)
    mlp.limit = 16.0
    mlp.gate_up_proj = GateUpProjection()
    mlp.down_proj = DownProjection()

    device = torch.device("cuda", torch.cuda.current_device())
    for rows in (1, 2, 4):
        x = torch.linspace(
            -4,
            4,
            steps=rows * 8,
            device=device,
        ).reshape(rows, 8)
        output = mlp.forward(x.to(torch.bfloat16))
        if output.dtype != torch.bfloat16 or not bool(
            torch.isfinite(output).all().item()
        ):
            raise PreflightError(
                "original Step3p5MLP.forward did not return finite BF16 output"
            )
    mlp.forward(
        torch.full((2, 8), 2.0, dtype=torch.bfloat16, device=device)
    )
    mlp.forward(torch.ones((8, 8), dtype=torch.bfloat16, device=device))
    torch.cuda.synchronize(device)

    state = read_json_object(
        Path(config["run_dir"]) / "capture-state.json",
        "rank-0 capture state",
    )
    expected_counts = {
        "saved_shape_count": 3,
        "repeated_call_count": 1,
        "skipped_call_count": 1,
    }
    for field, expected in expected_counts.items():
        if state.get(field) != expected:
            raise PreflightError(
                f"capture state {field} must be {expected}, got {state.get(field)!r}"
            )

    write_json(
        Path(config["run_dir"]) / "preflight-capture-summary.json",
        {
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
            "loaded_model_source": str(loaded_model_source),
            "applied_hooks": [FORWARD_HOOK_TARGET, MLP_HOOK_TARGET],
            "tp_rank": 0,
            "tensor_parallel_size": 8,
            "rank_zero_format_only": True,
            "checkpoint_loaded": False,
            "loaded_model_replay_performed": False,
            **expected_counts,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("capture",))
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_capture_worker(args.config)
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
