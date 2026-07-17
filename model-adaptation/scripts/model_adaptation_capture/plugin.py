"""SGLang plugin entry point for rank-0 Step3p5MLP capture and replay."""

import contextvars
import json
import os
from pathlib import Path
from typing import Any, Optional

from .collector import CandidateCollector
from .contracts import (
    CAPTURE_CONFIG_ENV,
    FORWARD_HOOK_TARGET,
    MLP_HOOK_TARGET,
    REPLAY_CONFIG_ENV,
)
from .replay import LoadedModelReplay


_execution_phase: contextvars.ContextVar[str] = contextvars.ContextVar(
    "model_adaptation_execution_phase",
    default="unknown",
)
_collector: Optional[CandidateCollector] = None
_replay: Optional[LoadedModelReplay] = None
_config_path: Optional[Path] = None
_mode: Optional[str] = None


def _phase_from_forward_batch(forward_batch: Any) -> str:
    forward_mode = getattr(forward_batch, "forward_mode", None)
    name = getattr(forward_mode, "name", None)
    if isinstance(name, str) and name:
        return name.lower()
    rendered = str(forward_mode) if forward_mode is not None else "unknown"
    return rendered.lower() if rendered else "unknown"


def _forward_batch(args: tuple, kwargs: dict) -> Any:
    if "forward_batch" in kwargs:
        return kwargs["forward_batch"]
    # HookRegistry passes the bound method arguments as
    # self, input_ids, positions, forward_batch, ...
    return args[3] if len(args) > 3 else None


def around_target_forward(original, *args, **kwargs):
    token = _execution_phase.set(_phase_from_forward_batch(_forward_batch(args, kwargs)))
    try:
        return original(*args, **kwargs)
    finally:
        _execution_phase.reset(token)


def _tp_context(config_path: Path) -> tuple[int, int]:
    try:
        from sglang.srt.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        return group.rank_in_group, group.world_size
    except (AssertionError, ImportError):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        preflight_context = config.get("preflight_tp_context")
        if isinstance(preflight_context, dict):
            return preflight_context["rank"], preflight_context["size"]
        raise RuntimeError(
            "capture requires an initialized SGLang tensor-parallel group"
        )


def _get_collector() -> CandidateCollector:
    global _collector
    if _config_path is None:
        raise RuntimeError(f"{CAPTURE_CONFIG_ENV} was not set during plugin registration")
    if _collector is None:
        tp_rank, tp_size = _tp_context(_config_path)
        config = json.loads(_config_path.read_text(encoding="utf-8"))
        loaded_checkpoint = (
            None
            if "preflight_tp_context" in config
            else _loaded_checkpoint_identity()
        )
        _collector = CandidateCollector.from_config_path(
            _config_path,
            tp_rank=tp_rank,
            tp_size=tp_size,
            loaded_checkpoint=loaded_checkpoint,
        )
    return _collector


def _loaded_checkpoint_identity() -> dict[str, str]:
    from sglang.srt.server_args import get_global_server_args

    server_args = get_global_server_args()
    model_path = getattr(server_args, "model_path", None)
    revision = getattr(server_args, "revision", None)
    if not isinstance(model_path, str) or not model_path:
        raise RuntimeError("SGLang model_path is required for loaded replay")
    if not isinstance(revision, str) or not revision:
        raise RuntimeError(
            "SGLang revision is required to verify the loaded checkpoint"
        )
    return {
        "model_path": model_path,
        "revision": revision,
    }


def _get_replay() -> LoadedModelReplay:
    global _replay
    if _config_path is None:
        raise RuntimeError(f"{REPLAY_CONFIG_ENV} was not set during plugin registration")
    if _replay is None:
        tp_rank, tp_size = _tp_context(_config_path)
        _replay = LoadedModelReplay.from_config_path(
            _config_path,
            tp_rank=tp_rank,
            tp_size=tp_size,
            loaded_checkpoint=_loaded_checkpoint_identity(),
        )
    return _replay


def _model_instance_path(module: Any) -> str:
    gate_up_proj = getattr(module, "gate_up_proj", None)
    prefix = getattr(gate_up_proj, "prefix", None)
    suffix = ".gate_up_proj"
    if not isinstance(prefix, str) or not prefix.endswith(suffix):
        raise RuntimeError(
            "Step3p5MLP gate_up_proj prefix is required to identify the model instance"
        )
    return prefix[: -len(suffix)]


def around_step3p5_mlp(original, module, x):
    limit = getattr(module, "limit", None)
    if limit is None:
        return original(module, x)
    model_instance_path = _model_instance_path(module)
    if _mode == "replay":
        _get_replay().run_for_instance(
            original,
            module,
            model_instance_path=model_instance_path,
            device=x.device,
        )
        return original(module, x)
    output = original(module, x)
    _get_collector().record(
        x,
        output,
        model_instance_path=model_instance_path,
        limit=limit,
        execution_phase=_execution_phase.get(),
    )
    return output


def register() -> None:
    """Register hooks only for launches that explicitly supply a capture config."""
    global _config_path, _mode
    capture_config = os.environ.get(CAPTURE_CONFIG_ENV)
    replay_config = os.environ.get(REPLAY_CONFIG_ENV)
    if capture_config and replay_config:
        raise RuntimeError(
            f"{CAPTURE_CONFIG_ENV} and {REPLAY_CONFIG_ENV} are mutually exclusive"
        )
    configured = capture_config or replay_config
    if configured is None:
        return
    _config_path = Path(configured).resolve()
    _mode = "capture" if capture_config else "replay"

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        FORWARD_HOOK_TARGET,
        around_target_forward,
        HookType.AROUND,
    )
    HookRegistry.register(
        MLP_HOOK_TARGET,
        around_step3p5_mlp,
        HookType.AROUND,
    )


def _reset_for_tests() -> None:
    global _collector, _config_path, _mode, _replay
    _collector = None
    _replay = None
    _config_path = None
    _mode = None
    _execution_phase.set("unknown")
