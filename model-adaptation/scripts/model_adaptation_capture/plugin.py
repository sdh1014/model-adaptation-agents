"""SGLang plugin entry point for bounded Step-3.7 Golden capture."""

import contextvars
import os
from pathlib import Path
from typing import Any, Optional

from .collector import CandidateCollector
from .contracts import (
    CAPTURE_CONFIG_ENV,
    FORWARD_HOOK_TARGET,
    SWIGLU_HOOK_TARGET,
)


_execution_phase: contextvars.ContextVar[str] = contextvars.ContextVar(
    "model_adaptation_execution_phase",
    default="unknown",
)
_collector: Optional[CandidateCollector] = None
_config_path: Optional[Path] = None


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


def _get_collector() -> CandidateCollector:
    global _collector
    if _config_path is None:
        raise RuntimeError(f"{CAPTURE_CONFIG_ENV} was not set during plugin registration")
    if _collector is None:
        _collector = CandidateCollector.from_config_path(_config_path)
    return _collector


def around_step_swiglu(original, gate_up, limit):
    return _get_collector().capture(
        original,
        gate_up,
        limit,
        execution_phase=_execution_phase.get(),
    )


def register() -> None:
    """Register hooks only for launches that explicitly supply a capture config."""
    global _config_path
    configured = os.environ.get(CAPTURE_CONFIG_ENV)
    if not configured:
        return
    _config_path = Path(configured).resolve()

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        FORWARD_HOOK_TARGET,
        around_target_forward,
        HookType.AROUND,
    )
    HookRegistry.register(
        SWIGLU_HOOK_TARGET,
        around_step_swiglu,
        HookType.AROUND,
    )


def _reset_for_tests() -> None:
    global _collector, _config_path
    _collector = None
    _config_path = None
    _execution_phase.set("unknown")
