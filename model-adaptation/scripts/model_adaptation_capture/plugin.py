"""SGLang plugin entry point for bounded MLP and Kernel Call capture."""

import contextvars
import json
import os
from pathlib import Path
from typing import Any, Optional

from .contracts import (
    ATTENTION_CAPTURE_SEAM,
    ATTENTION_OPERATOR_ID,
    CAPTURE_CONFIG_ENV,
    FORWARD_HOOK_TARGET,
    GEMMA_FUSED_ADD_RMSNORM_CAPTURE_SEAM,
    GEMMA_FUSED_ADD_RMSNORM_OPERATOR_ID,
    GEMMA_RMSNORM_CAPTURE_SEAM,
    GEMMA_RMSNORM_OPERATOR_ID,
    MLP_HOOK_TARGET,
    REPLAY_CONFIG_ENV,
    SESSION_CONFIG_SCHEMA,
    SWIGLU_CLAMP_HOOK_TARGET,
    SWIGLU_CLAMP_OPERATOR_ID,
    TOPK_SIGMOID_CAPTURE_SEAM,
    TOPK_SIGMOID_OPERATOR_ID,
)


_execution_phase: contextvars.ContextVar[str] = contextvars.ContextVar(
    "model_adaptation_execution_phase",
    default="unknown",
)
_collector: Optional[Any] = None
_collectors: dict[str, Any] = {}
_replay: Optional[Any] = None
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


def _get_collector(operator_id: Optional[str] = None) -> Any:
    global _collector, _collectors
    if _config_path is None:
        raise RuntimeError(f"{CAPTURE_CONFIG_ENV} was not set during plugin registration")
    config = json.loads(_config_path.read_text(encoding="utf-8"))
    if config.get("schema") == SESSION_CONFIG_SCHEMA:
        if not isinstance(operator_id, str) or not operator_id:
            raise RuntimeError("session capture requires an operator id")
        if operator_id in _collectors:
            return _collectors[operator_id]
        matches = [
            item
            for item in config.get("operators", [])
            if isinstance(item, dict) and item.get("operator_id") == operator_id
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"session config must contain exactly one operator {operator_id!r}"
            )
        from .kernel_call import PlannedKernelCallCollector

        tp_rank, tp_size = _tp_context(_config_path)
        loaded_checkpoint = (
            None
            if "preflight_tp_context" in config
            else _loaded_checkpoint_identity()
        )
        _collectors[operator_id] = PlannedKernelCallCollector(
            matches[0],
            tp_rank=tp_rank,
            tp_size=tp_size,
            loaded_checkpoint=loaded_checkpoint,
        )
        return _collectors[operator_id]

    if operator_id is not None and config.get("operator_id") != operator_id:
        raise RuntimeError(
            f"capture config does not contain requested operator {operator_id!r}"
        )
    if _collector is None:
        tp_rank, tp_size = _tp_context(_config_path)
        if config.get("operator_id") == SWIGLU_CLAMP_OPERATOR_ID:
            from .kernel_call import KernelCallCollector

            collector_type = KernelCallCollector
        else:
            from .collector import CandidateCollector

            collector_type = CandidateCollector
        loaded_checkpoint = (
            None
            if "preflight_tp_context" in config
            else _loaded_checkpoint_identity()
        )
        _collector = collector_type.from_config_path(
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


def _get_replay() -> Any:
    global _replay
    if _config_path is None:
        raise RuntimeError(f"{REPLAY_CONFIG_ENV} was not set during plugin registration")
    if _replay is None:
        from .replay import LoadedModelReplay

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


def around_swiglu_clamp(original, x, gemm1_limit):
    output = original(x, gemm1_limit)
    collector = _get_collector(SWIGLU_CLAMP_OPERATOR_ID)
    if hasattr(collector, "record_call"):
        collector.record_call(
            inputs={"x": x},
            parameters={},
            non_tensor_args={"gemm1_limit": gemm1_limit},
            outputs={"output": output},
        )
    else:
        collector.record(
            x,
            output,
            gemm1_limit=gemm1_limit,
        )
    return output


def around_gemma_rmsnorm(original, x, weight, eps):
    output = original(x, weight, eps)
    _get_collector(GEMMA_RMSNORM_OPERATOR_ID).record_call(
        inputs={"x": x},
        parameters={"weight": weight},
        non_tensor_args={"eps": eps},
        outputs={"output": output},
    )
    return output


def around_gemma_fused_add_rmsnorm(original, x, residual, weight, eps):
    collector = _get_collector(GEMMA_FUSED_ADD_RMSNORM_OPERATOR_ID)
    if collector.is_capture_rank:
        capture_x = x.detach().clone()
        capture_residual = residual.detach().clone()
    else:
        capture_x = x
        capture_residual = residual
    result = original(x, residual, weight, eps)
    collector.record_call(
        inputs={"x": capture_x, "residual": capture_residual},
        parameters={"weight": weight},
        non_tensor_args={"eps": eps},
        outputs={"mutated_x": x, "mutated_residual": residual},
    )
    return result


def around_topk_sigmoid(
    original,
    topk_weights,
    topk_ids,
    gating_output,
    renormalize,
    correction_bias,
):
    result = original(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
        correction_bias,
    )
    _get_collector(TOPK_SIGMOID_OPERATOR_ID).record_call(
        inputs={"gating_output": gating_output},
        parameters={"correction_bias": correction_bias},
        non_tensor_args={"renormalize": renormalize},
        outputs={"topk_weights": topk_weights, "topk_ids": topk_ids},
    )
    return result


def around_vision_prefill_attention(
    original,
    q,
    k,
    v,
    o,
    b_start_loc,
    b_seq_len,
    max_input_len,
    is_causal=True,
    sm_scale=None,
):
    result = original(
        q,
        k,
        v,
        o,
        b_start_loc,
        b_seq_len,
        max_input_len,
        is_causal=is_causal,
        sm_scale=sm_scale,
    )
    _get_collector(ATTENTION_OPERATOR_ID).record_call(
        inputs={
            "q": q,
            "k": k,
            "v": v,
            "b_start_loc": b_start_loc,
            "b_seq_len": b_seq_len,
        },
        parameters={},
        non_tensor_args={
            "max_input_len": max_input_len,
            "is_causal": is_causal,
            "sm_scale": sm_scale,
        },
        outputs={"output": o},
    )
    return result


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
    config = json.loads(_config_path.read_text(encoding="utf-8"))

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if config.get("schema") == SESSION_CONFIG_SCHEMA:
        hooks = {
            SWIGLU_CLAMP_OPERATOR_ID: (
                SWIGLU_CLAMP_HOOK_TARGET,
                around_swiglu_clamp,
            ),
            GEMMA_RMSNORM_OPERATOR_ID: (
                GEMMA_RMSNORM_CAPTURE_SEAM,
                around_gemma_rmsnorm,
            ),
            GEMMA_FUSED_ADD_RMSNORM_OPERATOR_ID: (
                GEMMA_FUSED_ADD_RMSNORM_CAPTURE_SEAM,
                around_gemma_fused_add_rmsnorm,
            ),
            TOPK_SIGMOID_OPERATOR_ID: (
                TOPK_SIGMOID_CAPTURE_SEAM,
                around_topk_sigmoid,
            ),
            ATTENTION_OPERATOR_ID: (
                ATTENTION_CAPTURE_SEAM,
                around_vision_prefill_attention,
            ),
        }
        operator_entries = config.get("operators")
        if not isinstance(operator_entries, list) or not operator_entries:
            raise RuntimeError("session capture config requires operators")
        seen = set()
        for item in operator_entries:
            operator_id = item.get("operator_id") if isinstance(item, dict) else None
            if operator_id not in hooks or operator_id in seen:
                raise RuntimeError(
                    f"unsupported or duplicate session operator {operator_id!r}"
                )
            target, hook = hooks[operator_id]
            HookRegistry.register(target, hook, HookType.AROUND)
            seen.add(operator_id)
        return

    if config.get("operator_id") == SWIGLU_CLAMP_OPERATOR_ID:
        if _mode != "capture":
            raise RuntimeError(
                "the standalone SwiGLU replay does not use the SGLang plugin"
            )
        HookRegistry.register(
            SWIGLU_CLAMP_HOOK_TARGET,
            around_swiglu_clamp,
            HookType.AROUND,
        )
        return

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
    global _collector, _collectors, _config_path, _mode, _replay
    _collector = None
    _collectors = {}
    _replay = None
    _config_path = None
    _mode = None
    _execution_phase.set("unknown")
