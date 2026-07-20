"""Run the historical CUDA capture-adapter validation worker."""

import argparse
import hashlib
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
    KERNEL_REPLAY_CONFIG_SCHEMA,
    KERNEL_REPLAY_RESULT_SCHEMA,
    MLP_HOOK_TARGET,
    PLUGIN_ENTRY_POINT,
    SESSION_CONFIG_SCHEMA,
    SWIGLU_CLAMP_HOOK_TARGET,
    SWIGLU_CLAMP_OPERATOR_ID,
)


# The preflight-* filenames below are retained only so revision 1-5 evidence
# remains reproducible. This module is not the environment preflight.
class PreflightError(RuntimeError):
    """Historical capture-adapter validation could not form trusted evidence."""


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


def _replay_worker_environment(
    config_path: Path,
    sglang_worktree: Path,
) -> Dict[str, str]:
    environment = _worker_environment(config_path, sglang_worktree)
    environment.pop(CAPTURE_CONFIG_ENV, None)
    environment.pop("SGLANG_PLUGINS", None)
    return environment


def _capture_artifact_fingerprint(
    run_dir: Path,
) -> tuple[str, Dict[str, str]]:
    state_path = run_dir / "capture-state.json"
    sample_paths = sorted((run_dir / "samples").glob("*.pt"))
    if not state_path.is_file() or len(sample_paths) != 3:
        raise PreflightError(
            "rank-0 preflight must produce capture state and exactly three samples"
        )
    state_digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
    sample_digests = {
        str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sample_paths
    }
    return state_digest, sample_digests


def _session_artifact_fingerprint(
    config: Dict[str, Any],
) -> Dict[str, tuple[str, Dict[str, str]]]:
    fingerprints = {}
    for item in config.get("operators", []):
        operator_id = item.get("operator_id") if isinstance(item, dict) else None
        operator_run = Path(item.get("run_dir", "")).resolve()
        state_path = operator_run / "capture-state.json"
        sample_paths = sorted((operator_run / "samples").glob("*.pt"))
        if (
            not isinstance(operator_id, str)
            or not operator_id
            or not state_path.is_file()
            or len(sample_paths) != 3
        ):
            raise PreflightError(
                "rank-0 session preflight must produce one state and exactly "
                f"three samples for {operator_id!r}"
            )
        fingerprints[operator_id] = (
            hashlib.sha256(state_path.read_bytes()).hexdigest(),
            {
                str(path.relative_to(operator_run)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sample_paths
            },
        )
    if len(fingerprints) != len(config.get("operators", [])):
        raise PreflightError("session capture contains duplicate operators")
    return fingerprints


def _run_session_replay_preflight(
    config_path: Path,
    sglang_worktree: Path,
    config: Dict[str, Any],
    evidence: list[str],
) -> tuple[bool, list[str]]:
    run_dir = config_path.resolve().parent
    try:
        rank_zero_fingerprint = _session_artifact_fingerprint(config)
    except PreflightError:
        return False, evidence

    rank_filter_config = json.loads(json.dumps(config))
    rank_filter_config["preflight_tp_context"] = {
        "rank": 1,
        "size": config["tensor_parallel_size"],
    }
    for item in rank_filter_config["operators"]:
        item["preflight_tp_context"] = rank_filter_config[
            "preflight_tp_context"
        ]
    rank_filter_config_path = run_dir / "preflight-rank-filter-config.json"
    write_json(rank_filter_config_path, rank_filter_config)
    rank_filter = subprocess.run(
        [
            sys.executable,
            "-m",
            "model_adaptation_capture.capture_adapter_validation",
            "--mode",
            "capture",
            "--config",
            str(rank_filter_config_path.resolve()),
        ],
        cwd=sglang_worktree,
        env=_worker_environment(rank_filter_config_path, sglang_worktree),
        text=True,
        capture_output=True,
        check=False,
    )
    (run_dir / "preflight-rank-filter.log").write_text(
        "\n".join(
            [
                "worker_mode=session-rank-filter",
                f"returncode={rank_filter.returncode}",
                "--- stdout ---",
                rank_filter.stdout,
                "--- stderr ---",
                rank_filter.stderr,
            ]
        ),
        encoding="utf-8",
    )
    evidence.extend(
        [
            "preflight-rank-filter-config.json",
            "preflight-rank-filter.log",
        ]
    )
    summary_path = run_dir / "preflight-rank-filter-summary.json"
    if not summary_path.is_file():
        return False, evidence
    evidence.append("preflight-rank-filter-summary.json")
    summary = read_json_object(summary_path, "rank-filter summary")
    expected_summary = {
        "applied_hooks": [
            item["hook_target"] for item in config["operators"]
        ],
        "runtime_tp_rank": 1,
        "capture_tp_rank": config["tp_rank"],
        "tensor_parallel_size": config["tensor_parallel_size"],
        "checkpoint_loaded": False,
    }
    try:
        rank_filter_preserved_capture = (
            _session_artifact_fingerprint(config) == rank_zero_fingerprint
        )
    except PreflightError:
        rank_filter_preserved_capture = False
    if (
        rank_filter.returncode != 0
        or summary != expected_summary
        or not rank_filter_preserved_capture
    ):
        return False, evidence

    all_passed = True
    for item in config["operators"]:
        operator_run = Path(item["run_dir"]).resolve()
        sample_records = []
        for sample_path in sorted((operator_run / "samples").glob("*.pt")):
            sample_records.append(
                {
                    "shape_id": sample_path.stem,
                    "path": str(sample_path.relative_to(operator_run)),
                    "size": sample_path.stat().st_size,
                    "sha256": hashlib.sha256(
                        sample_path.read_bytes()
                    ).hexdigest(),
                }
            )
        sample_files_path = operator_run / "sample-files.json"
        write_json(
            sample_files_path,
            {
                "schema": "golden-sample-files/v1",
                "spec_binding": config["spec_binding"],
                "operator_id": item["operator_id"],
                "files": sample_records,
            },
        )
        sample_files_digest = hashlib.sha256(
            sample_files_path.read_bytes()
        ).hexdigest()
        replay_config = {
            "schema": KERNEL_REPLAY_CONFIG_SCHEMA,
            "spec_binding": config["spec_binding"],
            "operator_id": item["operator_id"],
            "adapter": item["adapter"],
            "sample_fields": item["sample_fields"],
            "execution_site": "cuda",
            "invocation_target": item["replay"]["cuda_target"],
            "golden_run": str(operator_run),
            "run_dir": str(operator_run),
            "tensor_parallel_size": config["tensor_parallel_size"],
            "tp_rank": config["tp_rank"],
            "precision_gate": item["precision_gate"],
            "sample_files_sha256": sample_files_digest,
            "allow_active_capture": True,
        }
        replay_config_path = operator_run / "preflight-replay-config.json"
        write_json(replay_config_path, replay_config)
        replay = subprocess.run(
            [
                sys.executable,
                "-m",
                "model_adaptation_capture.kernel_replay",
                "--config",
                str(replay_config_path.resolve()),
            ],
            cwd=sglang_worktree,
            env=_replay_worker_environment(config_path, sglang_worktree),
            text=True,
            capture_output=True,
            check=False,
        )
        replay_log_path = operator_run / "preflight-replay.log"
        replay_log_path.write_text(
            "\n".join(
                [
                    "worker_mode=cuda-self-replay",
                    f"operator_id={item['operator_id']}",
                    f"returncode={replay.returncode}",
                    "--- stdout ---",
                    replay.stdout,
                    "--- stderr ---",
                    replay.stderr,
                ]
            ),
            encoding="utf-8",
        )
        relative_root = str(operator_run.relative_to(run_dir))
        evidence.extend(
            [
                f"{relative_root}/sample-files.json",
                f"{relative_root}/preflight-replay-config.json",
                f"{relative_root}/preflight-replay.log",
            ]
        )
        worker_result_path = operator_run / "worker-result.json"
        if not worker_result_path.is_file():
            all_passed = False
            continue
        evidence.append(f"{relative_root}/worker-result.json")
        worker_result = read_json_object(
            worker_result_path,
            "preflight replay result",
        )
        checks = {
            "schema": KERNEL_REPLAY_RESULT_SCHEMA,
            "spec_binding": config["spec_binding"],
            "operator_id": item["operator_id"],
            "execution_site": "cuda",
            "invocation_target": item["replay"]["cuda_target"],
            "tp_rank": config["tp_rank"],
            "tensor_parallel_size": config["tensor_parallel_size"],
            "precision_gate": item["precision_gate"],
            "sample_files_sha256": sample_files_digest,
            "actual_tensors_saved": False,
            "checked_shape_count": 3,
            "failed_shape_count": 0,
        }
        all_passed = (
            all_passed
            and replay.returncode == 0
            and worker_result.get("passed") is True
            and all(
                worker_result.get(field) == expected
                for field, expected in checks.items()
            )
        )
    return all_passed, evidence


def run_capture_adapter_validation(
    config_path: Path,
    sglang_worktree: Path,
) -> tuple[bool, list[str]]:
    run_dir = config_path.resolve().parent
    config = read_json_object(config_path, "capture config")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "model_adaptation_capture.capture_adapter_validation",
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
    if config.get("schema") == SESSION_CONFIG_SCHEMA:
        if completed.returncode != 0:
            return False, evidence
        return _run_session_replay_preflight(
            config_path,
            sglang_worktree,
            config,
            evidence,
        )
    if (
        completed.returncode != 0
        or config.get("operator_id") != SWIGLU_CLAMP_OPERATOR_ID
    ):
        return completed.returncode == 0, evidence

    try:
        rank_zero_fingerprint = _capture_artifact_fingerprint(run_dir)
    except PreflightError:
        return False, evidence
    rank_filter_config = json.loads(json.dumps(config))
    rank_filter_config["preflight_tp_context"] = {
        "rank": 1,
        "size": config["tensor_parallel_size"],
    }
    rank_filter_config_path = run_dir / "preflight-rank-filter-config.json"
    write_json(rank_filter_config_path, rank_filter_config)
    rank_filter = subprocess.run(
        [
            sys.executable,
            "-m",
            "model_adaptation_capture.capture_adapter_validation",
            "--mode",
            "capture",
            "--config",
            str(rank_filter_config_path.resolve()),
        ],
        cwd=sglang_worktree,
        env=_worker_environment(rank_filter_config_path, sglang_worktree),
        text=True,
        capture_output=True,
        check=False,
    )
    (run_dir / "preflight-rank-filter.log").write_text(
        "\n".join(
            [
                "worker_mode=rank-filter",
                f"returncode={rank_filter.returncode}",
                "--- stdout ---",
                rank_filter.stdout,
                "--- stderr ---",
                rank_filter.stderr,
            ]
        ),
        encoding="utf-8",
    )
    evidence.extend(
        [
            "preflight-rank-filter-config.json",
            "preflight-rank-filter.log",
        ]
    )
    rank_filter_summary = run_dir / "preflight-rank-filter-summary.json"
    if not rank_filter_summary.is_file():
        return False, evidence
    evidence.append("preflight-rank-filter-summary.json")
    rank_filter_result = read_json_object(
        rank_filter_summary,
        "rank-filter summary",
    )
    rank_filter_checks = {
        "applied_hooks": [SWIGLU_CLAMP_HOOK_TARGET],
        "runtime_tp_rank": 1,
        "capture_tp_rank": config["tp_rank"],
        "tensor_parallel_size": config["tensor_parallel_size"],
        "hook_call_count": 5,
        "checkpoint_loaded": False,
    }
    rank_filter_proved = rank_filter_result == rank_filter_checks
    try:
        rank_filter_preserved_capture = (
            _capture_artifact_fingerprint(run_dir) == rank_zero_fingerprint
        )
    except PreflightError:
        rank_filter_preserved_capture = False
    if (
        rank_filter.returncode != 0
        or not rank_filter_proved
        or not rank_filter_preserved_capture
    ):
        return False, evidence

    sample_file_records = []
    for sample_path in sorted((run_dir / "samples").glob("*.pt")):
        sample_file_records.append(
            {
                "shape_id": sample_path.stem,
                "path": str(sample_path.relative_to(run_dir)),
                "size": sample_path.stat().st_size,
                "sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
            }
        )
    sample_files_path = run_dir / "sample-files.json"
    write_json(
        sample_files_path,
        {
            "schema": "golden-sample-files/v1",
            "spec_binding": config["spec_binding"],
            "operator_id": config["operator_id"],
            "files": sample_file_records,
        },
    )
    sample_files_digest = hashlib.sha256(
        sample_files_path.read_bytes()
    ).hexdigest()
    evidence.append("sample-files.json")

    replay_config = {
        "schema": KERNEL_REPLAY_CONFIG_SCHEMA,
        "spec_binding": config["spec_binding"],
        "operator_id": config["operator_id"],
        "execution_site": "cuda",
        "invocation_target": SWIGLU_CLAMP_OPERATOR_ID,
        "golden_run": str(run_dir),
        "run_dir": str(run_dir),
        "tensor_parallel_size": config["tensor_parallel_size"],
        "tp_rank": config["tp_rank"],
        "precision_gate": config["precision_gate"],
        "sample_files_sha256": sample_files_digest,
        "allow_active_capture": True,
    }
    replay_config_path = run_dir / "preflight-replay-config.json"
    write_json(replay_config_path, replay_config)
    replay = subprocess.run(
        [
            sys.executable,
            "-m",
            "model_adaptation_capture.kernel_replay",
            "--config",
            str(replay_config_path.resolve()),
        ],
        cwd=sglang_worktree,
        env=_replay_worker_environment(config_path, sglang_worktree),
        text=True,
        capture_output=True,
        check=False,
    )
    (run_dir / "preflight-replay.log").write_text(
        "\n".join(
            [
                "worker_mode=cuda-self-replay",
                f"returncode={replay.returncode}",
                "--- stdout ---",
                replay.stdout,
                "--- stderr ---",
                replay.stderr,
            ]
        ),
        encoding="utf-8",
    )
    evidence.extend(
        [
            "preflight-replay-config.json",
            "preflight-replay.log",
        ]
    )
    worker_result_path = run_dir / "worker-result.json"
    if not worker_result_path.is_file():
        return False, evidence
    evidence.append("worker-result.json")
    worker_result = read_json_object(
        worker_result_path,
        "preflight replay result",
    )
    checks = {
        "schema": KERNEL_REPLAY_RESULT_SCHEMA,
        "spec_binding": config["spec_binding"],
        "operator_id": config["operator_id"],
        "execution_site": "cuda",
        "invocation_target": config["operator_id"],
        "tp_rank": config["tp_rank"],
        "tensor_parallel_size": config["tensor_parallel_size"],
        "precision_gate": config["precision_gate"],
        "sample_files_sha256": sample_files_digest,
        "actual_tensors_saved": False,
        "checked_shape_count": 3,
        "failed_shape_count": 0,
    }
    return (
        replay.returncode == 0
        and worker_result.get("passed") is True
        and all(worker_result.get(field) == expected for field, expected in checks.items())
    ), evidence


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
    if config.get("schema") == SESSION_CONFIG_SCHEMA:
        expected_hooks = [
            item["hook_target"] for item in config["operators"]
        ]
        missing_hooks = sorted(set(expected_hooks) - HookRegistry._patched)
        if missing_hooks:
            raise PreflightError(
                "SGLang did not apply session capture hooks: "
                + ", ".join(missing_hooks)
            )

        from sglang.srt.layers.attention.triton_ops.prefill_attention import (
            context_attention_fwd,
        )
        from sglang.srt.layers.layernorm import (
            gemma_fused_add_rmsnorm,
            gemma_rmsnorm,
        )
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            _swiglu_silu_clamp_mul,
        )
        from sglang.srt.layers.moe.topk import topk_sigmoid

        for item in config["operators"]:
            # The fixed source is the SGLang-side module that owns the capture
            # seam (the hook target the plugin patches), not the module where the
            # underlying kernel object is defined. Re-exported sgl_kernel calls
            # such as gemma_rmsnorm keep their __module__ inside the compiled
            # sgl_kernel package, so resolving call.__module__ would point at
            # sgl_kernel/elementwise.py instead of the pinned SGLang worktree
            # file. Resolve the loaded source from the hook target's module so
            # every operator is checked against the SGLang-side call site.
            seam_module = item["hook_target"].rsplit(".", 1)[0]
            loaded_source = Path(sys.modules[seam_module].__file__).resolve()
            expected_source = Path(
                item["source"]["capture_module_path"]
            ).resolve()
            if loaded_source != expected_source:
                raise PreflightError(
                    f"loaded source {loaded_source} for {item['operator_id']} "
                    f"does not match fixed source {expected_source}"
                )

        device = torch.device("cuda", torch.cuda.current_device())
        for rows in (1, 2, 4, 2, 8):
            swiglu_x = torch.linspace(
                -16,
                16,
                steps=rows * 16,
                dtype=torch.bfloat16,
                device=device,
            ).reshape(rows, 16)
            _swiglu_silu_clamp_mul(swiglu_x, 7.0)

            norm_x = torch.linspace(
                -4,
                4,
                steps=rows * 16,
                dtype=torch.bfloat16,
                device=device,
            ).reshape(rows, 16)
            weight = torch.linspace(
                -0.25,
                0.25,
                steps=16,
                dtype=torch.bfloat16,
                device=device,
            )
            gemma_rmsnorm(norm_x, weight, 1e-6)
            fused_x = norm_x.clone()
            residual = torch.ones_like(fused_x)
            gemma_fused_add_rmsnorm(fused_x, residual, weight, 1e-6)

            gating_output = torch.linspace(
                -3,
                3,
                steps=rows * 16,
                dtype=torch.float32,
                device=device,
            ).reshape(rows, 16)
            correction_bias = torch.linspace(
                -0.1,
                0.1,
                steps=16,
                dtype=torch.float32,
                device=device,
            )
            topk_weights = torch.empty(
                (rows, 8),
                dtype=torch.float32,
                device=device,
            )
            topk_ids = torch.empty(
                (rows, 8),
                dtype=torch.int32,
                device=device,
            )
            topk_sigmoid(
                topk_weights,
                topk_ids,
                gating_output,
                True,
                correction_bias,
            )

            token_count = rows * 64
            q = torch.linspace(
                -1,
                1,
                steps=token_count * 2 * 64,
                dtype=torch.bfloat16,
                device=device,
            ).reshape(token_count, 2, 64)
            k = q.clone()
            v = q.flip(0).contiguous()
            output = torch.empty_like(q)
            b_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
            b_seq_len = torch.tensor(
                [token_count],
                dtype=torch.int32,
                device=device,
            )
            context_attention_fwd(
                q,
                k,
                v,
                output,
                b_start_loc,
                b_seq_len,
                token_count,
                is_causal=False,
                sm_scale=None,
            )
        torch.cuda.synchronize(device)

        runtime_tp_rank = config["preflight_tp_context"]["rank"]
        if runtime_tp_rank != config["tp_rank"]:
            write_json(
                Path(config["run_dir"])
                / "preflight-rank-filter-summary.json",
                {
                    "applied_hooks": expected_hooks,
                    "runtime_tp_rank": runtime_tp_rank,
                    "capture_tp_rank": config["tp_rank"],
                    "tensor_parallel_size": config["tensor_parallel_size"],
                    "checkpoint_loaded": False,
                },
            )
            return

        saved_shape_counts = {}
        for item in config["operators"]:
            state = read_json_object(
                Path(item["run_dir"]) / "capture-state.json",
                f"rank-0 capture state for {item['operator_id']}",
            )
            expected_counts = {
                "saved_shape_count": 3,
                "repeated_call_count": 1,
                "skipped_call_count": 1,
            }
            for field, expected in expected_counts.items():
                if state.get(field) != expected:
                    raise PreflightError(
                        f"{item['operator_id']} capture state {field} must be "
                        f"{expected}, got {state.get(field)!r}"
                    )
            saved_shape_counts[item["operator_id"]] = 3
        write_json(
            Path(config["run_dir"]) / "preflight-capture-summary.json",
            {
                "torch_version": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(device),
                "applied_hooks": expected_hooks,
                "tp_rank": 0,
                "tensor_parallel_size": 8,
                "checkpoint_loaded": False,
                "saved_shape_counts": saved_shape_counts,
            },
        )
        return

    if config.get("operator_id") == SWIGLU_CLAMP_OPERATOR_ID:
        missing_hooks = sorted({SWIGLU_CLAMP_HOOK_TARGET} - HookRegistry._patched)
        if missing_hooks:
            raise PreflightError(
                "SGLang did not apply capture hooks: " + ", ".join(missing_hooks)
            )
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            _swiglu_silu_clamp_mul,
        )

        loaded_source = Path(
            sys.modules[_swiglu_silu_clamp_mul.__module__].__file__
        ).resolve()
        expected_source = Path(
            config["source"]["capture_module_path"]
        ).resolve()
        if loaded_source != expected_source:
            raise PreflightError(
                f"loaded Kernel Call source {loaded_source} does not match "
                f"fixed source {expected_source}"
            )
        device = torch.device("cuda", torch.cuda.current_device())
        for rows in (1, 2, 4):
            x = torch.linspace(
                -16,
                16,
                steps=rows * 16,
                dtype=torch.bfloat16,
                device=device,
            ).reshape(rows, 16)
            output = _swiglu_silu_clamp_mul(x, 7.0)
            if (
                output.shape != (rows, 8)
                or output.dtype != torch.bfloat16
                or not bool(torch.isfinite(output).all().item())
            ):
                raise PreflightError(
                    "existing SwiGLU clamp call did not return finite BF16 output"
                )
        _swiglu_silu_clamp_mul(
            torch.full(
                (2, 16),
                2.0,
                dtype=torch.bfloat16,
                device=device,
            ),
            7.0,
        )
        _swiglu_silu_clamp_mul(
            torch.ones(
                (8, 16),
                dtype=torch.bfloat16,
                device=device,
            ),
            7.0,
        )
        torch.cuda.synchronize(device)
        runtime_tp_rank = config["preflight_tp_context"]["rank"]
        if runtime_tp_rank != config["tp_rank"]:
            write_json(
                Path(config["run_dir"])
                / "preflight-rank-filter-summary.json",
                {
                    "applied_hooks": [SWIGLU_CLAMP_HOOK_TARGET],
                    "runtime_tp_rank": runtime_tp_rank,
                    "capture_tp_rank": config["tp_rank"],
                    "tensor_parallel_size": config["tensor_parallel_size"],
                    "hook_call_count": 5,
                    "checkpoint_loaded": False,
                },
            )
            return
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
                    f"capture state {field} must be {expected}, "
                    f"got {state.get(field)!r}"
                )
        write_json(
            Path(config["run_dir"]) / "preflight-capture-summary.json",
            {
                "torch_version": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(device),
                "loaded_kernel_source": str(loaded_source),
                "applied_hooks": [SWIGLU_CLAMP_HOOK_TARGET],
                "tp_rank": 0,
                "tensor_parallel_size": 8,
                "rank_zero_format_only": True,
                "checkpoint_loaded": False,
                "new_process_self_replay_performed": False,
                **expected_counts,
            },
        )
        return

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
        print(f"capture adapter validation worker: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
