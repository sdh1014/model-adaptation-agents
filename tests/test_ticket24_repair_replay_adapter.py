"""Keep candidate replay on the original Kunlun Kernel Call boundary."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "model-adaptation" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from model_adaptation_capture.contracts import (
    KUNLUN_SWIGLU_TARGET,
    SWIGLU_KUNLUN_SOURCE_RELATIVE_PATH,
)
from model_adaptation_capture.kernel_replay import (
    KernelReplayError,
    run_kernel_replay_worker,
)
import replay_compare
from test_ticket23_kernel_replay import (
    BINDING,
    OPERATOR_ID,
    PRECISION,
    write_golden_run,
)


CORRECTION_RUN = ROOT / "runs" / "inline-repair-correction-r5-001"
OLD_ATTEMPT = ROOT / "runs" / "repair-attempt-1-r5-001"


def _clamped(x: torch.Tensor, limit: float) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = torch.nn.functional.silu(x[..., :d].float()).clamp(max=limit)
    linear = x[..., d:].float().clamp(min=-limit, max=limit)
    return (gate * linear).to(x.dtype)


def write_clamp_golden_run(path: Path, *, limit: float = 7.0) -> None:
    from model_adaptation_capture.contracts import shape_id_for

    checkpoint = {
        "id": "stepfun-ai/Step-3.7-Flash@fixture",
        "model_path": "stepfun-ai/Step-3.7-Flash",
        "revision": "fixture",
        "config_digest": "config-digest",
    }
    samples = path / "samples"
    samples.mkdir(parents=True)
    x = torch.tensor([[1.0, 2.0, 9.0, 4.0]], dtype=torch.bfloat16)
    expected = _clamped(x, limit)
    shape_id = shape_id_for(x.shape)
    signature = {
        "operator_id": OPERATOR_ID,
        "model_path": "target",
        "tensor_parallel_size": 8,
        "tp_rank": 0,
        "inputs": {
            "x": {
                "kind": "tensor",
                "shape": list(x.shape),
                "dtype": "bfloat16",
                "layout": "strided",
                "stride": list(x.stride()),
            }
        },
        "parameters": {},
        "non_tensor_args": {"gemm1_limit": float(limit)},
    }
    sample_path = samples / f"{shape_id}.pt"
    torch.save(
        {
            "schema": "kernel-call-sample/v1",
            "spec_binding": BINDING,
            "operator_id": OPERATOR_ID,
            "tp_rank": 0,
            "tensor_parallel_size": 8,
            "signature": signature,
            "inputs": {"x": x},
            "parameters": {},
            "non_tensor_args": {"gemm1_limit": float(limit)},
            "outputs": {"output": expected},
        },
        sample_path,
    )
    sidecar = {
        "schema": "golden-sample-files/v1",
        "spec_binding": BINDING,
        "operator_id": OPERATOR_ID,
        "files": [
            {
                "shape_id": shape_id,
                "path": f"samples/{shape_id}.pt",
                "size": sample_path.stat().st_size,
                "sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
            }
        ],
    }
    sidecar_path = path / "sample-files.json"
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar_sha = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    (path / "capture-state.json").write_text(
        json.dumps(
            {
                "schema": "kernel-call-capture-state/v1",
                "spec_binding": BINDING,
                "operator_id": OPERATOR_ID,
                "tp_rank": 0,
                "tensor_parallel_size": 8,
                "checkpoint": checkpoint,
                "loaded_checkpoint": {
                    "model_path": checkpoint["model_path"],
                    "revision": checkpoint["revision"],
                },
                "status": "SEALED",
                "capture_closed": True,
                "saved_shape_count": 1,
                "repeated_call_count": 0,
                "skipped_call_count": 0,
                "samples": [
                    {
                        "shape_id": shape_id,
                        "file": f"samples/{shape_id}.pt",
                        "signature": signature,
                        "repeat_count": 0,
                    }
                ],
                "skipped_signatures": [],
                "self_replay": {
                    "passed": True,
                    "checked_shape_count": 1,
                    "sample_files_sha256": sidecar_sha,
                    "worker_result_sha256": "0" * 64,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def candidate_metadata(root: Path, *, relevant: bool = True) -> dict:
    root.mkdir(parents=True)
    worktree = root / "sglang-kunlun-worktree"
    source_path = worktree / SWIGLU_KUNLUN_SOURCE_RELATIVE_PATH
    source_path.parent.mkdir(parents=True)
    baseline_source = (
        "def unquantized_fused_moe_apply_kunlun(layer, y):\n"
        "    out1 = y\n"
        "    kunlun_ops.swiglu(x=y, y=out1)\n"
        "    return out1\n"
    )
    source_path.write_text(baseline_source, encoding="utf-8")
    (worktree / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=worktree, check=True)
    subprocess.run(
        ["git", "config", "user.email", "candidate@example.com"],
        cwd=worktree,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Candidate Test"],
        cwd=worktree,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=worktree, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "baseline"],
        cwd=worktree,
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        text=True,
    ).strip()
    if relevant:
        source_path.write_text(
            (
                "def unquantized_fused_moe_apply_kunlun(layer, y):\n"
                "    out1 = y\n"
                "    limit = layer.moe_runner_config.gemm1_clamp_limit\n"
                "    if limit is None:\n"
                "        kunlun_ops.swiglu(x=y, y=out1)\n"
                "    else:\n"
                "        kunlun_ops.swiglu(\n"
                "            x=y, y=out1, limit=float(limit)\n"
                "        )\n"
                "    return out1\n"
            ),
            encoding="utf-8",
        )
        modified_paths = [SWIGLU_KUNLUN_SOURCE_RELATIVE_PATH]
    else:
        (worktree / "README.md").write_text(
            "unrelated candidate\n",
            encoding="utf-8",
        )
        modified_paths = ["README.md"]
    patch = subprocess.check_output(
        [
            "git",
            "diff",
            "--binary",
            "--no-ext-diff",
            revision,
            "--",
            *modified_paths,
        ],
        cwd=worktree,
    )
    result_path = root / "candidate-result.json"
    patch_path = root / "candidate.patch"
    patch_path.write_bytes(patch)
    result = {
        "tool": "workspace_guard.py",
        "action": "record-candidate",
        "spec_binding": BINDING,
        "baseline_revision": revision,
        "worktree": str(worktree.resolve()),
        "operator_id": OPERATOR_ID,
        "allowed_paths": modified_paths,
        "modified_paths": modified_paths,
        "passed": True,
        "workspace_state": "CANDIDATE_RECORDED",
        "patch_path": "candidate.patch",
        "patch_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "candidate_operator_id": OPERATOR_ID,
        "modified_paths": modified_paths,
        "result": str(result_path.resolve()),
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "patch_sha256": result["patch_sha256"],
        "sglang_kunlun_worktree": result["worktree"],
        "sglang_kunlun_revision": revision,
    }
    if relevant:
        metadata.update(
            {
                "original_call_source": SWIGLU_KUNLUN_SOURCE_RELATIVE_PATH,
                "original_call_source_sha256": hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest(),
                "original_call_arguments": {
                    "x": "y",
                    "y": "out1",
                    "limit": (
                        "float(layer.moe_runner_config.gemm1_clamp_limit)"
                    ),
                },
                "original_call_none_fallback": True,
            }
        )
    return metadata


def replay_config(
    golden_run: Path,
    run_dir: Path,
    *,
    candidate: dict = None,
) -> dict:
    config = {
        "schema": "kernel-call-replay-config/v1",
        "spec_binding": BINDING,
        "operator_id": OPERATOR_ID,
        "execution_site": "p800",
        "invocation_target": KUNLUN_SWIGLU_TARGET,
        "golden_run": str(golden_run),
        "run_dir": str(run_dir),
        "tensor_parallel_size": 8,
        "tp_rank": 0,
        "precision_gate": PRECISION,
        "sample_files_sha256": hashlib.sha256(
            (golden_run / "sample-files.json").read_bytes()
        ).hexdigest(),
        "allow_active_capture": False,
    }
    if candidate is not None:
        config["candidate"] = candidate
    return config


class OriginalBoundaryReplayTest(unittest.TestCase):
    def test_candidate_passes_limit_to_existing_kunlun_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            golden_run = workspace / "golden"
            write_clamp_golden_run(golden_run)
            candidate = candidate_metadata(workspace / "attempt")
            calls = []
            module = types.ModuleType("kunlun_ops")

            def swiglu(*, x, y, limit=None):
                calls.append(limit)
                if limit is None:
                    d = x.shape[-1] // 2
                    y.copy_(
                        (
                            torch.nn.functional.silu(x[..., :d].float())
                            * x[..., d:].float()
                        ).to(x.dtype)
                    )
                else:
                    y.copy_(_clamped(x, float(limit)))

            module.swiglu = swiglu
            with patch.dict(sys.modules, {"kunlun_ops": module}):
                baseline = run_kernel_replay_worker(
                    replay_config(
                        golden_run,
                        workspace / "baseline",
                    ),
                    device=torch.device("cpu"),
                )
                repaired = run_kernel_replay_worker(
                    replay_config(
                        golden_run,
                        workspace / "candidate-replay",
                        candidate=candidate,
                    ),
                    device=torch.device("cpu"),
                )

            self.assertFalse(baseline["passed"])
            self.assertTrue(repaired["passed"])
            self.assertEqual(calls, [None, 7.0])
            self.assertEqual(
                repaired["invocation_target"],
                KUNLUN_SWIGLU_TARGET,
            )

    def test_arbitrary_callable_protocol_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            golden_run = workspace / "golden"
            write_golden_run(golden_run)
            config = replay_config(golden_run, workspace / "replay")
            config["invocation_target"] = (
                "repair-kernel-call/v1:module:callable"
            )
            with self.assertRaisesRegex(
                KernelReplayError,
                "invocation target",
            ):
                run_kernel_replay_worker(
                    config,
                    p800_call=lambda x, limit: x[..., :2],
                    device=torch.device("cpu"),
                )

    def test_candidate_metadata_is_all_or_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            golden_run = workspace / "golden"
            write_golden_run(golden_run)
            candidate = candidate_metadata(workspace / "attempt")
            del candidate["patch_sha256"]
            with self.assertRaisesRegex(
                KernelReplayError,
                "metadata has drifted",
            ):
                run_kernel_replay_worker(
                    replay_config(
                        golden_run,
                        workspace / "replay",
                        candidate=candidate,
                    ),
                    p800_call=lambda x, limit: x[..., :2],
                    device=torch.device("cpu"),
                )

    def test_retired_repair_callable_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            golden_run = workspace / "golden"
            write_golden_run(golden_run)
            config = replay_config(golden_run, workspace / "replay")
            config["repair_entry"] = "module:callable"
            with self.assertRaisesRegex(
                KernelReplayError,
                "retired repair callable protocol",
            ):
                run_kernel_replay_worker(
                    config,
                    p800_call=lambda x, limit: x[..., :2],
                    device=torch.device("cpu"),
                )


class CandidateBindingTest(unittest.TestCase):
    def test_replay_tool_binds_the_recorded_candidate_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            attempt = workspace / "attempt"
            candidate = candidate_metadata(attempt)
            loaded = replay_compare.load_candidate_replay_metadata(
                Path(candidate["result"]),
                binding=BINDING,
                worktree=Path(candidate["sglang_kunlun_worktree"]),
                baseline_revision=candidate["sglang_kunlun_revision"],
            )
            self.assertEqual(loaded, candidate)

            (attempt / "candidate.patch").write_text(
                "changed after recording\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                replay_compare.ToolError,
                "SHA-256",
            ):
                replay_compare.load_candidate_replay_metadata(
                    Path(candidate["result"]),
                    binding=BINDING,
                    worktree=Path(candidate["sglang_kunlun_worktree"]),
                    baseline_revision=candidate["sglang_kunlun_revision"],
                )

    def test_unrelated_patch_cannot_enable_candidate_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = candidate_metadata(
                Path(temp_dir) / "attempt",
                relevant=False,
            )
            with self.assertRaisesRegex(
                replay_compare.ToolError,
                "original SwiGLU production call site",
            ):
                replay_compare.load_candidate_replay_metadata(
                    Path(candidate["result"]),
                    binding=BINDING,
                    worktree=Path(candidate["sglang_kunlun_worktree"]),
                    baseline_revision=candidate["sglang_kunlun_revision"],
                )

    def test_live_worktree_must_still_match_candidate_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = candidate_metadata(Path(temp_dir) / "attempt")
            source_path = (
                Path(candidate["sglang_kunlun_worktree"])
                / SWIGLU_KUNLUN_SOURCE_RELATIVE_PATH
            )
            source_path.write_text(
                source_path.read_text(encoding="utf-8")
                + "# changed after candidate recording\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                replay_compare.ToolError,
                "worktree bytes",
            ):
                replay_compare.load_candidate_replay_metadata(
                    Path(candidate["result"]),
                    binding=BINDING,
                    worktree=Path(candidate["sglang_kunlun_worktree"]),
                    baseline_revision=candidate["sglang_kunlun_revision"],
                )

    def test_corrected_patch_is_inline_and_old_run_is_immutable(self) -> None:
        corrected = (CORRECTION_RUN / "candidate.patch").read_text(
            encoding="utf-8"
        )
        historical = (OLD_ATTEMPT / "candidate.patch").read_text(
            encoding="utf-8"
        )
        self.assertIn("def unquantized_fused_moe_apply_kunlun(", corrected)
        self.assertIn(
            "kunlun_ops.swiglu(x=y, y=out1, limit=float(gemm1_limit))",
            corrected,
        )
        self.assertNotIn("def apply_gemm1_swiglu_clamp", corrected)
        self.assertIn("def apply_gemm1_swiglu_clamp", historical)

    def test_flow_has_no_arbitrary_repair_callable_protocol(self) -> None:
        source_paths = (
            SCRIPTS / "model_adaptation_capture" / "contracts.py",
            SCRIPTS / "model_adaptation_capture" / "kernel_replay.py",
            SCRIPTS / "replay_compare.py",
            SCRIPTS / "workspace_guard.py",
        )
        source = "\n".join(
            path.read_text(encoding="utf-8") for path in source_paths
        )
        self.assertNotIn("repair-kernel-call/v1", source)
        self.assertNotIn("def normalize_repair_entry", source)
        self.assertNotIn("_resolve_repair_call", source)
        self.assertIn("retired repair callable protocol", source)


if __name__ == "__main__":
    unittest.main()
