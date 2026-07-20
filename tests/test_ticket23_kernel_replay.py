import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "model-adaptation" / "scripts"
sys.path.insert(0, str(SCRIPTS))
REVISION5_SPEC = (
    ROOT
    / "runs"
    / "handoff-build-r5-001"
    / "bundle"
    / "migration-spec.md"
)

from model_adaptation_capture.contracts import shape_id_for
from model_adaptation_capture.kernel_replay import (
    KernelReplayError,
    run_kernel_replay_worker,
)
import replay_compare


OPERATOR_ID = (
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe."
    "_swiglu_silu_clamp_mul"
)
BINDING = {
    "spec_id": "step3p7-flash-p800-demo",
    "contract_revision": 5,
    "contract_data_sha256": "fixture-sha256",
}
PRECISION = {
    "comparator": "torch.testing.assert_close",
    "atol": 0.01,
    "rtol": 0.02,
    "require_exact_structure": True,
    "require_same_dtype": True,
    "require_finite": True,
    "check_stride": False,
}


def write_golden_run(
    path: Path,
    *,
    binding: dict = BINDING,
    checkpoint: dict = None,
    status: str = "SEALED",
) -> None:
    if checkpoint is None:
        checkpoint = {
            "id": "stepfun-ai/Step-3.7-Flash@fixture",
            "model_path": "stepfun-ai/Step-3.7-Flash",
            "revision": "fixture",
            "config_digest": "config-digest",
        }
    samples = path / "samples"
    samples.mkdir(parents=True)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    expected = torch.tensor([[2.0, 4.0]], dtype=torch.bfloat16)
    shape_id = shape_id_for(x.shape)
    signature = {
        "operator_id": OPERATOR_ID,
        "model_path": "target",
        "tensor_parallel_size": 8,
        "tp_rank": 0,
        "inputs": {
            "x": {
                "kind": "tensor",
                "shape": [1, 4],
                "dtype": "bfloat16",
                "layout": "strided",
                "stride": [4, 1],
            }
        },
        "parameters": {},
        "non_tensor_args": {"gemm1_limit": 7.0},
    }
    sample_path = samples / f"{shape_id}.pt"
    torch.save(
        {
            "schema": "kernel-call-sample/v1",
            "spec_binding": binding,
            "operator_id": OPERATOR_ID,
            "tp_rank": 0,
            "tensor_parallel_size": 8,
            "signature": signature,
            "inputs": {"x": x},
            "parameters": {},
            "non_tensor_args": {"gemm1_limit": 7.0},
            "outputs": {"output": expected},
        },
        sample_path,
    )
    (path / "sample-files.json").write_text(
        json.dumps(
            {
                "schema": "golden-sample-files/v1",
                "spec_binding": binding,
                "operator_id": OPERATOR_ID,
                "files": [
                    {
                        "shape_id": shape_id,
                        "path": f"samples/{shape_id}.pt",
                        "size": sample_path.stat().st_size,
                        "sha256": hashlib.sha256(
                            sample_path.read_bytes()
                        ).hexdigest(),
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "capture-state.json").write_text(
        json.dumps(
            {
                "schema": "kernel-call-capture-state/v1",
                "spec_binding": binding,
                "operator_id": OPERATOR_ID,
                "tp_rank": 0,
                "tensor_parallel_size": 8,
                "checkpoint": checkpoint,
                "loaded_checkpoint": {
                    "model_path": checkpoint["model_path"],
                    "revision": checkpoint["revision"],
                },
                "status": status,
                "capture_closed": status != "ACTIVE",
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
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def replay_config(
    golden_run: Path,
    run_dir: Path,
    execution_site: str,
) -> dict:
    return {
        "schema": "kernel-call-replay-config/v1",
        "spec_binding": BINDING,
        "operator_id": OPERATOR_ID,
        "execution_site": execution_site,
        "invocation_target": (
            OPERATOR_ID if execution_site == "cuda" else "kunlun_ops.swiglu"
        ),
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


class Ticket23KernelReplayTest(unittest.TestCase):
    def test_cuda_and_p800_use_distinct_existing_call_seams(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            golden_run = workspace / "golden-001"
            write_golden_run(golden_run)

            calls = []

            def cuda_call(x, gemm1_limit):
                calls.append(("cuda", gemm1_limit))
                return torch.tensor([[2.0, 4.0]], dtype=x.dtype)

            def p800_call(x, gemm1_limit):
                calls.append(("p800", gemm1_limit))
                return torch.tensor([[2.0, 4.0]], dtype=x.dtype)

            cuda_run = workspace / "cuda-replay"
            cuda_result = run_kernel_replay_worker(
                replay_config(golden_run, cuda_run, "cuda"),
                cuda_call=cuda_call,
                p800_call=p800_call,
                device=torch.device("cpu"),
            )
            p800_run = workspace / "p800-replay"
            p800_result = run_kernel_replay_worker(
                replay_config(golden_run, p800_run, "p800"),
                cuda_call=cuda_call,
                p800_call=p800_call,
                device=torch.device("cpu"),
            )

            self.assertTrue(cuda_result["passed"])
            self.assertTrue(p800_result["passed"])
            self.assertEqual(calls, [("cuda", 7.0), ("p800", 7.0)])
            self.assertEqual(
                cuda_result["invocation_target"],
                OPERATOR_ID,
            )
            self.assertEqual(
                p800_result["invocation_target"],
                "kunlun_ops.swiglu",
            )
            self.assertFalse(list(workspace.rglob("*actual*.pt")))

    def test_precision_failure_is_recorded_without_saving_actual_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            golden_run = workspace / "golden-001"
            write_golden_run(golden_run)
            run_dir = workspace / "p800-replay"

            result = run_kernel_replay_worker(
                replay_config(golden_run, run_dir, "p800"),
                cuda_call=lambda x, limit: x[..., :2],
                p800_call=lambda x, limit: torch.zeros(
                    (1, 2),
                    dtype=x.dtype,
                ),
                device=torch.device("cpu"),
            )

            self.assertFalse(result["passed"])
            self.assertEqual(result["checked_shape_count"], 1)
            self.assertEqual(result["failed_shape_count"], 1)
            self.assertEqual(result["errors"][0]["type"], "AssertionError")
            self.assertFalse(list(run_dir.rglob("*.pt")))

    def test_p800_invocation_failure_is_recorded_as_an_operator_gap(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            golden_run = workspace / "golden-001"
            write_golden_run(golden_run)
            run_dir = workspace / "p800-replay"

            def unsupported_call(_x, _limit):
                raise RuntimeError("P800 SwiGLU invocation is unsupported")

            result = run_kernel_replay_worker(
                replay_config(golden_run, run_dir, "p800"),
                p800_call=unsupported_call,
                device=torch.device("cpu"),
            )

            self.assertFalse(result["passed"])
            self.assertEqual(result["failed_shape_count"], 1)
            self.assertEqual(result["errors"][0]["type"], "RuntimeError")
            self.assertIn("unsupported", result["errors"][0]["message"])

    def test_missing_p800_dependency_is_a_tool_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            golden_run = workspace / "golden-001"
            write_golden_run(golden_run)
            run_dir = workspace / "p800-replay"

            def missing_dependency(_x, _limit):
                raise ImportError("kunlun_ops is unavailable")

            with self.assertRaisesRegex(ImportError, "kunlun_ops"):
                run_kernel_replay_worker(
                    replay_config(golden_run, run_dir, "p800"),
                    p800_call=missing_dependency,
                    device=torch.device("cpu"),
                )

            self.assertFalse((run_dir / "worker-result.json").exists())

    def test_comparator_runtime_failure_is_a_tool_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            golden_run = workspace / "golden-001"
            write_golden_run(golden_run)
            run_dir = workspace / "p800-replay"

            with patch.object(
                torch.testing,
                "assert_close",
                side_effect=RuntimeError("comparator unavailable"),
            ), self.assertRaisesRegex(RuntimeError, "comparator unavailable"):
                run_kernel_replay_worker(
                    replay_config(golden_run, run_dir, "p800"),
                    p800_call=lambda x, limit: torch.tensor(
                        [[2.0, 4.0]],
                        dtype=x.dtype,
                    ),
                    device=torch.device("cpu"),
                )

            self.assertFalse((run_dir / "worker-result.json").exists())

    def test_rejects_non_tensor_argument_drift_between_state_and_payload(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            golden_run = workspace / "golden-001"
            write_golden_run(golden_run)
            sample_path = next((golden_run / "samples").glob("*.pt"))
            payload = torch.load(
                sample_path,
                map_location="cpu",
                weights_only=True,
            )
            payload["non_tensor_args"]["gemm1_limit"] = 6.0
            torch.save(payload, sample_path)

            with self.assertRaisesRegex(KernelReplayError, "signature"):
                run_kernel_replay_worker(
                    replay_config(
                        golden_run,
                        workspace / "cuda-replay",
                        "cuda",
                    ),
                    cuda_call=lambda x, limit: torch.tensor(
                        [[2.0, 4.0]],
                        dtype=x.dtype,
                    ),
                    p800_call=lambda x, limit: x[..., :2],
                    device=torch.device("cpu"),
                )

    def test_replay_tool_seals_cuda_golden_then_runs_p800_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            scan = json.loads(
                (ROOT / "runs" / "scan-006" / "result.json").read_text(
                    encoding="utf-8"
                )
            )
            binding = scan["spec_binding"]
            checkpoint = {
                "id": scan["source"]["checkpoint_id"],
                "model_path": scan["source"]["checkpoint_id"].rsplit("@", 1)[0],
                "revision": scan["source"]["checkpoint_id"].rsplit("@", 1)[1],
                "config_digest": scan["source"]["checkpoint_config_sha256"],
            }
            golden_run = workspace / "golden-001"
            write_golden_run(
                golden_run,
                binding=binding,
                checkpoint=checkpoint,
                status="ACTIVE",
            )
            worktree = workspace / "sglang"
            worktree.mkdir()

            def run_worker(command, **kwargs):
                config_path = Path(command[-1])
                config = json.loads(config_path.read_text(encoding="utf-8"))
                result = run_kernel_replay_worker(
                    config,
                    cuda_call=lambda x, limit: torch.tensor(
                        [[2.0, 4.0]],
                        dtype=x.dtype,
                    ),
                    p800_call=lambda x, limit: torch.tensor(
                        [[2.0, 4.0]],
                        dtype=x.dtype,
                    ),
                    device=torch.device("cpu"),
                )
                return subprocess.CompletedProcess(
                    command,
                    0 if result["passed"] else 1,
                    stdout="",
                    stderr="",
                )

            cuda_run = workspace / "cuda-self-replay-001"
            with patch.object(
                replay_compare,
                "resolve_git_revision",
                return_value="49e384ce9d304648e9959666ecb8ce8cd98d0deb",
            ), patch.object(replay_compare.subprocess, "run", side_effect=run_worker):
                replay_compare.run_kernel_replay(
                    REVISION5_SPEC,
                    cuda_run,
                    ROOT / "runs" / "scan-006" / "result.json",
                    OPERATOR_ID,
                    golden_run,
                    execution_site="cuda",
                    sglang_worktree=worktree,
                )

            cuda_result = json.loads(
                (cuda_run / "result.json").read_text(encoding="utf-8")
            )
            self.assertTrue(cuda_result["passed"])
            self.assertEqual(cuda_result["execution_site"], "cuda")
            sealed_state = json.loads(
                (golden_run / "capture-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(sealed_state["status"], "SEALED")
            self.assertTrue(sealed_state["capture_closed"])
            self.assertTrue(sealed_state["self_replay"]["passed"])

            p800_run = workspace / "p800-baseline-001"
            with patch.object(replay_compare.subprocess, "run", side_effect=run_worker):
                replay_compare.run_kernel_replay(
                    REVISION5_SPEC,
                    p800_run,
                    ROOT / "runs" / "scan-006" / "result.json",
                    OPERATOR_ID,
                    golden_run,
                    execution_site="p800",
                )

            p800_result = json.loads(
                (p800_run / "result.json").read_text(encoding="utf-8")
            )
            self.assertTrue(p800_result["passed"])
            self.assertEqual(
                p800_result["invocation_target"],
                "kunlun_ops.swiglu",
            )
            self.assertEqual(
                json.loads(
                    (golden_run / "capture-state.json").read_text(
                        encoding="utf-8"
                    )
                ),
                sealed_state,
            )

            sealed_state["self_replay"]["sample_files_sha256"] = "0" * 64
            (golden_run / "capture-state.json").write_text(
                json.dumps(sealed_state, indent=2) + "\n",
                encoding="utf-8",
            )
            rejected_run = workspace / "p800-baseline-wrong-samples"
            with self.assertRaisesRegex(
                replay_compare.ToolError,
                "sample_files_sha256",
            ):
                replay_compare.run_kernel_replay(
                    REVISION5_SPEC,
                    rejected_run,
                    ROOT / "runs" / "scan-006" / "result.json",
                    OPERATOR_ID,
                    golden_run,
                    execution_site="p800",
                )
            self.assertFalse(rejected_run.exists())


if __name__ == "__main__":
    unittest.main()
