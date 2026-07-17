import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "model-adaptation" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from model_adaptation_capture.kernel_call import (
    KernelCallCollector,
    KernelCallError,
)


OPERATOR_ID = (
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe."
    "_swiglu_silu_clamp_mul"
)
CHECKPOINT = {
    "id": "stepfun-ai/Step-3.7-Flash@fixture",
    "model_path": "stepfun-ai/Step-3.7-Flash",
    "revision": "fixture",
    "config_digest": "config-digest",
}
LOADED_CHECKPOINT = {
    "model_path": "stepfun-ai/Step-3.7-Flash",
    "revision": "fixture",
}


def write_config(path: Path, run_dir: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "golden-capture-config/v1",
                "spec_binding": {
                    "spec_id": "step3p7-flash-p800-demo",
                    "contract_revision": 5,
                    "contract_data_sha256": "fixture-sha256",
                },
                "operator_id": OPERATOR_ID,
                "activation_guard": (
                    "unquantized routed MoE layers 43 and 44 with "
                    "gemm1_clamp_limit=7 and the CUDA legacy Triton runner"
                ),
                "model_path": "target",
                "max_shapes": 3,
                "tensor_parallel_size": 8,
                "tp_rank": 0,
                "serialization": "kernel-call-torch-save/v1",
                "capture_device_type": "cpu",
                "dtype": "bfloat16",
                "checkpoint": CHECKPOINT,
                "precision_gate": {
                    "comparator": "torch.testing.assert_close",
                    "atol": 0.01,
                    "rtol": 0.02,
                    "require_exact_structure": True,
                    "require_same_dtype": True,
                    "require_finite": True,
                    "check_stride": False,
                },
                "run_dir": str(run_dir),
                "hook_target": OPERATOR_ID,
                "boundary": {
                    "inputs": ["x"],
                    "parameters": [],
                    "non_tensor_args": ["gemm1_limit"],
                    "outputs": ["output"],
                },
                "sample_fields": {
                    "inputs": ["x"],
                    "parameters": [],
                    "non_tensor_args": ["gemm1_limit"],
                    "outputs": ["output"],
                },
                "replay": {
                    "mode": "standalone-kernel-call",
                    "cuda_target": OPERATOR_ID,
                    "p800_target": "kunlun_ops.swiglu",
                    "weights_in_golden_sample": False,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


class Ticket23KernelCallRuntimeTest(unittest.TestCase):
    def test_rank_zero_saves_only_three_swiglu_shapes_without_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "golden-001"
            run_dir.mkdir()
            config_path = run_dir / "capture-config.json"
            write_config(config_path, run_dir)

            rank_one = KernelCallCollector.from_config_path(
                config_path,
                tp_rank=1,
                tp_size=8,
                loaded_checkpoint=LOADED_CHECKPOINT,
            )
            x = torch.ones((1, 16), dtype=torch.bfloat16)
            rank_one.record(x, x[..., :8], gemm1_limit=7.0)
            self.assertFalse((run_dir / "capture-state.json").exists())

            collector = KernelCallCollector.from_config_path(
                config_path,
                tp_rank=0,
                tp_size=8,
                loaded_checkpoint=LOADED_CHECKPOINT,
            )
            for rows in (1, 2, 4):
                x = torch.arange(
                    rows * 16,
                    dtype=torch.bfloat16,
                ).reshape(rows, 16)
                collector.record(x, x[..., :8] * 2, gemm1_limit=7.0)

            duplicate = torch.ones((2, 16), dtype=torch.bfloat16)
            collector.record(
                duplicate,
                duplicate[..., :8],
                gemm1_limit=7.0,
            )
            fourth = torch.ones((8, 16), dtype=torch.bfloat16)
            collector.record(fourth, fourth[..., :8], gemm1_limit=7.0)

            state = json.loads(
                (run_dir / "capture-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["saved_shape_count"], 3)
            self.assertEqual(state["repeated_call_count"], 1)
            self.assertEqual(state["skipped_call_count"], 1)
            self.assertEqual(state["tp_rank"], 0)
            self.assertEqual(state["loaded_checkpoint"], LOADED_CHECKPOINT)
            self.assertFalse((run_dir / "ranks").exists())

            sample_paths = sorted((run_dir / "samples").glob("*.pt"))
            self.assertEqual(len(sample_paths), 3)
            payload = torch.load(
                sample_paths[0],
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(payload["schema"], "kernel-call-sample/v1")
            self.assertEqual(payload["operator_id"], OPERATOR_ID)
            self.assertEqual(set(payload["inputs"]), {"x"})
            self.assertEqual(payload["parameters"], {})
            self.assertEqual(
                payload["non_tensor_args"],
                {"gemm1_limit": 7.0},
            )
            self.assertEqual(set(payload["outputs"]), {"output"})
            self.assertNotIn("weight", repr(payload).lower())
            self.assertNotIn("checkpoint", repr(payload).lower())

    def test_rejects_parameter_or_non_finite_kernel_boundary_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "golden-001"
            run_dir.mkdir()
            config_path = run_dir / "capture-config.json"
            write_config(config_path, run_dir)
            collector = KernelCallCollector.from_config_path(
                config_path,
                tp_rank=0,
                tp_size=8,
                loaded_checkpoint=LOADED_CHECKPOINT,
            )

            parameter = torch.nn.Parameter(
                torch.ones((1, 16), dtype=torch.bfloat16),
                requires_grad=False,
            )
            with self.assertRaisesRegex(KernelCallError, "Parameter"):
                collector.record(
                    parameter,
                    parameter.detach()[..., :8],
                    gemm1_limit=7.0,
                )

            non_finite = torch.ones((1, 16), dtype=torch.bfloat16)
            non_finite[0, 0] = float("nan")
            with self.assertRaisesRegex(KernelCallError, "non-finite"):
                collector.record(
                    non_finite,
                    non_finite[..., :8],
                    gemm1_limit=7.0,
                )


if __name__ == "__main__":
    unittest.main()
