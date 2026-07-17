import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "model-adaptation" / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from model_adaptation_capture.collector import CandidateCollector, CaptureError


LOADED_CHECKPOINT = {
    "model_path": "stepfun-ai/Step-3.7-Flash",
    "revision": "fixture",
}


def write_capture_config(path: Path, run_dir: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "golden-capture-config/v1",
                "spec_binding": {
                    "spec_id": "step3p7-flash-p800-demo",
                    "contract_revision": 4,
                    "contract_data_sha256": "fixture-sha256",
                },
                "operator_id": "Step3p5MLP.forward",
                "activation_guard": "self.limit is not None",
                "model_path": "target",
                "max_shapes": 3,
                "tensor_parallel_size": 8,
                "tp_rank": 0,
                "serialization": "candidate-torch-save/v2",
                "run_dir": str(run_dir),
                "capture_device_type": "cpu",
                "dtype": "bfloat16",
                "checkpoint": {
                    "id": "stepfun-ai/Step-3.7-Flash@fixture",
                    "model_path": "stepfun-ai/Step-3.7-Flash",
                    "revision": "fixture",
                    "config_digest": "config-digest",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


class Ticket12CaptureCandidateTest(unittest.TestCase):
    def test_rejects_parameter_and_non_finite_boundary_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "golden-001"
            run_dir.mkdir()
            config_path = run_dir / "capture-config.json"
            write_capture_config(config_path, run_dir)
            collector = CandidateCollector.from_config_path(
                config_path,
                tp_rank=0,
                tp_size=8,
                loaded_checkpoint=LOADED_CHECKPOINT,
            )

            parameter = torch.nn.Parameter(
                torch.ones((1, 8), dtype=torch.bfloat16),
                requires_grad=False,
            )
            with self.assertRaisesRegex(
                CaptureError,
                "nn.Parameter inputs are forbidden",
            ):
                collector.record(
                    parameter,
                    parameter.detach() * 2,
                    model_instance_path="model.layers.43.share_expert",
                    limit=16.0,
                    execution_phase="decode",
                )

            non_finite = torch.ones((1, 8), dtype=torch.bfloat16)
            non_finite[0, 0] = float("nan")
            with self.assertRaisesRegex(CaptureError, "non-finite"):
                collector.record(
                    non_finite,
                    non_finite * 2,
                    model_instance_path="model.layers.43.share_expert",
                    limit=16.0,
                    execution_phase="decode",
                )

            state = json.loads(
                (run_dir / "capture-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["saved_shape_count"], 0)


if __name__ == "__main__":
    unittest.main()
