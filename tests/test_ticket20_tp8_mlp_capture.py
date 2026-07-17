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


OPERATOR_ID = "Step3p5MLP.forward"
MODEL_INSTANCE_PATH = "model.layers.43.share_expert"
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
                "operator_id": OPERATOR_ID,
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


class Ticket20Tp8MlpCaptureTest(unittest.TestCase):
    def test_tp8_capture_saves_only_rank_zero_for_three_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "golden-001"
            run_dir.mkdir()
            config_path = run_dir / "capture-config.json"
            write_capture_config(config_path, run_dir)

            non_capture_rank = CandidateCollector.from_config_path(
                config_path,
                tp_rank=1,
                tp_size=8,
                loaded_checkpoint=LOADED_CHECKPOINT,
            )
            ignored_x = torch.ones((1, 8), dtype=torch.bfloat16)
            non_capture_rank.record(
                ignored_x,
                ignored_x * 2,
                model_instance_path=MODEL_INSTANCE_PATH,
                limit=16.0,
                execution_phase="decode",
            )
            self.assertFalse((run_dir / "capture-state.json").exists())
            self.assertEqual(list((run_dir / "samples").glob("*.pt")), [])

            collector = CandidateCollector.from_config_path(
                config_path,
                tp_rank=0,
                tp_size=8,
                loaded_checkpoint=LOADED_CHECKPOINT,
            )
            first_shape = torch.arange(
                8,
                dtype=torch.bfloat16,
            ).reshape(1, 8)
            collector.record(
                first_shape,
                first_shape * 2,
                model_instance_path=MODEL_INSTANCE_PATH,
                limit=16.0,
                execution_phase="decode",
            )
            collector.record(
                first_shape,
                first_shape * 2,
                model_instance_path="model.layers.44.share_expert",
                limit=16.0,
                execution_phase="prefill",
            )

            for rows in (2, 4):
                x = torch.arange(
                    rows * 8,
                    dtype=torch.bfloat16,
                ).reshape(rows, 8)
                collector.record(
                    x,
                    x * 2,
                    model_instance_path=MODEL_INSTANCE_PATH,
                    limit=16.0,
                    execution_phase="decode",
                )

            fourth = torch.ones((8, 8), dtype=torch.bfloat16)
            collector.record(
                fourth,
                fourth * 2,
                model_instance_path=MODEL_INSTANCE_PATH,
                limit=16.0,
                execution_phase="decode",
            )

            sample_paths = sorted((run_dir / "samples").glob("*.pt"))
            self.assertEqual(len(sample_paths), 3)
            self.assertFalse((run_dir / "ranks").exists())

            state = json.loads(
                (run_dir / "capture-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["tp_rank"], 0)
            self.assertEqual(state["tensor_parallel_size"], 8)
            self.assertEqual(state["saved_shape_count"], 3)
            self.assertEqual(state["repeated_call_count"], 1)
            self.assertEqual(state["skipped_call_count"], 1)
            self.assertEqual(state["status"], "ACTIVE")
            self.assertFalse(state["capture_closed"])
            self.assertEqual(state["loaded_checkpoint"], LOADED_CHECKPOINT)
            self.assertTrue(
                all("shape_id" in sample for sample in state["samples"])
            )
            self.assertTrue(
                all("signature_id" not in sample for sample in state["samples"])
            )

            payload = torch.load(
                sample_paths[0],
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(payload["schema"], "capture-candidate/v1")
            self.assertEqual(payload["operator_id"], OPERATOR_ID)
            self.assertEqual(payload["model_instance_path"], MODEL_INSTANCE_PATH)
            self.assertEqual(payload["tp_rank"], 0)
            self.assertEqual(payload["tensor_parallel_size"], 8)
            self.assertEqual(set(payload["inputs"]), {"x"})
            self.assertEqual(payload["parameters"], {"limit": 16.0})
            self.assertNotIn("weight", repr(payload).lower())
            self.assertNotIn("gate_up", repr(payload).lower())
            self.assertNotIn('"gate"', repr(payload).lower())
            self.assertNotIn('"up"', repr(payload).lower())

            state["capture_closed"] = True
            (run_dir / "capture-state.json").write_text(
                json.dumps(state) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CaptureError, "capture is closed"):
                collector.record(
                    torch.ones((16, 8), dtype=torch.bfloat16),
                    torch.ones((16, 8), dtype=torch.bfloat16),
                    model_instance_path=MODEL_INSTANCE_PATH,
                    limit=16.0,
                    execution_phase="decode",
                )

            with self.assertRaisesRegex(
                CaptureError,
                "loaded checkpoint",
            ):
                CandidateCollector.from_config_path(
                    config_path,
                    tp_rank=0,
                    tp_size=8,
                    loaded_checkpoint={
                        "model_path": "stepfun-ai/Another-Model",
                        "revision": "fixture",
                    },
                )


if __name__ == "__main__":
    unittest.main()
