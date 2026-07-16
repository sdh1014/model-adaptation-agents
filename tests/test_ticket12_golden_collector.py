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


OPERATOR_ID = "activation.step_swiglu_with_limit"


def write_capture_config(path: Path, run_dir: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "golden-capture-config/v0",
                "spec_binding": {
                    "spec_id": "step3p7-flash-p800-demo",
                    "contract_revision": 2,
                    "contract_data_sha256": "fixture-sha256",
                },
                "operator_id": OPERATOR_ID,
                "model_path": "target",
                "max_samples": 3,
                "serialization": "candidate-torch-save/v1",
                "run_dir": str(run_dir),
                "capture_device_type": "cpu",
                "dtype": "bfloat16",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def fake_swiglu(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    return gate.clamp(max=limit) * up.clamp(min=-limit, max=limit)


class Ticket12CaptureCandidateTest(unittest.TestCase):
    def test_keeps_three_distinct_signatures_and_portable_minimal_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "golden-001"
            samples_dir = run_dir / "samples"
            samples_dir.mkdir(parents=True)
            config_path = run_dir / "capture-config.json"
            write_capture_config(config_path, run_dir)
            collector = CandidateCollector.from_config_path(config_path)

            for rows in (1, 2, 4):
                gate_up = torch.arange(rows * 8, dtype=torch.bfloat16).reshape(rows, 8)
                actual = collector.capture(
                    fake_swiglu,
                    gate_up,
                    16.0,
                    execution_phase="decode",
                )
                torch.testing.assert_close(actual, fake_swiglu(gate_up, 16.0))

            duplicate = torch.full((2, 8), 2.0, dtype=torch.bfloat16)
            collector.capture(
                fake_swiglu,
                duplicate,
                16.0,
                execution_phase="decode",
            )
            fourth = torch.ones((8, 8), dtype=torch.bfloat16)
            collector.capture(
                fake_swiglu,
                fourth,
                16.0,
                execution_phase="decode",
            )

            sample_paths = sorted(samples_dir.glob("*.pt"))
            self.assertEqual(len(sample_paths), 3)
            state = json.loads(
                (run_dir / "capture-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["saved_sample_count"], 3)
            self.assertEqual(state["repeated_call_count"], 1)
            self.assertEqual(state["skipped_call_count"], 1)
            self.assertEqual(len(state["skipped_signatures"]), 1)
            self.assertEqual(
                state["skipped_signatures"][0]["signature"]["inputs"]["gate_up"][
                    "shape"
                ],
                [8, 8],
            )
            self.assertEqual(state["skipped_signatures"][0]["call_count"], 1)

            payload = torch.load(sample_paths[0], map_location="cpu", weights_only=True)
            self.assertEqual(payload["schema"], "capture-candidate/v0")
            self.assertEqual(
                set(payload),
                {
                    "schema",
                    "spec_binding",
                    "operator_id",
                    "signature",
                    "inputs",
                    "parameters",
                    "expected",
                },
            )
            self.assertEqual(set(payload["inputs"]), {"gate_up"})
            self.assertEqual(payload["parameters"], {"limit": 16.0})
            self.assertEqual(payload["inputs"]["gate_up"].device.type, "cpu")
            self.assertEqual(payload["expected"].device.type, "cpu")
            self.assertEqual(payload["inputs"]["gate_up"].dtype, torch.bfloat16)
            self.assertEqual(payload["expected"].dtype, torch.bfloat16)
            self.assertNotIn("weight", repr(payload).lower())
            self.assertNotIn("runtime", repr(payload).lower())

    def test_rejects_parameter_and_non_finite_tensor_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "golden-001"
            (run_dir / "samples").mkdir(parents=True)
            config_path = run_dir / "capture-config.json"
            write_capture_config(config_path, run_dir)
            collector = CandidateCollector.from_config_path(config_path)

            parameter = torch.nn.Parameter(
                torch.ones((1, 8), dtype=torch.bfloat16),
                requires_grad=False,
            )
            with self.assertRaisesRegex(CaptureError, "nn.Parameter inputs are forbidden"):
                collector.capture(
                    fake_swiglu,
                    parameter,
                    16.0,
                    execution_phase="decode",
                )

            non_finite = torch.ones((1, 8), dtype=torch.bfloat16)
            non_finite[0, 0] = float("nan")
            with self.assertRaisesRegex(CaptureError, "non-finite"):
                collector.capture(
                    fake_swiglu,
                    non_finite,
                    16.0,
                    execution_phase="decode",
                )

            state = json.loads(
                (run_dir / "capture-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["saved_sample_count"], 0)


if __name__ == "__main__":
    unittest.main()
