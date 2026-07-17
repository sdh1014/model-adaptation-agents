import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
REPLAY_COMPARE = REPO_ROOT / "model-adaptation" / "scripts" / "replay_compare.py"
SCAN_004 = REPO_ROOT / "runs" / "scan-004" / "result.json"
HISTORICAL_SHA256 = {
    "runs/spec-binding-002/result.json": (
        "cdc6d620f226e1d8fe30227bdecf536a4ae8df02f77ac4a33ed2e4507c1fdc5a"
    ),
    "runs/scan-003/result.json": (
        "6cedbde0105bcadc5de1b26c96d3c46aeef6cd72f82e37b7a34a02d9ea7572b5"
    ),
}


def revision_4_contract() -> dict:
    return {
        "schema": "migration-spec/v0",
        "spec_id": "step3p7-flash-p800-demo",
        "contract_revision": 4,
        "model": "Step-3.7-Flash",
        "source": {
            "sglang_revision": "49e384ce9d304648e9959666ecb8ce8cd98d0deb",
            "sglang_kunlun_revision": "546ad8c682392922792bbbfe53a8bf575545f118",
        },
        "checkpoint": {
            "id": "stepfun-ai/Step-3.7-Flash@fixture",
            "config_digest": "config-digest",
        },
        "model_path": {
            "target_entry": "Step3p7ForConditionalGeneration.forward",
            "draft_entry": None,
        },
        "operator_boundary": {
            "id": "Step3p5MLP.forward",
            "activation_guard": "self.limit is not None",
            "tp_rank": 0,
        },
        "runtime": {
            "tensor_parallel_size": 8,
            "dtype": "bfloat16",
            "quantization": None,
            "speculative_algorithm": None,
            "cuda_graph_backend_decode": "disabled",
            "cuda_graph_backend_prefill": "disabled",
            "attention_backend": None,
        },
        "demo_input_mode": "text-only",
        "limits": {
            "max_shapes_per_operator": 3,
            "max_repair_attempts": 5,
        },
        "precision_gate": {
            "comparator": "torch.testing.assert_close",
            "atol": 0.01,
            "rtol": 0.02,
            "require_exact_structure": True,
            "require_same_dtype": True,
            "require_finite": True,
            "check_stride": False,
        },
    }


def write_spec(path: Path, contract: dict) -> None:
    path.write_text(
        "\n".join(
            (
                "# Migration Spec",
                "<!-- CONTRACT-DATA: BEGIN -->",
                json.dumps(contract, ensure_ascii=False, indent=2),
                "<!-- CONTRACT-DATA: END -->",
                "",
            )
        ),
        encoding="utf-8",
    )


class Ticket19MlpBoundaryTest(unittest.TestCase):
    def test_scan_004_uses_only_the_original_mlp_gap_boundary(self) -> None:
        result = json.loads(SCAN_004.read_text(encoding="utf-8"))
        self.assertEqual(result["spec_binding"]["contract_revision"], 4)
        self.assertEqual(
            result["source"]["sglang_revision"],
            "49e384ce9d304648e9959666ecb8ce8cd98d0deb",
        )
        self.assertEqual(
            result["source"]["sglang_kunlun_revision"],
            "546ad8c682392922792bbbfe53a8bf575545f118",
        )
        self.assertEqual(result["supersedes"], "runs/scan-003")
        self.assertEqual(len(result["operators"]), 13)

        capture_required = [
            item
            for item in result["operators"]
            if item["verdict"] == "CAPTURE_REQUIRED"
        ]
        self.assertEqual(len(capture_required), 1)
        operator = capture_required[0]
        self.assertEqual(operator["operator_id"], "Step3p5MLP.forward")
        self.assertEqual(operator["activation_guard"], "self.limit is not None")
        self.assertEqual(operator["boundary"], "x to output")
        self.assertEqual(
            operator["state_dependency"],
            "same checkpoint, TP8, current-rank model weights",
        )
        self.assertEqual(
            operator["hook_target"],
            "sglang.srt.models.step3p5.Step3p5MLP.forward",
        )
        for chain in operator["call_chains"]["target"]:
            self.assertEqual(chain[-1], "share_expert(Step3p5MLP.forward)")

        self.assertEqual(
            result["capture_plan"],
            [
                {
                    "operator_id": "Step3p5MLP.forward",
                    "max_distinct_shapes": 3,
                    "tp_rank": 0,
                    "replay_mode": "loaded_model",
                    "weights_in_golden_sample": False,
                    "reason": (
                        "Replay rank 0 inside the loaded TP8 model using the "
                        "same checkpoint weights."
                    ),
                }
            ],
        )
        serialized = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("step_swiglu_with_limit", serialized)
        self.assertNotIn("step3p5_ops.py", serialized)

        for relative_path, expected in HISTORICAL_SHA256.items():
            with self.subTest(path=relative_path):
                self.assertEqual(
                    hashlib.sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest(),
                    expected,
                )

    def test_revision_4_contract_accepts_original_mlp_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            spec = workspace / "migration-spec.md"
            run_dir = workspace / "runs" / "binding"
            write_spec(spec, revision_4_contract())

            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPLAY_COMPARE),
                    "--spec",
                    str(spec),
                    "--run-dir",
                    str(run_dir),
                    "--mode",
                    "synthetic",
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(
                (run_dir / "result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["spec_binding"]["contract_revision"], 4)

    def test_revision_4_contract_rejects_a_nonzero_validation_rank(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            spec = workspace / "migration-spec.md"
            run_dir = workspace / "runs" / "binding"
            contract = revision_4_contract()
            contract["operator_boundary"]["tp_rank"] = 1
            write_spec(spec, contract)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPLAY_COMPARE),
                    "--spec",
                    str(spec),
                    "--run-dir",
                    str(run_dir),
                    "--mode",
                    "synthetic",
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("operator_boundary.tp_rank", completed.stderr)


if __name__ == "__main__":
    unittest.main()
