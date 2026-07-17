import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "scan-003"
RESULT_PATH = RUN_DIR / "result.json"
CONFIG_PATH = RUN_DIR / "checkpoint-config-summary.json"
EXPECTED_CONTRACT_SHA256 = (
    "c7c59b8dfa5900fa889f7229004774af80939376e4c520a56fd3b7e154a07e1b"
)
HISTORICAL_RUN_SHA256 = {
    "runs/scan-001/result.json": (
        "d77a87ad9da57277fde94cd123f4afa935ae2c31ff8f00daa09a2a9a11e4dc6d"
    ),
    "runs/scan-002/result.json": (
        "cbe87935248e0f80eed199c5843d1236908bb2b46dcc0014f33bb14fb684926d"
    ),
    "runs/scan-003/result.json": (
        "6cedbde0105bcadc5de1b26c96d3c46aeef6cd72f82e37b7a34a02d9ea7572b5"
    ),
}


class Ticket18TargetOnlyEagerScanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_scan_is_bound_to_revision_3_target_only_eager_contract(self):
        self.assertEqual(
            self.result["spec_binding"],
            {
                "spec_id": "step3p7-flash-p800-demo",
                "contract_revision": 3,
                "contract_data_sha256": EXPECTED_CONTRACT_SHA256,
            },
        )
        runtime = self.result["runtime_profile"]
        self.assertIsNone(runtime["speculative_algorithm"])
        self.assertEqual(runtime["cuda_graph_backend_decode"], "disabled")
        self.assertEqual(runtime["cuda_graph_backend_prefill"], "disabled")
        self.assertNotIn("eagle_draft_resolution", self.result)
        self.assertEqual(set(self.result["entrypoints"]), {"target"})
        self.assertEqual(
            self.result["entrypoints"]["target"]["entry"],
            "Step3p7ForConditionalGeneration.forward",
        )

    def test_every_operator_has_only_a_complete_target_call_chain(self):
        for operator in self.result["operators"]:
            with self.subTest(operator=operator["operator_id"]):
                self.assertEqual(operator["model_path"], ["target"])
                self.assertEqual(set(operator["call_chains"]), {"target"})
                self.assertTrue(operator["call_chains"]["target"])
                for chain in operator["call_chains"]["target"]:
                    self.assertGreaterEqual(len(chain), 2)
                    self.assertEqual(
                        chain[0],
                        "Step3p7ForConditionalGeneration.forward",
                    )
                self.assertTrue(operator["cuda_impl"])
                self.assertTrue(operator["kunlun_impl"])

    def test_inventory_counts_gap_queue_and_capture_plan_are_complete(self):
        operators = self.result["operators"]
        counts = self.result["operator_counts"]
        self.assertEqual(len(operators), counts["total"])
        self.assertEqual(
            sum(item["verdict"] == "READY" for item in operators),
            counts["ready"],
        )
        self.assertEqual(
            sum(item["verdict"] == "CAPTURE_REQUIRED" for item in operators),
            counts["capture_required"],
        )
        self.assertEqual(
            sum(item["verdict"] == "NEEDS_HUMAN" for item in operators),
            counts["needs_human"],
        )
        self.assertEqual(
            {item["operator_id"] for item in self.result["gap_queue"]},
            {
                "activation.step_swiglu_with_limit",
                "norm.gemma_rms",
                "moe.topk_sigmoid_bias",
                "moe.bf16_clamped",
                "attention.radix",
            },
        )
        self.assertEqual(
            {item["operator_id"] for item in self.result["capture_plan"]},
            {"activation.step_swiglu_with_limit"},
        )

    def test_loaded_config_summary_contains_target_activation_only(self):
        self.assertEqual(
            self.config["raw_config_sha256"],
            self.result["source"]["checkpoint_config_sha256"],
        )
        active = self.config["activation_summary"]
        self.assertEqual(active["target_layer_ids"], list(range(45)))
        self.assertEqual(active["target_dense_layers"], [0, 1, 2])
        self.assertEqual(active["target_moe_layers"], list(range(3, 45)))
        self.assertNotIn("draft_layer_id", active)
        self.assertEqual(
            active["shared_swiglu_nonzero_limits"],
            {"43": 16, "44": 16},
        )

    def test_scan_supersedes_scan_002_without_rewriting_history(self):
        self.assertEqual(self.result["supersedes"], "runs/scan-002")
        for relative_path, expected_digest in HISTORICAL_RUN_SHA256.items():
            with self.subTest(path=relative_path):
                actual = hashlib.sha256(
                    (ROOT / relative_path).read_bytes()
                ).hexdigest()
                self.assertEqual(actual, expected_digest)

    def test_spec_supersedes_scan_003_with_revision_4_cuda_preflight(self):
        spec = (ROOT / "migration-spec.md").read_text(encoding="utf-8")
        self.assertIn("- `scan_run`: `runs/scan-004`", spec)
        self.assertIn("- `last_run`: `runs/scan-004`", spec)
        self.assertIn("- `draft_coverage`: `NOT_APPLICABLE`", spec)
        self.assertIn("- `phase`: `CUDA_CAPTURE`", spec)
        self.assertIn(
            "`next_action`: `在 CUDA 机器运行 revision 4 capture_golden.py "
            "preflight，验证 scan-004、原始 Step3p5MLP.forward Hook 和 rank 0 样本格式`",
            spec,
        )

    def test_all_declared_evidence_files_exist(self):
        self.assertTrue(self.result["scan_complete"])
        for evidence in self.result["evidence"]:
            self.assertTrue((RUN_DIR / evidence).is_file(), evidence)


if __name__ == "__main__":
    unittest.main()
