import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "scan-002"
RESULT_PATH = RUN_DIR / "result.json"
CONFIG_PATH = RUN_DIR / "checkpoint-config-summary.json"


class Ticket11ScanArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads(RESULT_PATH.read_text())
        cls.config = json.loads(CONFIG_PATH.read_text())

    def test_inventory_counts_and_evidence_files_are_complete(self):
        operators = self.result["operators"]
        counts = self.result["operator_counts"]

        self.assertEqual(len(operators), counts["total"])
        self.assertEqual(
            sum(op["verdict"] == "READY" for op in operators), counts["ready"]
        )
        self.assertEqual(
            sum(op["verdict"] == "CAPTURE_REQUIRED" for op in operators),
            counts["capture_required"],
        )
        self.assertEqual(
            sum(op["verdict"] == "NEEDS_HUMAN" for op in operators),
            counts["needs_human"],
        )
        self.assertTrue(self.result["scan_complete"])
        for evidence in self.result["evidence"]:
            self.assertTrue((RUN_DIR / evidence).is_file(), evidence)

    def test_every_operator_has_entry_to_boundary_chains(self):
        expected_entry = {
            "target": "Step3p7ForConditionalGeneration.forward",
            "draft": "Step3p5MTP.forward",
        }

        for operator in self.result["operators"]:
            with self.subTest(operator=operator["operator_id"]):
                self.assertEqual(
                    set(operator["model_path"]), set(operator["call_chains"])
                )
                for path in operator["model_path"]:
                    chains = operator["call_chains"][path]
                    self.assertTrue(chains)
                    for chain in chains:
                        self.assertGreaterEqual(len(chain), 2)
                        self.assertEqual(chain[0], expected_entry[path])

    def test_target_embedding_happens_before_language_model_forward(self):
        operators = {op["operator_id"]: op for op in self.result["operators"]}
        expected_prefix = [
            "Step3p7ForConditionalGeneration.forward",
            "general_mm_embed_routine(text-only)",
            "Step3p5ForCausalLM.get_input_embeddings",
            "Step3p5Model.get_input_embeddings",
            "VocabParallelEmbedding.forward",
        ]

        embedding_chain = operators["embedding.vocab_parallel"]["call_chains"][
            "target"
        ][0]
        collective_embedding_chain = operators["collective.tp8"]["call_chains"][
            "target"
        ][0]
        self.assertEqual(embedding_chain[:5], expected_prefix)
        self.assertEqual(collective_embedding_chain[:5], expected_prefix)
        self.assertNotIn("Step3p5ForCausalLM.forward", embedding_chain)

    def test_every_operator_has_cuda_and_kunlun_code_evidence(self):
        anchor = re.compile(r"(?:sglang|sglang-kunlun):[^\s]+:\d")

        for operator in self.result["operators"]:
            with self.subTest(operator=operator["operator_id"]):
                self.assertTrue(operator["cuda_impl"])
                self.assertTrue(operator["kunlun_impl"])
                self.assertTrue(
                    any(anchor.search(item) for item in operator["cuda_impl"])
                )
                self.assertTrue(
                    any(anchor.search(item) for item in operator["kunlun_impl"])
                )

    def test_gap_queue_contains_every_follow_up_item(self):
        operators = {op["operator_id"]: op for op in self.result["operators"]}
        gap_ids = {item["operator_id"] for item in self.result["gap_queue"]}
        capture_ids = {item["operator_id"] for item in self.result["capture_plan"]}

        self.assertEqual(
            gap_ids,
            {
                "activation.step_swiglu_with_limit",
                "norm.gemma_rms",
                "moe.topk_sigmoid_bias",
                "moe.bf16_clamped",
                "attention.radix",
            },
        )
        self.assertEqual(capture_ids, {"activation.step_swiglu_with_limit"})
        self.assertEqual(
            operators["activation.step_swiglu_with_limit"]["verdict"],
            "CAPTURE_REQUIRED",
        )
        for operator_id in gap_ids - capture_ids:
            self.assertEqual(operators[operator_id]["verdict"], "NEEDS_HUMAN")

    def test_linear_and_collective_chains_use_real_callable_nodes(self):
        operators = {op["operator_id"]: op for op in self.result["operators"]}
        linear = operators["linear.bf16"]["call_chains"]
        all_linear_nodes = [
            node
            for path_chains in linear.values()
            for chain in path_chains
            for node in chain
        ]

        self.assertNotIn("qkv_proj(ParallelLinear.forward)", all_linear_nodes)
        self.assertNotIn("gate_up_proj(ParallelLinear.forward)", all_linear_nodes)
        self.assertIn("qkv_proj(QKVParallelLinear.__call__)", all_linear_nodes)
        self.assertIn(
            "gate_up_proj(MergedColumnParallelLinear.__call__)", all_linear_nodes
        )
        self.assertIn("ColumnParallelLinear.forward", all_linear_nodes)

        collective = operators["collective.tp8"]["call_chains"]
        self.assertEqual(len(collective["target"]), 1)
        self.assertEqual(len(collective["draft"]), 1)

    def test_source_resolves_default_moe_branch(self):
        runtime = self.result["runtime_profile"]

        self.assertIsNone(runtime["moe_a2a_backend_argument"])
        self.assertEqual(runtime["source_resolved_moe_a2a_backend"], "none")
        self.assertIsNone(runtime["moe_runner_backend_argument"])
        self.assertEqual(runtime["server_args_moe_runner_backend"], "auto")
        self.assertEqual(
            runtime["cuda_effective_unquantized_moe_runner_backend"], "triton"
        )

    def test_loaded_config_evidence_drives_activation_summary(self):
        self.assertEqual(
            self.config["raw_config_sha256"],
            self.result["source"]["checkpoint_config_sha256"],
        )
        loaded = self.config["loaded_config"]
        active = self.config["activation_summary"]

        self.assertEqual(loaded["torch_dtype"], "bfloat16")
        self.assertIsNone(loaded["quantization_config"])
        self.assertEqual(loaded["num_hidden_layers"], 45)
        self.assertEqual(active["target_dense_layers"], [0, 1, 2])
        self.assertEqual(active["target_moe_layers"], list(range(3, 45)))
        self.assertEqual(active["draft_layer_id"], 45)
        self.assertEqual(active["draft_layer_type"], "sliding_attention")
        self.assertFalse(active["draft_is_moe"])
        self.assertEqual(active["draft_model_runner_count"], 3)
        self.assertEqual(active["draft_model_indices"], [0, 1, 2])
        self.assertEqual(active["draft_checkpoint_weight_layer_ids"], [45, 46, 47])
        self.assertEqual(active["routed_swiglu_nonzero_limits"], {"43": 7, "44": 7})
        self.assertEqual(
            active["shared_swiglu_nonzero_limits"], {"43": 16, "44": 16}
        )

    def test_historical_scan_is_superseded_without_rewrite(self):
        spec = (ROOT / "migration-spec.md").read_text()
        target_only_scan = json.loads(
            (ROOT / "runs" / "scan-003" / "result.json").read_text()
        )
        mlp_scan = json.loads(
            (ROOT / "runs" / "scan-004" / "result.json").read_text()
        )
        current_scan = json.loads(
            (ROOT / "runs" / "scan-005" / "result.json").read_text()
        )
        self.assertIn("- `scan_run`: `runs/scan-005`", spec)
        self.assertEqual(self.result["supersedes"], "runs/scan-001")
        self.assertEqual(target_only_scan["supersedes"], "runs/scan-002")
        self.assertEqual(mlp_scan["supersedes"], "runs/scan-003")
        self.assertEqual(current_scan["supersedes"], "runs/scan-004")

    def test_eagle_defaults_cover_all_three_draft_weight_layers(self):
        eagle = self.result["eagle_draft_resolution"]

        self.assertFalse(eagle["explicit_mtp_flag"])
        self.assertFalse(eagle["explicit_speculative_tuning_arguments"])
        self.assertEqual(
            eagle["source_resolved_defaults"],
            {
                "speculative_num_steps": 3,
                "speculative_eagle_topk": 1,
                "speculative_num_draft_tokens": 4,
            },
        )
        self.assertEqual(eagle["draft_model_indices"], [0, 1, 2])
        self.assertEqual(eagle["checkpoint_weight_layer_ids"], [45, 46, 47])
        self.assertEqual(eagle["operator_branch_config_layer_id"], 45)


if __name__ == "__main__":
    unittest.main()
