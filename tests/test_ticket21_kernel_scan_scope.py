import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "migration-spec.md"
SKILL = ROOT / "model-adaptation" / "SKILL.md"
SPEC_TEMPLATE = (
    ROOT / "model-adaptation" / "references" / "migration-spec-template.md"
)
SCAN_005 = ROOT / "runs" / "scan-005" / "result.json"
SPEC_BINDING_004 = ROOT / "runs" / "spec-binding-004" / "result.json"
REPLAY_COMPARE = ROOT / "model-adaptation" / "scripts" / "replay_compare.py"
CAPTURE_GOLDEN = ROOT / "model-adaptation" / "scripts" / "capture_golden.py"
CONTRACT_BEGIN = "<!-- CONTRACT-DATA: BEGIN -->"
CONTRACT_END = "<!-- CONTRACT-DATA: END -->"
HISTORICAL_SHA256 = {
    "runs/scan-004/result.json": (
        "6b0d7531937a06a116a323d3211a9a3cfb29b7233f2f067e421ca3cd22cb01c5"
    ),
    "runs/spec-binding-003/result.json": (
        "7498fa0888d8584cd3e5cbd3f46bc425da49d7d7918d57d00bab66829f571122"
    ),
}


def contract_from_text(text: str) -> dict:
    start = text.index(CONTRACT_BEGIN) + len(CONTRACT_BEGIN)
    end = text.index(CONTRACT_END)
    return json.loads(text[start:end].strip())


def write_contract(path: Path, contract: dict) -> None:
    path.write_text(
        "\n".join(
            [
                "# Migration Spec",
                CONTRACT_BEGIN,
                json.dumps(contract, ensure_ascii=False, indent=2),
                CONTRACT_END,
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_synthetic(spec: Path, run_dir: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPYCACHEPREFIX"] = "/tmp/model-adaptation-agents-pyc"
    return subprocess.run(
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
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


class Ticket21KernelScanScopeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec_text = SPEC.read_text(encoding="utf-8")
        cls.contract = contract_from_text(cls.spec_text)
        cls.scan = json.loads(SCAN_005.read_text(encoding="utf-8"))

    def test_revision_5_contract_fixes_scope_without_preselecting_operator(self) -> None:
        self.assertEqual(self.contract["contract_revision"], 5)
        self.assertNotIn("operator_boundary", self.contract)
        self.assertNotIn("demo_input_mode", self.contract)
        self.assertNotIn("active_operator", self.contract)
        self.assertEqual(
            self.contract["scan_scope"],
            {
                "model_paths": ["target"],
                "granularity": "kernel-call",
                "input_modes": ["text-only", "single-image"],
            },
        )

    def test_contract_allows_only_direct_call_parameters_not_model_weights(self) -> None:
        self.assertEqual(
            self.contract["sample_policy"],
            {
                "capture_tp_rank": 0,
                "save_inputs": True,
                "save_expected_outputs": True,
                "save_direct_parameter_tensors": True,
                "parameter_scope": "selected-kernel-call-current-rank",
                "save_full_checkpoint": False,
                "save_module_state_dict": False,
            },
        )

    def test_new_specs_start_with_the_same_kernel_scan_contract_shape(self) -> None:
        template_contract = contract_from_text(
            SPEC_TEMPLATE.read_text(encoding="utf-8")
        )
        self.assertEqual(template_contract["scan_scope"], self.contract["scan_scope"])
        self.assertEqual(
            template_contract["sample_policy"],
            self.contract["sample_policy"],
        )
        self.assertNotIn("operator_boundary", template_contract)
        self.assertNotIn("active_operator", template_contract)

    def test_revision_5_binding_is_executable_but_rejects_contract_preselection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            accepted_spec = workspace / "accepted.md"
            write_contract(accepted_spec, self.contract)
            accepted = run_synthetic(accepted_spec, workspace / "accepted-run")
            self.assertEqual(accepted.returncode, 0, accepted.stderr)

            for forbidden_field in ("active_operator", "operator_boundary"):
                with self.subTest(field=forbidden_field):
                    rejected_contract = json.loads(json.dumps(self.contract))
                    rejected_contract[forbidden_field] = (
                        "sgl_kernel.gemma_rmsnorm"
                        if forbidden_field == "active_operator"
                        else {"id": "Step3p5MLP.forward"}
                    )
                    rejected_spec = workspace / f"rejected-{forbidden_field}.md"
                    write_contract(rejected_spec, rejected_contract)
                    rejected = run_synthetic(
                        rejected_spec,
                        workspace / f"rejected-{forbidden_field}-run",
                    )
                    self.assertEqual(rejected.returncode, 2)
                    self.assertIn(forbidden_field, rejected.stderr)

    def test_scan_005_is_kernel_level_and_bound_to_revision_5(self) -> None:
        expected_digest = hashlib.sha256(
            json.dumps(
                self.contract,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(self.scan["spec_binding"]["contract_revision"], 5)
        self.assertEqual(
            self.scan["spec_binding"]["contract_data_sha256"],
            expected_digest,
        )
        binding_result = json.loads(SPEC_BINDING_004.read_text(encoding="utf-8"))
        self.assertEqual(binding_result["spec_binding"], self.scan["spec_binding"])
        self.assertTrue(binding_result["passed"])
        self.assertEqual(self.scan["supersedes"], "runs/scan-004")
        self.assertTrue(self.scan["scan_complete"])
        self.assertEqual(self.scan["scan_scope"], self.contract["scan_scope"])
        for operator in self.scan["operators"]:
            with self.subTest(operator=operator["operator_id"]):
                self.assertTrue(operator["call_chain"])
                self.assertEqual(
                    operator["call_chain"][0],
                    "Step3p7ForConditionalGeneration.forward",
                )
                self.assertIn(operator["input_mode"], self.contract["scan_scope"]["input_modes"])
                self.assertTrue(operator["kernel_call"]["symbol"])
                self.assertTrue(operator["kernel_call"]["kind"])
                self.assertTrue(operator["cuda_impl"])
                self.assertTrue(operator["kunlun_impl"])
                self.assertTrue(operator["boundary"]["inputs"])
                self.assertTrue(operator["boundary"]["outputs"])

    def test_candidate_comparison_covers_requested_and_bypassed_calls(self) -> None:
        operators = {item["operator_id"]: item for item in self.scan["operators"]}
        self.assertIn("sgl_kernel.topk_sigmoid", operators)
        self.assertIn(
            "sglang.srt.layers.attention.triton_ops.prefill_attention._fwd_kernel",
            operators,
        )
        self.assertIn("sgl_kernel.gemma_rmsnorm", operators)
        self.assertEqual(
            operators["sgl_kernel.topk_sigmoid"]["kernel_call"]["kind"],
            "cuda-extension",
        )
        self.assertEqual(
            operators[
                "sglang.srt.layers.attention.triton_ops.prefill_attention._fwd_kernel"
            ]["kernel_call"]["kind"],
            "triton",
        )
        self.assertEqual(
            operators[
                "sglang.srt.layers.moe.moe_runner.triton_utils."
                "fused_moe_triton_kernels.fused_moe_kernel"
            ]["verdict"],
            "READY",
        )
        self.assertEqual(
            operators[
                "sglang.srt.layers.attention.triton_ops.extend_attention._fwd_kernel"
            ]["verdict"],
            "READY",
        )

    def test_inventory_counts_and_gap_queue_match_operator_verdicts(self) -> None:
        operators = self.scan["operators"]
        counts = self.scan["operator_counts"]
        self.assertEqual(counts["total"], len(operators))
        self.assertEqual(
            counts["ready"],
            sum(item["verdict"] == "READY" for item in operators),
        )
        self.assertEqual(
            counts["capture_required"],
            sum(item["verdict"] == "CAPTURE_REQUIRED" for item in operators),
        )
        self.assertEqual(
            counts["needs_human"],
            sum(item["verdict"] == "NEEDS_HUMAN" for item in operators),
        )
        self.assertEqual(
            {item["operator_id"] for item in self.scan["gap_queue"]},
            {
                item["operator_id"]
                for item in operators
                if item["verdict"] == "CAPTURE_REQUIRED"
            },
        )

    def test_active_operator_is_selected_after_scan_from_gap_queue(self) -> None:
        selection = self.scan["selection"]
        self.assertTrue(selection["evaluated_after_scan"])
        active_operator = selection["active_operator"]
        gap_ids = {item["operator_id"] for item in self.scan["gap_queue"]}
        compared_ids = {item["operator_id"] for item in selection["candidates"]}
        self.assertEqual(active_operator, "sgl_kernel.gemma_rmsnorm")
        self.assertIn(active_operator, gap_ids)
        self.assertTrue(
            {
                "sgl_kernel.gemma_rmsnorm",
                "sgl_kernel.topk_sigmoid",
                "sglang.srt.layers.attention.triton_ops.prefill_attention._fwd_kernel",
            }.issubset(compared_ids)
        )
        self.assertEqual(
            [item["operator_id"] for item in self.scan["capture_plan"]],
            [active_operator],
        )
        self.assertIn(
            "- `active_operator`: `sgl_kernel.gemma_rmsnorm`",
            self.spec_text,
        )
        self.assertIn("- `scan_run`: `runs/scan-005`", self.spec_text)

    def test_selected_capture_plan_obeys_parameter_policy(self) -> None:
        plan = self.scan["capture_plan"][0]
        self.assertEqual(plan["max_distinct_shapes"], 3)
        self.assertEqual(plan["tp_rank"], 0)
        self.assertEqual(plan["saved_parameters"], ["weight"])
        self.assertTrue(plan["save_direct_parameter_tensors"])
        self.assertFalse(plan["save_full_checkpoint"])
        self.assertFalse(plan["save_module_state_dict"])
        self.assertEqual(plan["replay_mode"], "standalone-kernel-call")

    def test_skill_describes_scan_then_select_and_defers_missing_adapter(self) -> None:
        skill = SKILL.read_text(encoding="utf-8")
        self.assertIn("kernel-call", skill)
        self.assertIn("text-only", skill)
        self.assertIn("single-image", skill)
        self.assertIn("扫描完成后", skill)
        self.assertIn("active_operator", skill)
        self.assertIn("当前 kernel 调用直接使用", skill)
        self.assertIn("capture/replay adapter", skill)

    def test_revision_5_cannot_fall_back_to_the_historical_mlp_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "capture"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(CAPTURE_GOLDEN),
                    "--spec",
                    str(SPEC),
                    "--run-dir",
                    str(run_dir),
                    "--mode",
                    "prepare",
                    "--scan-result",
                    str(SCAN_005),
                    "--operator-id",
                    "sgl_kernel.gemma_rmsnorm",
                    "--sglang-worktree",
                    str(ROOT),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("capture/replay adapter is not implemented", completed.stderr)
            self.assertIn("sgl_kernel.gemma_rmsnorm", completed.stderr)
            self.assertFalse(run_dir.exists())

    def test_revision_5_model_replay_reports_the_missing_kernel_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPLAY_COMPARE),
                    "--spec",
                    str(SPEC),
                    "--run-dir",
                    str(workspace / "replay"),
                    "--mode",
                    "prepare-model-replay",
                    "--golden-run",
                    str(workspace / "golden"),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("kernel capture/replay adapter is not implemented", completed.stderr)
            self.assertFalse((workspace / "replay").exists())

    def test_historical_revision_4_runs_are_unchanged(self) -> None:
        for relative_path, expected in HISTORICAL_SHA256.items():
            with self.subTest(path=relative_path):
                actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
