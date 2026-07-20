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
SCAN_006 = ROOT / "runs" / "scan-006" / "result.json"
SPEC_BINDING_004 = ROOT / "runs" / "spec-binding-004" / "result.json"
REPLAY_COMPARE = ROOT / "model-adaptation" / "scripts" / "replay_compare.py"
CAPTURE_GOLDEN = ROOT / "model-adaptation" / "scripts" / "capture_golden.py"
SWIGLU_CLAMP = (
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe."
    "_swiglu_silu_clamp_mul"
)
CONTRACT_BEGIN = "<!-- CONTRACT-DATA: BEGIN -->"
CONTRACT_END = "<!-- CONTRACT-DATA: END -->"
HISTORICAL_SHA256 = {
    "runs/scan-005/result.json": (
        "3d9fc10e3c1382a13b09f7902e7c27819b22eef4148d77f5e2fce9d1c995852c"
    ),
    "runs/scan-005/scan.log": (
        "f7b9c872144415eac0aa96e1a3bbd1ea165a8579506be16496d9af3e3b861b39"
    ),
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
        cls.current_contract = contract_from_text(cls.spec_text)
        cls.contract = json.loads(json.dumps(cls.current_contract))
        cls.contract["contract_revision"] = 5
        cls.scan = json.loads(SCAN_006.read_text(encoding="utf-8"))

    def test_revision_5_contract_fixes_scope_without_preselecting_operator(self) -> None:
        self.assertEqual(self.contract["contract_revision"], 5)
        self.assertEqual(self.current_contract["contract_revision"], 6)
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

    def test_contract_shape_does_not_depend_on_revision_number(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            legacy = json.loads(json.dumps(self.contract))
            legacy["contract_revision"] = 5
            legacy.pop("scan_scope")
            legacy.pop("sample_policy")
            legacy["operator_boundary"] = {
                "id": "Step3p5MLP.forward",
                "activation_guard": "self.limit is not None",
                "tp_rank": 0,
            }
            legacy["demo_input_mode"] = "text-only"
            legacy_spec = workspace / "legacy-revision-5.md"
            write_contract(legacy_spec, legacy)
            completed = run_synthetic(legacy_spec, workspace / "legacy-run")
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_scan_006_is_kernel_level_and_bound_to_revision_5(self) -> None:
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
        self.assertEqual(self.scan["supersedes"], "runs/scan-005")
        self.assertIn(
            {
                "path": "runs/scan-005/result.json",
                "sha256": HISTORICAL_SHA256["runs/scan-005/result.json"],
            },
            self.scan["historical_inputs"],
        )
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
        self.assertIn(SWIGLU_CLAMP, operators)
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
        fused_moe = operators[
            "sglang.srt.layers.moe.moe_runner.triton_utils."
            "fused_moe_triton_kernels.fused_moe_kernel"
        ]
        self.assertEqual(
            fused_moe["boundary"]["outputs"],
            ["mutated_C_output_buffer"],
        )
        self.assertTrue(
            {
                "A",
                "C_output_buffer",
                "topk_weights",
                "topk_ids",
                "sorted_token_ids",
                "expert_ids",
                "num_tokens_post_padded",
            }.issubset(fused_moe["boundary"]["inputs"])
        )
        self.assertTrue(
            {
                "B_expert_weight",
                "bias",
                "B_scale",
                "B_zp",
            }.issubset(fused_moe["boundary"]["parameters"])
        )
        self.assertTrue(
            {
                "mul_routed_weight",
                "top_k",
                "config",
                "compute_type",
                "filter_expert",
            }.issubset(fused_moe["boundary"]["non_tensor_args"])
        )
        self.assertEqual(
            operators[
                "sglang.srt.layers.attention.triton_ops.extend_attention._fwd_kernel"
            ]["verdict"],
            "READY",
        )

    def test_single_image_inventory_uses_the_real_step3p7_vision_path(self) -> None:
        operators = {item["operator_id"]: item for item in self.scan["operators"]}
        self.assertNotIn("sgl_kernel.rmsnorm", operators)
        vision = operators[
            "sglang.srt.layers.attention.triton_ops.prefill_attention._fwd_kernel"
        ]
        self.assertNotIn("Step3VisionEncoder.forward", vision["call_chain"])
        self.assertIn(
            "Step3p7ForConditionalGeneration.get_image_feature",
            vision["call_chain"],
        )
        self.assertIn("PerceptionEncoder.forward", vision["call_chain"])
        self.assertIn(
            "PerceptionEncoderVisionBlock.forward",
            vision["call_chain"],
        )

    def test_inventory_counts_and_gap_queue_match_operator_verdicts(self) -> None:
        operators = self.scan["operators"]
        counts = self.scan["operator_counts"]
        self.assertEqual(
            counts,
            {
                "ready": 6,
                "capture_required": 5,
                "needs_human": 0,
                "total": 11,
            },
        )
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
        self.assertEqual(active_operator, SWIGLU_CLAMP)
        self.assertIn(active_operator, gap_ids)
        self.assertTrue(
            {
                SWIGLU_CLAMP,
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
            "- `active_operator`: `null`",
            self.spec_text,
        )
        self.assertIn("- `scan_run`: `null`", self.spec_text)
        self.assertIn("`scan-006` 和 revision 5", self.spec_text)

    def test_selected_capture_plan_obeys_parameter_policy(self) -> None:
        plan = self.scan["capture_plan"][0]
        operator = next(
            item
            for item in self.scan["operators"]
            if item["operator_id"] == SWIGLU_CLAMP
        )
        self.assertEqual(operator["kernel_call"]["kind"], "torch-compile")
        self.assertEqual(
            operator["boundary"],
            {
                "inputs": ["x"],
                "parameters": [],
                "non_tensor_args": ["gemm1_limit"],
                "outputs": ["output"],
            },
        )
        self.assertEqual(plan["max_distinct_shapes"], 3)
        self.assertEqual(plan["tp_rank"], 0)
        self.assertEqual(plan["saved_parameters"], [])
        self.assertTrue(plan["save_direct_parameter_tensors"])
        self.assertFalse(plan["save_full_checkpoint"])
        self.assertFalse(plan["save_module_state_dict"])
        self.assertEqual(plan["replay_mode"], "standalone-kernel-call")

    def test_capture_validation_rejects_parameters_outside_the_call_boundary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            malicious_scan = json.loads(json.dumps(self.scan))
            malicious_scan["capture_plan"][0]["saved_parameters"] = [
                "entire_checkpoint_tensor"
            ]
            scan_path = workspace / "scan.json"
            scan_path.write_text(
                json.dumps(malicious_scan, ensure_ascii=False),
                encoding="utf-8",
            )
            historical_spec = workspace / "migration-spec.md"
            write_contract(historical_spec, self.contract)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(CAPTURE_GOLDEN),
                    "--spec",
                    str(historical_spec),
                    "--run-dir",
                    str(workspace / "capture"),
                    "--mode",
                    "prepare",
                    "--scan-result",
                    str(scan_path),
                    "--operator-id",
                    SWIGLU_CLAMP,
                    "--sglang-worktree",
                    str(ROOT),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn(
                "saved parameters do not match the direct call parameters",
                completed.stderr,
            )

    def test_skill_describes_scan_then_select_and_defers_missing_adapter(self) -> None:
        skill = SKILL.read_text(encoding="utf-8")
        self.assertIn("kernel-call", skill)
        self.assertIn("text-only", skill)
        self.assertIn("single-image", skill)
        self.assertIn("扫描完成后", skill)
        self.assertIn("active_operator", skill)
        self.assertIn("当前 kernel 调用直接使用", skill)
        self.assertIn("capture/replay adapter", skill)

    def test_revision_5_cannot_fall_back_to_the_historical_mlp_operator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            run_dir = workspace / "capture"
            historical_spec = workspace / "migration-spec.md"
            write_contract(historical_spec, self.contract)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(CAPTURE_GOLDEN),
                    "--spec",
                    str(historical_spec),
                    "--run-dir",
                    str(run_dir),
                    "--mode",
                    "prepare",
                    "--scan-result",
                    str(SCAN_006),
                    "--operator-id",
                    "Step3p5MLP.forward",
                    "--sglang-worktree",
                    str(ROOT),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("Step3p5MLP.forward", completed.stderr)
            self.assertIn("exactly one operator", completed.stderr)
            self.assertFalse(run_dir.exists())

    def test_revision_5_cannot_use_the_historical_loaded_model_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            historical_spec = workspace / "migration-spec.md"
            write_contract(historical_spec, self.contract)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPLAY_COMPARE),
                    "--spec",
                    str(historical_spec),
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
            self.assertIn(
                "loaded-model MLP replay is not valid for a kernel-scan Contract",
                completed.stderr,
            )
            self.assertIn("--mode kernel-replay", completed.stderr)
            self.assertFalse((workspace / "replay").exists())

    def test_historical_revision_4_runs_are_unchanged(self) -> None:
        for relative_path, expected in HISTORICAL_SHA256.items():
            with self.subTest(path=relative_path):
                actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
