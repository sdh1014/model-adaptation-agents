import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
REPLAY_COMPARE = REPO_ROOT / "model-adaptation" / "scripts" / "replay_compare.py"
SPEC_TEMPLATE = (
    REPO_ROOT / "model-adaptation" / "references" / "migration-spec-template.md"
)
MODEL_ADAPTATION_SKILL = REPO_ROOT / "model-adaptation" / "SKILL.md"
EXPECTED_CONTRACT_SHA256 = (
    "9ee412b66d22e66c0d6904f3cfbf2b5fae9c8311cb32ad0668f2ffc37a0251db"
)


def approved_contract() -> dict:
    return {
        "schema": "migration-spec/v0",
        "spec_id": "step3p7-flash-p800-demo",
        "contract_revision": 1,
        "model": "Step-3.7-Flash",
        "source": {
            "sglang_revision": "sglang-rev",
            "sglang_kunlun_revision": "kunlun-rev",
        },
        "checkpoint": {
            "id": "checkpoint-rev",
            "config_digest": "config-sha256",
        },
        "model_path": {
            "target_entry": "Step3p7ForConditionalGeneration.forward",
            "draft_entry": None,
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
            "max_samples_per_operator": 3,
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


def write_spec(path: Path, contract: dict, working_state: str = "state_revision: 0") -> None:
    contract_json = json.dumps(contract, ensure_ascii=False, indent=2)
    path.write_text(
        "\n".join(
            [
                "# Migration Spec",
                "<!-- HUMAN-OWNED CONTRACT: BEGIN -->",
                "<!-- CONTRACT-DATA: BEGIN -->",
                contract_json,
                "<!-- CONTRACT-DATA: END -->",
                "<!-- HUMAN-OWNED CONTRACT: END -->",
                "<!-- AGENT-WRITABLE WORKING STATE: BEGIN -->",
                working_state,
                "<!-- AGENT-WRITABLE WORKING STATE: END -->",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_tool(*args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPYCACHEPREFIX"] = "/tmp/model-adaptation-agents-pyc"
    return subprocess.run(
        [sys.executable, str(REPLAY_COMPARE), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


class Ticket10SpecBindingTest(unittest.TestCase):
    def test_target_only_eager_contract_runs_without_speculative_decoding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            spec_path = workspace / "migration-spec.md"
            run_dir = workspace / "runs" / "synthetic-eager"
            write_spec(spec_path, approved_contract())

            completed = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(run_dir),
                "--mode",
                "synthetic",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            self.assertTrue(result["passed"])

    def test_approved_spec_runs_synthetic_comparison_with_contract_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            spec_path = workspace / "migration-spec.md"
            case_path = workspace / "case.json"
            run_dir = workspace / "runs" / "synthetic-001"
            write_spec(spec_path, approved_contract())
            case_path.write_text(
                json.dumps({"expected": [1.0, 2.0], "actual": [1.0, 2.0]}),
                encoding="utf-8",
            )

            completed = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(run_dir),
                "--mode",
                "synthetic",
                "--case",
                str(case_path),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["tool"], "replay_compare.py")
            self.assertEqual(result["action"], "synthetic")
            self.assertTrue(result["passed"])
            self.assertEqual(
                result["spec_binding"],
                {
                    "spec_id": "step3p7-flash-p800-demo",
                    "contract_revision": 1,
                    "contract_data_sha256": EXPECTED_CONTRACT_SHA256,
                },
            )
            self.assertEqual(result["evidence"], ["synthetic-case.json", "replay.log"])
            self.assertTrue((run_dir / "synthetic-case.json").is_file())
            self.assertTrue((run_dir / "replay.log").is_file())

    def test_template_with_unapproved_placeholders_stops_before_creating_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            spec_path = workspace / "migration-spec.md"
            case_path = workspace / "case.json"
            run_dir = workspace / "runs" / "synthetic-001"
            template = SPEC_TEMPLATE.read_text(encoding="utf-8")
            self.assertIn('"model": null', template)
            draft = template.replace(
                '"model": null', '"model": "Step-3.7-Flash"', 1
            )
            spec_path.write_text(draft, encoding="utf-8")
            case_path.write_text(
                json.dumps({"expected": [1.0], "actual": [1.0]}),
                encoding="utf-8",
            )

            completed = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(run_dir),
                "--mode",
                "synthetic",
                "--case",
                str(case_path),
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("Contract Data is not approved", completed.stderr)
            self.assertIn("source.sglang_revision", completed.stderr)
            self.assertNotIn("precision_gate.atol", completed.stderr)
            self.assertFalse(run_dir.exists())
            self.assertIn("- `status`: `NEEDS_HUMAN`", draft)
            self.assertIn("- `phase`: `SCAN`", draft)
            self.assertIn("- `next_action`: `none`", draft)

    def test_template_defines_target_only_eager_runtime(self) -> None:
        template = SPEC_TEMPLATE.read_text(encoding="utf-8")

        self.assertIn('"draft_entry": null', template)
        self.assertIn('"speculative_algorithm": null', template)
        self.assertIn('"cuda_graph_backend_decode": "disabled"', template)
        self.assertIn('"cuda_graph_backend_prefill": "disabled"', template)

    def test_business_failure_is_a_result_but_contract_drift_is_a_tool_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            spec_path = workspace / "migration-spec.md"
            case_path = workspace / "case.json"
            write_spec(spec_path, approved_contract())
            case_path.write_text(
                json.dumps({"expected": [1.0], "actual": [2.0]}),
                encoding="utf-8",
            )

            comparison = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(workspace / "runs" / "synthetic-fail"),
                "--mode",
                "synthetic",
                "--case",
                str(case_path),
            )

            self.assertEqual(comparison.returncode, 0, comparison.stderr)
            comparison_result = json.loads(
                (workspace / "runs" / "synthetic-fail" / "result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertFalse(comparison_result["passed"])

            drifted_contract = approved_contract()
            drifted_contract["limits"]["max_samples_per_operator"] = 4
            write_spec(spec_path, drifted_contract)
            rejected_run = workspace / "runs" / "contract-drift"

            rejected = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(rejected_run),
                "--mode",
                "synthetic",
                "--case",
                str(case_path),
            )

            self.assertEqual(rejected.returncode, 2)
            self.assertIn("max_samples_per_operator must be 3", rejected.stderr)
            self.assertFalse(rejected_run.exists())

    def test_existing_result_survives_working_state_change_but_not_contract_change(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            spec_path = workspace / "migration-spec.md"
            case_path = workspace / "case.json"
            original_run = workspace / "runs" / "synthetic-001"
            contract = approved_contract()
            write_spec(spec_path, contract, working_state="state_revision: 0")
            case_path.write_text(
                json.dumps({"expected": [1.0], "actual": [1.0]}),
                encoding="utf-8",
            )
            created = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(original_run),
                "--mode",
                "synthetic",
                "--case",
                str(case_path),
            )
            self.assertEqual(created.returncode, 0, created.stderr)

            write_spec(spec_path, contract, working_state="state_revision: 1")
            working_state_check = workspace / "runs" / "binding-working-state"
            accepted = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(working_state_check),
                "--mode",
                "validate-binding",
                "--result",
                str(original_run / "result.json"),
            )

            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            accepted_result = json.loads(
                (working_state_check / "result.json").read_text(encoding="utf-8")
            )
            self.assertTrue(accepted_result["passed"])
            self.assertEqual(accepted_result["mismatched_fields"], [])

            revised_contract = approved_contract()
            revised_contract["contract_revision"] = 2
            revised_contract["checkpoint"]["id"] = "checkpoint-rev-2"
            write_spec(spec_path, revised_contract, working_state="state_revision: 2")
            contract_check = workspace / "runs" / "binding-contract-change"
            rejected = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(contract_check),
                "--mode",
                "validate-binding",
                "--result",
                str(original_run / "result.json"),
            )

            self.assertEqual(rejected.returncode, 0, rejected.stderr)
            rejected_result = json.loads(
                (contract_check / "result.json").read_text(encoding="utf-8")
            )
            self.assertFalse(rejected_result["passed"])
            self.assertEqual(
                rejected_result["mismatched_fields"],
                ["contract_revision", "contract_data_sha256"],
            )

    def test_skill_defines_the_first_approved_contract_transition(self) -> None:
        skill = MODEL_ADAPTATION_SKILL.read_text(encoding="utf-8")

        self.assertIn(
            "`observed_contract_revision` 设为当前 `contract_revision`", skill
        )
        self.assertIn("`state_revision` 加一", skill)
        self.assertIn("`status: ACTIVE`、`phase: SCAN`", skill)
        self.assertIn("`last_completed_action: contract_approved`", skill)
        self.assertIn("`next_action: 运行 Spec 绑定自检`", skill)
        self.assertIn("`last_completed_action: spec_binding_smoke_passed`", skill)
        self.assertIn("`last_run: runs/spec-binding-001`", skill)
        self.assertIn("`next_action: 完成 target-only eager 扫描并生成 Scan Run`", skill)

    def test_binding_smoke_run_is_recoverable_from_the_same_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            spec_path = workspace / "migration-spec.md"
            smoke_run = workspace / "runs" / "spec-binding-001"
            write_spec(
                spec_path,
                approved_contract(),
                working_state="\n".join(
                    [
                        "observed_contract_revision: 1",
                        "state_revision: 1",
                        "status: ACTIVE",
                        "phase: SCAN",
                        "last_completed_action: contract_approved",
                        "last_run: null",
                        "next_action: 运行 Spec 绑定自检",
                    ]
                ),
            )

            completed = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(smoke_run),
                "--mode",
                "synthetic",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            smoke_result = json.loads(
                (smoke_run / "result.json").read_text(encoding="utf-8")
            )
            self.assertTrue(smoke_result["passed"])

            recovered_working_state = "\n".join(
                [
                    "observed_contract_revision: 1",
                    "state_revision: 2",
                    "status: ACTIVE",
                    "phase: SCAN",
                    "last_completed_action: spec_binding_smoke_passed",
                    "last_run: runs/spec-binding-001",
                    "next_action: 完成 target-only eager 扫描并生成 Scan Run",
                ]
            )
            write_spec(spec_path, approved_contract(), recovered_working_state)

            recovery_check = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(workspace / "runs" / "recovery-check-001"),
                "--mode",
                "validate-binding",
                "--result",
                str(smoke_run / "result.json"),
            )

            self.assertEqual(recovery_check.returncode, 0, recovery_check.stderr)
            recovery_result = json.loads(
                (
                    workspace / "runs" / "recovery-check-001" / "result.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(recovery_result["passed"])
            restored_spec = spec_path.read_text(encoding="utf-8")
            self.assertIn("last_run: runs/spec-binding-001", restored_spec)
            self.assertIn(
                "next_action: 完成 target-only eager 扫描并生成 Scan Run",
                restored_spec,
            )

    def test_reversed_contract_markers_are_a_clean_tool_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            spec_path = workspace / "migration-spec.md"
            case_path = workspace / "case.json"
            run_dir = workspace / "runs" / "invalid-markers"
            spec_path.write_text(
                "\n".join(
                    [
                        "<!-- CONTRACT-DATA: END -->",
                        json.dumps(approved_contract()),
                        "<!-- CONTRACT-DATA: BEGIN -->",
                    ]
                ),
                encoding="utf-8",
            )
            case_path.write_text(
                json.dumps({"expected": 1, "actual": 1}), encoding="utf-8"
            )

            completed = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(run_dir),
                "--mode",
                "synthetic",
                "--case",
                str(case_path),
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("Contract Data markers are out of order", completed.stderr)
            self.assertNotIn("Traceback", completed.stderr)
            self.assertFalse(run_dir.exists())

    def test_non_finite_precision_tolerance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            spec_path = workspace / "migration-spec.md"
            case_path = workspace / "case.json"
            run_dir = workspace / "runs" / "invalid-tolerance"
            contract = approved_contract()
            contract["precision_gate"]["atol"] = float("nan")
            write_spec(spec_path, contract)
            case_path.write_text(
                json.dumps({"expected": 1, "actual": 1}), encoding="utf-8"
            )

            completed = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(run_dir),
                "--mode",
                "synthetic",
                "--case",
                str(case_path),
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn(
                "precision_gate.atol must be a finite non-negative number",
                completed.stderr,
            )
            self.assertFalse(run_dir.exists())

    def test_tp8_bf16_target_only_eager_runtime_profile_is_required_and_fixed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            spec_path = workspace / "migration-spec.md"
            case_path = workspace / "case.json"
            case_path.write_text(
                json.dumps({"expected": 1, "actual": 1}), encoding="utf-8"
            )

            missing_runtime = approved_contract()
            del missing_runtime["runtime"]
            write_spec(spec_path, missing_runtime)
            missing = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(workspace / "runs" / "missing-runtime"),
                "--mode",
                "synthetic",
                "--case",
                str(case_path),
            )

            self.assertEqual(missing.returncode, 2)
            self.assertIn("runtime.tensor_parallel_size", missing.stderr)
            self.assertIn("runtime.dtype", missing.stderr)
            self.assertIn("runtime.cuda_graph_backend_decode", missing.stderr)
            self.assertIn("runtime.cuda_graph_backend_prefill", missing.stderr)

            wrong_dtype = approved_contract()
            wrong_dtype["runtime"]["dtype"] = "float16"
            write_spec(spec_path, wrong_dtype)
            rejected = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(workspace / "runs" / "wrong-dtype"),
                "--mode",
                "synthetic",
                "--case",
                str(case_path),
            )

            self.assertEqual(rejected.returncode, 2)
            self.assertIn("runtime.dtype must be 'bfloat16'", rejected.stderr)
            self.assertFalse((workspace / "runs" / "wrong-dtype").exists())

            wrong_tp = approved_contract()
            wrong_tp["runtime"]["tensor_parallel_size"] = 4
            write_spec(spec_path, wrong_tp)
            rejected = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(workspace / "runs" / "wrong-tp"),
                "--mode",
                "synthetic",
                "--case",
                str(case_path),
            )

            self.assertEqual(rejected.returncode, 2)
            self.assertIn(
                "runtime.tensor_parallel_size must be 8", rejected.stderr
            )
            self.assertFalse((workspace / "runs" / "wrong-tp").exists())

            wrong_quantization = approved_contract()
            wrong_quantization["runtime"]["quantization"] = "w8a8_int8"
            write_spec(spec_path, wrong_quantization)
            rejected = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(workspace / "runs" / "wrong-quantization"),
                "--mode",
                "synthetic",
                "--case",
                str(case_path),
            )

            self.assertEqual(rejected.returncode, 2)
            self.assertIn(
                "runtime.quantization must be None", rejected.stderr
            )
            self.assertFalse(
                (workspace / "runs" / "wrong-quantization").exists()
            )

            wrong_speculative_algorithm = approved_contract()
            wrong_speculative_algorithm["runtime"]["speculative_algorithm"] = "EAGLE"
            write_spec(spec_path, wrong_speculative_algorithm)
            rejected = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(workspace / "runs" / "wrong-speculative-algorithm"),
                "--mode",
                "synthetic",
                "--case",
                str(case_path),
            )

            self.assertEqual(rejected.returncode, 2)
            self.assertIn(
                "runtime.speculative_algorithm must be None", rejected.stderr
            )
            self.assertFalse(
                (workspace / "runs" / "wrong-speculative-algorithm").exists()
            )

            wrong_decode_graph_backend = approved_contract()
            wrong_decode_graph_backend["runtime"][
                "cuda_graph_backend_decode"
            ] = "full"
            write_spec(spec_path, wrong_decode_graph_backend)
            rejected = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(workspace / "runs" / "decode-graph-enabled"),
                "--mode",
                "synthetic",
                "--case",
                str(case_path),
            )

            self.assertEqual(rejected.returncode, 2)
            self.assertIn(
                "runtime.cuda_graph_backend_decode must be 'disabled'",
                rejected.stderr,
            )
            self.assertFalse(
                (workspace / "runs" / "decode-graph-enabled").exists()
            )

            wrong_prefill_graph_backend = approved_contract()
            wrong_prefill_graph_backend["runtime"][
                "cuda_graph_backend_prefill"
            ] = "tc_piecewise"
            write_spec(spec_path, wrong_prefill_graph_backend)
            rejected = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(workspace / "runs" / "prefill-graph-enabled"),
                "--mode",
                "synthetic",
                "--case",
                str(case_path),
            )

            self.assertEqual(rejected.returncode, 2)
            self.assertIn(
                "runtime.cuda_graph_backend_prefill must be 'disabled'",
                rejected.stderr,
            )
            self.assertFalse(
                (workspace / "runs" / "prefill-graph-enabled").exists()
            )

            explicit_attention_backend = approved_contract()
            explicit_attention_backend["runtime"]["attention_backend"] = "fa3"
            write_spec(spec_path, explicit_attention_backend)
            rejected = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(workspace / "runs" / "explicit-attention-backend"),
                "--mode",
                "synthetic",
                "--case",
                str(case_path),
            )

            self.assertEqual(rejected.returncode, 2)
            self.assertIn(
                "runtime.attention_backend must be None", rejected.stderr
            )
            self.assertFalse(
                (workspace / "runs" / "explicit-attention-backend").exists()
            )

            wrong_tolerance = approved_contract()
            wrong_tolerance["precision_gate"]["atol"] = 0.02
            write_spec(spec_path, wrong_tolerance)
            rejected = run_tool(
                "--spec",
                str(spec_path),
                "--run-dir",
                str(workspace / "runs" / "wrong-tolerance"),
                "--mode",
                "synthetic",
                "--case",
                str(case_path),
            )

            self.assertEqual(rejected.returncode, 2)
            self.assertIn(
                "precision_gate.atol must be 0.01", rejected.stderr
            )
            self.assertFalse(
                (workspace / "runs" / "wrong-tolerance").exists()
            )


if __name__ == "__main__":
    unittest.main()
