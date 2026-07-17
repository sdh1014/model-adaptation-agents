import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_GOLDEN = REPO_ROOT / "model-adaptation" / "scripts" / "capture_golden.py"
OPERATOR_ID = "activation.step_swiglu_with_limit"


def run_command(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPYCACHEPREFIX"] = "/tmp/model-adaptation-agents-pyc"
    return subprocess.run(
        list(args),
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def create_sglang_fixture(root: Path) -> tuple[Path, str]:
    source = root / "sglang"
    helper = source / "python" / "sglang" / "srt" / "models" / "step3p5_ops.py"
    helper.parent.mkdir(parents=True)
    helper.write_text(
        "def step_swiglu_with_limit(x, limit):\n"
        "    return x\n",
        encoding="utf-8",
    )
    for command in (
        ("git", "init", "-q"),
        ("git", "config", "user.email", "capture-test@example.com"),
        ("git", "config", "user.name", "Capture Test"),
        ("git", "add", "."),
        ("git", "commit", "-qm", "fixture"),
    ):
        completed = run_command(*command, cwd=source)
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
    revision = run_command("git", "rev-parse", "HEAD", cwd=source).stdout.strip()
    return source, revision


def write_spec(path: Path, revision: str) -> dict:
    contract = {
        "schema": "migration-spec/v0",
        "spec_id": "step3p7-flash-p800-demo",
        "contract_revision": 3,
        "model": "Step-3.7-Flash",
        "source": {
            "sglang_revision": revision,
            "sglang_kunlun_revision": "kunlun-revision",
        },
        "checkpoint": {
            "id": "stepfun-ai/Step-3.7-Flash@fixture",
            "config_digest": "config-digest",
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
    path.write_text(
        "\n".join(
            [
                "# Migration Spec",
                "<!-- CONTRACT-DATA: BEGIN -->",
                json.dumps(contract, ensure_ascii=False, indent=2),
                "<!-- CONTRACT-DATA: END -->",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return contract


def spec_binding(contract: dict) -> dict:
    canonical = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "spec_id": contract["spec_id"],
        "contract_revision": contract["contract_revision"],
        "contract_data_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def write_scan_result(path: Path, revision: str, binding: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "spec_binding": binding,
                "scan_complete": True,
                "source": {
                    "sglang_revision": revision,
                    "sglang_kunlun_revision": "kunlun-revision",
                    "checkpoint_id": "stepfun-ai/Step-3.7-Flash@fixture",
                    "checkpoint_config_sha256": "config-digest",
                },
                "operators": [
                    {
                        "operator_id": OPERATOR_ID,
                        "verdict": "CAPTURE_REQUIRED",
                        "model_path": ["target"],
                        "boundary": "gate_up plus scalar limit to output",
                        "state_dependency": "scalar limit only",
                        "cuda_impl": ["sglang:step3p5_ops.py:5-13"],
                        "kunlun_impl": ["sglang-kunlun:step3p5_ops.py:5-13"],
                    }
                ],
                "capture_plan": [
                    {
                        "operator_id": OPERATOR_ID,
                        "max_distinct_shapes": 3,
                        "reason": "weight-free capture candidate",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class Ticket12CudaCapturePreflightTest(unittest.TestCase):
    def test_prepare_binds_scan_candidate_and_fixed_source_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source, revision = create_sglang_fixture(workspace)
            spec = workspace / "migration-spec.md"
            scan_result = workspace / "scan-result.json"
            run_dir = workspace / "runs" / "golden-001"
            contract = write_spec(spec, revision)
            write_scan_result(scan_result, revision, spec_binding(contract))

            completed = run_command(
                sys.executable,
                str(CAPTURE_GOLDEN),
                "--spec",
                str(spec),
                "--run-dir",
                str(run_dir),
                "--mode",
                "prepare",
                "--scan-result",
                str(scan_result),
                "--operator-id",
                OPERATOR_ID,
                "--sglang-worktree",
                str(source),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            config = json.loads(
                (run_dir / "capture-config.json").read_text(encoding="utf-8")
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["action"], "prepare")
            self.assertEqual(result["capture_status"], "PREPARED")
            self.assertFalse(result["consumes_capture_session"])
            self.assertEqual(config["operator_id"], OPERATOR_ID)
            self.assertEqual(config["max_samples"], 3)
            self.assertEqual(config["source"]["sglang_revision"], revision)
            self.assertEqual(config["serialization"], "candidate-torch-save/v1")
            self.assertEqual(
                result["launch_environment"],
                {
                    "MODEL_ADAPTATION_CAPTURE_CONFIG": str(
                        (run_dir / "capture-config.json").resolve()
                    )
                },
            )
            self.assertTrue((run_dir / "samples").is_dir())

    def test_preflight_failure_is_evidence_but_does_not_consume_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source, revision = create_sglang_fixture(workspace)
            spec = workspace / "migration-spec.md"
            scan_result = workspace / "scan-result.json"
            run_dir = workspace / "runs" / "cuda-preflight-001"
            contract = write_spec(spec, revision)
            write_scan_result(scan_result, revision, spec_binding(contract))

            completed = run_command(
                sys.executable,
                str(CAPTURE_GOLDEN),
                "--spec",
                str(spec),
                "--run-dir",
                str(run_dir),
                "--mode",
                "preflight",
                "--scan-result",
                str(scan_result),
                "--operator-id",
                OPERATOR_ID,
                "--sglang-worktree",
                str(source),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            self.assertFalse(result["passed"])
            self.assertEqual(result["capture_status"], "PREFLIGHT_FAILED")
            self.assertFalse(result["consumes_capture_session"])
            self.assertTrue((run_dir / "preflight-capture.log").is_file())
            self.assertFalse((run_dir / "capture-state.json").exists())

    def test_fixed_source_mismatch_stops_before_creating_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source, _revision = create_sglang_fixture(workspace)
            expected_revision = "0" * 40
            spec = workspace / "migration-spec.md"
            scan_result = workspace / "scan-result.json"
            run_dir = workspace / "runs" / "golden-001"
            contract = write_spec(spec, expected_revision)
            write_scan_result(
                scan_result,
                expected_revision,
                spec_binding(contract),
            )

            completed = run_command(
                sys.executable,
                str(CAPTURE_GOLDEN),
                "--spec",
                str(spec),
                "--run-dir",
                str(run_dir),
                "--mode",
                "prepare",
                "--scan-result",
                str(scan_result),
                "--operator-id",
                OPERATOR_ID,
                "--sglang-worktree",
                str(source),
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("worktree revision does not match", completed.stderr)
            self.assertFalse(run_dir.exists())

    def test_scan_from_old_contract_is_rejected_before_creating_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source, revision = create_sglang_fixture(workspace)
            spec = workspace / "migration-spec.md"
            scan_result = workspace / "scan-result.json"
            run_dir = workspace / "runs" / "golden-001"
            contract = write_spec(spec, revision)
            old_binding = spec_binding(contract)
            old_binding["contract_revision"] = 2
            old_binding["contract_data_sha256"] = "0" * 64
            write_scan_result(scan_result, revision, old_binding)

            completed = run_command(
                sys.executable,
                str(CAPTURE_GOLDEN),
                "--spec",
                str(spec),
                "--run-dir",
                str(run_dir),
                "--mode",
                "prepare",
                "--scan-result",
                str(scan_result),
                "--operator-id",
                OPERATOR_ID,
                "--sglang-worktree",
                str(source),
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn(
                "Scan Run spec_binding does not match Contract Data",
                completed.stderr,
            )
            self.assertFalse(run_dir.exists())


if __name__ == "__main__":
    unittest.main()
