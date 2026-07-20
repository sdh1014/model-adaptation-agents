import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Optional
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
HANDOFF_BUNDLE = (
    REPO_ROOT / "model-adaptation" / "scripts" / "handoff_bundle.py"
)
MIGRATION_SPEC = REPO_ROOT / "migration-spec.md"
MODEL_ADAPTATION_SKILL = REPO_ROOT / "model-adaptation" / "SKILL.md"
OPERATOR_ID = (
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe."
    "_swiglu_silu_clamp_mul"
)


def current_contract() -> dict:
    text = MIGRATION_SPEC.read_text(encoding="utf-8")
    begin = "<!-- CONTRACT-DATA: BEGIN -->"
    end = "<!-- CONTRACT-DATA: END -->"
    contract = json.loads(text.split(begin, 1)[1].split(end, 1)[0])
    contract["contract_revision"] = 5
    return contract


def contract_binding(contract: dict) -> dict:
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "spec_id": contract["spec_id"],
        "contract_revision": contract["contract_revision"],
        "contract_data_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def write_spec(path: Path, contract: dict, state: str) -> None:
    path.write_text(
        "\n".join(
            [
                "# Migration Spec",
                "<!-- HUMAN-OWNED CONTRACT: BEGIN -->",
                "<!-- CONTRACT-DATA: BEGIN -->",
                json.dumps(contract, ensure_ascii=False, indent=2),
                "<!-- CONTRACT-DATA: END -->",
                "<!-- HUMAN-OWNED CONTRACT: END -->",
                "<!-- AGENT-WRITABLE WORKING STATE: BEGIN -->",
                state,
                "<!-- AGENT-WRITABLE WORKING STATE: END -->",
                "",
            ]
        ),
        encoding="utf-8",
    )


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def shape_id_for(shape: list[int]) -> str:
    return hashlib.sha256(canonical_json_bytes({"x": shape})).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sample_record_result_for(golden_run: Path) -> Path:
    return (
        golden_run.parent
        / f"{golden_run.name}-sample-record"
        / "result.json"
    )


def write_sealed_golden_run(
    path: Path,
    binding: dict,
    contract: Optional[dict] = None,
    sample_bytes: bytes = b"portable-golden-sample",
) -> None:
    if contract is None:
        contract = current_contract()
    shape_id = shape_id_for([1, 16])
    sample_name = f"{shape_id}.pt"
    checkpoint_id = contract["checkpoint"]["id"]
    model_path, revision = checkpoint_id.rsplit("@", 1)
    checkpoint = {
        "id": checkpoint_id,
        "model_path": model_path,
        "revision": revision,
        "config_digest": contract["checkpoint"]["config_digest"],
    }
    signature = {
        "operator_id": OPERATOR_ID,
        "model_path": "target",
        "tensor_parallel_size": 8,
        "tp_rank": 0,
        "inputs": {
            "x": {
                "kind": "tensor",
                "shape": [1, 16],
                "dtype": "bfloat16",
                "layout": "strided",
                "stride": [16, 1],
            }
        },
        "parameters": {},
        "non_tensor_args": {"gemm1_limit": 7.0},
    }
    (path / "samples").mkdir(parents=True)
    (path / "self-replay").mkdir()
    write_json(
        path / "capture-config.json",
        {
            "spec_binding": binding,
            "operator_id": OPERATOR_ID,
            "checkpoint": checkpoint,
            "tensor_parallel_size": 8,
            "tp_rank": 0,
            "precision_gate": contract["precision_gate"],
        },
    )
    write_json(
        path / "result.json",
        {
            "tool": "capture_golden.py",
            "action": "prepare",
            "spec_binding": binding,
            "operator_id": OPERATOR_ID,
            "passed": True,
        },
    )
    (path / "capture.log").write_text("capture=complete\n", encoding="utf-8")
    (path / "samples" / sample_name).write_bytes(sample_bytes)
    sample_file_evidence_path = path / "sample-files.json"
    sample_file_evidence = {
        "schema": "golden-sample-files/v1",
        "spec_binding": binding,
        "operator_id": OPERATOR_ID,
        "files": [
            {
                "shape_id": shape_id,
                "path": f"samples/{sample_name}",
                "size": len(sample_bytes),
                "sha256": hashlib.sha256(sample_bytes).hexdigest(),
            }
        ],
    }
    write_json(sample_file_evidence_path, sample_file_evidence)
    sample_files_digest = hashlib.sha256(
        sample_file_evidence_path.read_bytes()
    ).hexdigest()
    sample_record_result = sample_record_result_for(path)
    sample_record_result.parent.mkdir()
    write_json(
        sample_record_result,
        {
            "tool": "handoff_bundle.py",
            "action": "record-samples",
            "spec_binding": binding,
            "operator_id": OPERATOR_ID,
            "passed": True,
            "actual_tensors_saved": False,
            "golden_run": str(path.resolve()),
            "golden_status": "ACTIVE",
            "recorded_before_self_replay": True,
            "sample_file_count": 1,
            "sample_files_path": str(sample_file_evidence_path.resolve()),
            "sample_files_sha256": sample_files_digest,
            "evidence": ["handoff.log"],
        },
    )
    (sample_record_result.parent / "handoff.log").write_text(
        "action=record-samples\npassed=true\n",
        encoding="utf-8",
    )
    replay_config = {
        "schema": "kernel-call-replay-config/v1",
        "spec_binding": binding,
        "operator_id": OPERATOR_ID,
        "execution_site": "cuda",
        "invocation_target": OPERATOR_ID,
        "golden_run": str(path.resolve()),
        "run_dir": str((path / "self-replay").resolve()),
        "tensor_parallel_size": 8,
        "tp_rank": 0,
        "precision_gate": contract["precision_gate"],
        "sample_files_sha256": sample_files_digest,
        "allow_active_capture": True,
    }
    worker_result = {
        "schema": "kernel-call-replay-result/v1",
        "spec_binding": binding,
        "operator_id": OPERATOR_ID,
        "execution_site": "cuda",
        "invocation_target": OPERATOR_ID,
        "tp_rank": 0,
        "tensor_parallel_size": 8,
        "precision_gate": contract["precision_gate"],
        "sample_files_sha256": sample_files_digest,
        "actual_tensors_saved": False,
        "checked_shapes": [{"shape_id": shape_id, "passed": True}],
        "checked_shape_count": 1,
        "failed_shape_count": 0,
        "passed": True,
        "errors": [],
    }
    worker_digest = hashlib.sha256(canonical_json_bytes(worker_result)).hexdigest()
    write_json(path / "self-replay" / "replay-config.json", replay_config)
    write_json(path / "self-replay" / "worker-result.json", worker_result)
    write_json(
        path / "self-replay" / "result.json",
        {
            "tool": "replay_compare.py",
            "action": "kernel-replay",
            "spec_binding": binding,
            "operator_id": OPERATOR_ID,
            "execution_site": "cuda",
            "invocation_target": OPERATOR_ID,
            "passed": True,
            "checked_shape_count": 1,
            "failed_shape_count": 0,
            "precision_gate": contract["precision_gate"],
            "sample_files_sha256": sample_files_digest,
            "actual_tensors_saved": False,
            "evidence": [
                "replay-config.json",
                "worker-result.json",
                "replay.log",
            ],
        },
    )
    (path / "self-replay" / "replay.log").write_text(
        "execution_site=cuda\nreturncode=0\n",
        encoding="utf-8",
    )
    write_json(
        path / "capture-state.json",
        {
            "schema": "kernel-call-capture-state/v1",
            "spec_binding": binding,
            "operator_id": OPERATOR_ID,
            "tp_rank": 0,
            "tensor_parallel_size": 8,
            "checkpoint": checkpoint,
            "loaded_checkpoint": {
                "model_path": model_path,
                "revision": revision,
            },
            "status": "SEALED",
            "capture_closed": True,
            "saved_shape_count": 1,
            "samples": [
                {
                    "shape_id": shape_id,
                    "file": f"samples/{sample_name}",
                    "signature": signature,
                    "repeat_count": 0,
                }
            ],
            "self_replay": {
                "passed": True,
                "checked_shape_count": 1,
                "sample_files_sha256": sample_files_digest,
                "worker_result_sha256": worker_digest,
            },
        },
    )


def run_tool(*args: str) -> subprocess.CompletedProcess:
    arguments = list(args)
    if (
        "build" in arguments
        and "--sample-record-result" not in arguments
        and "--golden-run" in arguments
    ):
        golden_run = Path(arguments[arguments.index("--golden-run") + 1])
        arguments.extend(
            [
                "--sample-record-result",
                str(sample_record_result_for(golden_run)),
            ]
        )
    environment = os.environ.copy()
    environment["PYTHONPYCACHEPREFIX"] = "/tmp/model-adaptation-agents-pyc"
    return subprocess.run(
        [sys.executable, str(HANDOFF_BUNDLE), *arguments],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


class Ticket13HandoffBundleTest(unittest.TestCase):
    def test_build_and_verify_survive_a_manual_directory_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            contract = current_contract()
            binding = contract_binding(contract)
            current_spec = workspace / "migration-spec.md"
            next_spec = workspace / "waiting-migration-spec.md"
            golden_run = workspace / "runs" / "golden-001"
            build_run = workspace / "runs" / "handoff-build-001"

            write_spec(current_spec, contract, "status: ACTIVE")
            write_spec(next_spec, contract, "status: WAITING")
            current_spec_before = current_spec.read_bytes()
            write_sealed_golden_run(golden_run, binding)

            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(build_run),
                "--golden-run",
                str(golden_run),
                "--bundle-spec",
                str(next_spec),
            )

            self.assertEqual(built.returncode, 0, built.stderr)
            self.assertEqual(current_spec.read_bytes(), current_spec_before)
            build_result = json.loads(
                (build_run / "result.json").read_text(encoding="utf-8")
            )
            self.assertTrue(build_result["passed"])
            self.assertEqual(build_result["spec_binding"], binding)

            bundle = build_run / "bundle"
            self.assertEqual(
                (bundle / "migration-spec.md").read_bytes(),
                next_spec.read_bytes(),
            )
            manifest = json.loads(
                (bundle / "manifest.json").read_text(encoding="utf-8")
            )
            manifest_paths = [entry["path"] for entry in manifest["files"]]
            self.assertEqual(manifest_paths, sorted(manifest_paths))
            self.assertEqual(
                set(manifest_paths),
                {
                    "migration-spec.md",
                    "runs/golden-001/capture-config.json",
                    "runs/golden-001/capture-state.json",
                    "runs/golden-001/capture.log",
                    "runs/golden-001/result.json",
                    "runs/golden-001/sample-files.json",
                    (
                        "runs/golden-001/samples/"
                        f"{shape_id_for([1, 16])}.pt"
                    ),
                    "runs/golden-001/self-replay/replay-config.json",
                    "runs/golden-001/self-replay/replay.log",
                    "runs/golden-001/self-replay/result.json",
                    "runs/golden-001/self-replay/worker-result.json",
                },
            )

            copied_bundle = workspace / "manually-copied-bundle"
            shutil.copytree(bundle, copied_bundle)
            verify_run = workspace / "runs" / "handoff-verify-001"
            verified = run_tool(
                "--mode",
                "verify",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(verify_run),
                "--bundle-dir",
                str(copied_bundle),
            )

            self.assertEqual(verified.returncode, 0, verified.stderr)
            verify_result = json.loads(
                (verify_run / "result.json").read_text(encoding="utf-8")
            )
            self.assertTrue(verify_result["passed"])
            self.assertEqual(verify_result["spec_binding"], binding)
            self.assertEqual(
                verify_result["manifest_sha256"],
                build_result["manifest_sha256"],
            )
            self.assertEqual(
                {path.name for path in verify_run.iterdir()},
                {"result.json", "handoff.log"},
            )

    def test_verify_records_a_missing_file_as_failed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            contract = current_contract()
            binding = contract_binding(contract)
            current_spec = workspace / "migration-spec.md"
            next_spec = workspace / "waiting-migration-spec.md"
            golden_run = workspace / "runs" / "golden-001"
            build_run = workspace / "runs" / "handoff-build-001"

            write_spec(current_spec, contract, "status: ACTIVE")
            write_spec(next_spec, contract, "status: WAITING")
            write_sealed_golden_run(golden_run, binding)
            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(build_run),
                "--golden-run",
                str(golden_run),
                "--bundle-spec",
                str(next_spec),
            )
            self.assertEqual(built.returncode, 0, built.stderr)

            bundle = build_run / "bundle"
            next((bundle / "runs" / "golden-001" / "samples").glob("*.pt")).unlink()
            verify_run = workspace / "runs" / "handoff-verify-missing"
            verified = run_tool(
                "--mode",
                "verify",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(verify_run),
                "--bundle-dir",
                str(bundle),
            )

            self.assertEqual(verified.returncode, 0, verified.stderr)
            result = json.loads(
                (verify_run / "result.json").read_text(encoding="utf-8")
            )
            self.assertFalse(result["passed"])
            self.assertEqual(result["spec_binding"], binding)
            self.assertIn("missing file", result["error"])
            self.assertIn(
                f"samples/{shape_id_for([1, 16])}.pt",
                result["error"],
            )
            self.assertFalse(result["manifest_verified"])

    def test_verify_reports_an_unexpected_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            contract = current_contract()
            binding = contract_binding(contract)
            current_spec = workspace / "migration-spec.md"
            next_spec = workspace / "waiting-migration-spec.md"
            golden_run = workspace / "runs" / "golden-001"
            build_run = workspace / "runs" / "handoff-build-001"

            write_spec(current_spec, contract, "status: ACTIVE")
            write_spec(next_spec, contract, "status: WAITING")
            write_sealed_golden_run(golden_run, binding)
            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(build_run),
                "--golden-run",
                str(golden_run),
                "--bundle-spec",
                str(next_spec),
            )
            self.assertEqual(built.returncode, 0, built.stderr)

            bundle = build_run / "bundle"
            (bundle / "unexpected.txt").write_text("not in manifest\n", encoding="utf-8")
            verify_run = workspace / "runs" / "handoff-verify-extra"
            verified = run_tool(
                "--mode",
                "verify",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(verify_run),
                "--bundle-dir",
                str(bundle),
            )

            self.assertEqual(verified.returncode, 0, verified.stderr)
            result = json.loads(
                (verify_run / "result.json").read_text(encoding="utf-8")
            )
            self.assertFalse(result["passed"])
            self.assertIn("unexpected file", result["error"])
            self.assertIn("unexpected.txt", result["error"])

    def test_build_rejects_untrusted_self_replay_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            contract = current_contract()
            binding = contract_binding(contract)
            current_spec = workspace / "migration-spec.md"
            next_spec = workspace / "waiting-migration-spec.md"
            golden_run = workspace / "runs" / "golden-001"
            build_run = workspace / "runs" / "handoff-build-untrusted"

            write_spec(current_spec, contract, "status: ACTIVE")
            write_spec(next_spec, contract, "status: WAITING")
            write_sealed_golden_run(golden_run, binding)
            state_path = golden_run / "capture-state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["self_replay"]["worker_result_sha256"] = "not-a-digest"
            state_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(build_run),
                "--golden-run",
                str(golden_run),
                "--bundle-spec",
                str(next_spec),
            )

            self.assertEqual(built.returncode, 2)
            self.assertIn("worker_result_sha256", built.stderr)
            self.assertFalse(build_run.exists())

    def test_verify_rejects_size_content_and_contract_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            contract = current_contract()
            binding = contract_binding(contract)
            current_spec = workspace / "migration-spec.md"
            next_spec = workspace / "waiting-migration-spec.md"
            golden_run = workspace / "runs" / "golden-001"
            build_run = workspace / "runs" / "handoff-build-001"

            write_spec(current_spec, contract, "status: ACTIVE")
            write_spec(next_spec, contract, "status: WAITING")
            write_sealed_golden_run(golden_run, binding)
            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(build_run),
                "--golden-run",
                str(golden_run),
                "--bundle-spec",
                str(next_spec),
            )
            self.assertEqual(built.returncode, 0, built.stderr)
            original_bundle = build_run / "bundle"

            size_bundle = workspace / "size-changed-bundle"
            shutil.copytree(original_bundle, size_bundle)
            with (size_bundle / "runs" / "golden-001" / "capture.log").open(
                "ab"
            ) as stream:
                stream.write(b"changed")
            size_run = workspace / "runs" / "verify-size"
            size_result = run_tool(
                "--mode",
                "verify",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(size_run),
                "--bundle-dir",
                str(size_bundle),
            )
            self.assertEqual(size_result.returncode, 0, size_result.stderr)
            size_evidence = json.loads(
                (size_run / "result.json").read_text(encoding="utf-8")
            )
            self.assertFalse(size_evidence["passed"])
            self.assertIn("size differs", size_evidence["error"])

            content_bundle = workspace / "content-changed-bundle"
            shutil.copytree(original_bundle, content_bundle)
            sample_path = (
                content_bundle
                / "runs"
                / "golden-001"
                / "samples"
                / f"{shape_id_for([1, 16])}.pt"
            )
            sample_bytes = bytearray(sample_path.read_bytes())
            sample_bytes[0] ^= 0xFF
            sample_path.write_bytes(sample_bytes)
            content_run = workspace / "runs" / "verify-content"
            content_result = run_tool(
                "--mode",
                "verify",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(content_run),
                "--bundle-dir",
                str(content_bundle),
            )
            self.assertEqual(content_result.returncode, 0, content_result.stderr)
            content_evidence = json.loads(
                (content_run / "result.json").read_text(encoding="utf-8")
            )
            self.assertFalse(content_evidence["passed"])
            self.assertIn("sha256 differs", content_evidence["error"])

            changed_contract = json.loads(json.dumps(contract))
            changed_contract["source"]["sglang_revision"] = "0" * 40
            wrong_spec = workspace / "wrong-contract-spec.md"
            write_spec(wrong_spec, changed_contract, "status: WAITING")
            contract_run = workspace / "runs" / "verify-contract"
            contract_result = run_tool(
                "--mode",
                "verify",
                "--spec",
                str(wrong_spec),
                "--run-dir",
                str(contract_run),
                "--bundle-dir",
                str(original_bundle),
            )
            self.assertEqual(contract_result.returncode, 0, contract_result.stderr)
            contract_evidence = json.loads(
                (contract_run / "result.json").read_text(encoding="utf-8")
            )
            self.assertFalse(contract_evidence["passed"])
            self.assertIn("spec_binding", contract_evidence["error"])

    def test_build_rejects_a_non_golden_capture_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            contract = current_contract()
            binding = contract_binding(contract)
            current_spec = workspace / "migration-spec.md"
            next_spec = workspace / "waiting-migration-spec.md"
            golden_run = workspace / "runs" / "golden-001"
            build_run = workspace / "runs" / "handoff-build-wrong-schema"

            write_spec(current_spec, contract, "status: ACTIVE")
            write_spec(next_spec, contract, "status: WAITING")
            write_sealed_golden_run(golden_run, binding)
            state_path = golden_run / "capture-state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["schema"] = "not-a-golden-run"
            state_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(build_run),
                "--golden-run",
                str(golden_run),
                "--bundle-spec",
                str(next_spec),
            )

            self.assertEqual(built.returncode, 2)
            self.assertIn("schema", built.stderr)
            self.assertFalse(build_run.exists())

    def test_build_recomputes_the_cuda_self_replay_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            contract = current_contract()
            binding = contract_binding(contract)
            current_spec = workspace / "migration-spec.md"
            next_spec = workspace / "waiting-migration-spec.md"
            golden_run = workspace / "runs" / "golden-001"
            build_run = workspace / "runs" / "handoff-build-corrupt-replay"

            write_spec(current_spec, contract, "status: ACTIVE")
            write_spec(next_spec, contract, "status: WAITING")
            write_sealed_golden_run(golden_run, binding, contract)
            worker_path = golden_run / "self-replay" / "worker-result.json"
            worker = json.loads(worker_path.read_text(encoding="utf-8"))
            worker["checked_shapes"][0]["passed"] = False
            write_json(worker_path, worker)

            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(build_run),
                "--golden-run",
                str(golden_run),
                "--bundle-spec",
                str(next_spec),
            )

            self.assertEqual(built.returncode, 2)
            self.assertIn("worker_result_sha256", built.stderr)
            self.assertFalse(build_run.exists())

    def test_build_rejects_sample_bytes_changed_after_they_were_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            contract = current_contract()
            binding = contract_binding(contract)
            current_spec = workspace / "migration-spec.md"
            next_spec = workspace / "waiting-migration-spec.md"
            golden_run = workspace / "runs" / "golden-001"
            build_run = workspace / "runs" / "handoff-build-changed-sample"

            write_spec(current_spec, contract, "status: ACTIVE")
            write_spec(next_spec, contract, "status: WAITING")
            write_sealed_golden_run(golden_run, binding, contract)
            sample_path = next((golden_run / "samples").glob("*.pt"))
            sample_path.write_bytes(b"changed-after-self-replay")

            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(build_run),
                "--golden-run",
                str(golden_run),
                "--bundle-spec",
                str(next_spec),
            )

            self.assertEqual(built.returncode, 2)
            self.assertIn(
                "sample bytes do not match sample-files.json",
                built.stderr,
            )
            self.assertFalse(build_run.exists())

    def test_record_samples_writes_the_pre_replay_file_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            contract = current_contract()
            binding = contract_binding(contract)
            current_spec = workspace / "migration-spec.md"
            golden_run = workspace / "runs" / "golden-001"
            record_run = workspace / "runs" / "record-samples-001"

            write_spec(current_spec, contract, "status: ACTIVE")
            write_sealed_golden_run(golden_run, binding, contract)
            (golden_run / "sample-files.json").unlink()
            shutil.rmtree(golden_run / "self-replay")
            state_path = golden_run / "capture-state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["status"] = "ACTIVE"
            state["capture_closed"] = False
            state.pop("self_replay")
            write_json(state_path, state)

            recorded = run_tool(
                "--mode",
                "record-samples",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(record_run),
                "--golden-run",
                str(golden_run),
            )

            self.assertEqual(recorded.returncode, 0, recorded.stderr)
            result = json.loads(
                (record_run / "result.json").read_text(encoding="utf-8")
            )
            self.assertTrue(result["passed"])
            evidence = json.loads(
                (golden_run / "sample-files.json").read_text(encoding="utf-8")
            )
            sample_path = next((golden_run / "samples").glob("*.pt"))
            self.assertEqual(
                evidence["files"],
                [
                    {
                        "shape_id": shape_id_for([1, 16]),
                        "path": (
                            "samples/"
                            f"{shape_id_for([1, 16])}.pt"
                        ),
                        "size": sample_path.stat().st_size,
                        "sha256": hashlib.sha256(
                            sample_path.read_bytes()
                        ).hexdigest(),
                    }
                ],
            )

    def test_build_rejects_sample_and_sidecar_changed_after_self_replay(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            contract = current_contract()
            binding = contract_binding(contract)
            current_spec = workspace / "migration-spec.md"
            next_spec = workspace / "waiting-migration-spec.md"
            golden_run = workspace / "runs" / "golden-001"
            build_run = workspace / "runs" / "handoff-build-forged-samples"

            write_spec(current_spec, contract, "status: ACTIVE")
            write_spec(next_spec, contract, "status: WAITING")
            write_sealed_golden_run(golden_run, binding, contract)
            sample_path = next((golden_run / "samples").glob("*.pt"))
            sample_path.write_bytes(b"changed-after-self-replay")
            sidecar_path = golden_run / "sample-files.json"
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["files"][0]["size"] = sample_path.stat().st_size
            sidecar["files"][0]["sha256"] = hashlib.sha256(
                sample_path.read_bytes()
            ).hexdigest()
            write_json(sidecar_path, sidecar)
            forged_sample_digest = hashlib.sha256(
                sidecar_path.read_bytes()
            ).hexdigest()
            replay_dir = golden_run / "self-replay"
            replay_config_path = replay_dir / "replay-config.json"
            replay_config = json.loads(
                replay_config_path.read_text(encoding="utf-8")
            )
            replay_config["sample_files_sha256"] = forged_sample_digest
            write_json(replay_config_path, replay_config)
            worker_path = replay_dir / "worker-result.json"
            worker = json.loads(worker_path.read_text(encoding="utf-8"))
            worker["sample_files_sha256"] = forged_sample_digest
            write_json(worker_path, worker)
            replay_result_path = replay_dir / "result.json"
            replay_result = json.loads(
                replay_result_path.read_text(encoding="utf-8")
            )
            replay_result["sample_files_sha256"] = forged_sample_digest
            write_json(replay_result_path, replay_result)
            state_path = golden_run / "capture-state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["self_replay"]["sample_files_sha256"] = forged_sample_digest
            state["self_replay"]["worker_result_sha256"] = hashlib.sha256(
                canonical_json_bytes(worker)
            ).hexdigest()
            write_json(state_path, state)

            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(build_run),
                "--golden-run",
                str(golden_run),
                "--bundle-spec",
                str(next_spec),
            )

            self.assertEqual(built.returncode, 2)
            self.assertIn(
                "record-samples result does not match sample-files.json",
                built.stderr,
            )
            self.assertFalse(build_run.exists())

    def test_build_rejects_a_golden_run_for_another_operator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            contract = current_contract()
            binding = contract_binding(contract)
            current_spec = workspace / "migration-spec.md"
            next_spec = workspace / "waiting-migration-spec.md"
            golden_run = workspace / "runs" / "golden-001"
            build_run = workspace / "runs" / "handoff-build-wrong-operator"

            write_spec(current_spec, contract, "status: ACTIVE")
            write_spec(next_spec, contract, "status: WAITING")
            write_sealed_golden_run(golden_run, binding, contract)
            state_path = golden_run / "capture-state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["operator_id"] = "another.operator"
            write_json(state_path, state)

            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(build_run),
                "--golden-run",
                str(golden_run),
                "--bundle-spec",
                str(next_spec),
            )

            self.assertEqual(built.returncode, 2)
            self.assertIn("operator_id", built.stderr)
            self.assertFalse(build_run.exists())

    def test_verify_rejects_a_result_directory_inside_the_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            contract = current_contract()
            binding = contract_binding(contract)
            current_spec = workspace / "migration-spec.md"
            next_spec = workspace / "waiting-migration-spec.md"
            golden_run = workspace / "runs" / "golden-001"
            build_run = workspace / "runs" / "handoff-build-001"

            write_spec(current_spec, contract, "status: ACTIVE")
            write_spec(next_spec, contract, "status: WAITING")
            write_sealed_golden_run(golden_run, binding, contract)
            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(build_run),
                "--golden-run",
                str(golden_run),
                "--bundle-spec",
                str(next_spec),
            )
            self.assertEqual(built.returncode, 0, built.stderr)

            bundle = build_run / "bundle"
            verify_run = bundle / "verify-output"
            verified = run_tool(
                "--mode",
                "verify",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(verify_run),
                "--bundle-dir",
                str(bundle),
            )

            self.assertEqual(verified.returncode, 2)
            self.assertIn("must not overlap", verified.stderr)
            self.assertFalse(verify_run.exists())

    def test_build_rejects_a_result_directory_inside_the_golden_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            contract = current_contract()
            binding = contract_binding(contract)
            current_spec = workspace / "migration-spec.md"
            next_spec = workspace / "waiting-migration-spec.md"
            golden_run = workspace / "runs" / "golden-001"
            build_run = golden_run / "handoff-build"

            write_spec(current_spec, contract, "status: ACTIVE")
            write_spec(next_spec, contract, "status: WAITING")
            write_sealed_golden_run(golden_run, binding, contract)

            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(build_run),
                "--golden-run",
                str(golden_run),
                "--bundle-spec",
                str(next_spec),
            )

            self.assertEqual(built.returncode, 2)
            self.assertIn("must not overlap", built.stderr)
            self.assertFalse(build_run.exists())

    def test_build_rejects_a_result_directory_inside_sample_record_run(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            contract = current_contract()
            binding = contract_binding(contract)
            current_spec = workspace / "migration-spec.md"
            next_spec = workspace / "waiting-migration-spec.md"
            golden_run = workspace / "runs" / "golden-001"

            write_spec(current_spec, contract, "status: ACTIVE")
            write_spec(next_spec, contract, "status: WAITING")
            write_sealed_golden_run(golden_run, binding, contract)
            build_run = (
                sample_record_result_for(golden_run).parent
                / "nested-build"
            )

            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(build_run),
                "--golden-run",
                str(golden_run),
                "--bundle-spec",
                str(next_spec),
            )

            self.assertEqual(built.returncode, 2)
            self.assertIn("sample record Run must not overlap", built.stderr)
            self.assertFalse(build_run.exists())

    def test_build_rejects_self_replay_from_different_sample_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            contract = current_contract()
            binding = contract_binding(contract)
            current_spec = workspace / "migration-spec.md"
            next_spec = workspace / "waiting-migration-spec.md"
            golden_run = workspace / "runs" / "golden-001"
            other_golden = workspace / "runs" / "golden-002"
            build_run = workspace / "runs" / "handoff-build-swapped-replay"

            write_spec(current_spec, contract, "status: ACTIVE")
            write_spec(next_spec, contract, "status: WAITING")
            write_sealed_golden_run(golden_run, binding, contract)
            write_sealed_golden_run(
                other_golden,
                binding,
                contract,
                sample_bytes=b"different-sample-bytes",
            )
            shutil.rmtree(golden_run / "self-replay")
            shutil.copytree(
                other_golden / "self-replay",
                golden_run / "self-replay",
            )

            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(build_run),
                "--golden-run",
                str(golden_run),
                "--bundle-spec",
                str(next_spec),
            )

            self.assertEqual(built.returncode, 2)
            self.assertIn("sample_files_sha256", built.stderr)
            self.assertFalse(build_run.exists())

    def test_verify_rejects_an_extra_file_even_when_manifested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            contract = current_contract()
            binding = contract_binding(contract)
            current_spec = workspace / "migration-spec.md"
            next_spec = workspace / "waiting-migration-spec.md"
            golden_run = workspace / "runs" / "golden-001"
            build_run = workspace / "runs" / "handoff-build-001"

            write_spec(current_spec, contract, "status: ACTIVE")
            write_spec(next_spec, contract, "status: WAITING")
            write_sealed_golden_run(golden_run, binding, contract)
            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(build_run),
                "--golden-run",
                str(golden_run),
                "--bundle-spec",
                str(next_spec),
            )
            self.assertEqual(built.returncode, 0, built.stderr)

            bundle = build_run / "bundle"
            extra = bundle / "runs" / "golden-001" / "unrelated.txt"
            extra.write_text("manifested but not permitted\n", encoding="utf-8")
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"].append(
                {
                    "path": "runs/golden-001/unrelated.txt",
                    "size": extra.stat().st_size,
                    "sha256": hashlib.sha256(extra.read_bytes()).hexdigest(),
                }
            )
            manifest["files"].sort(key=lambda entry: entry["path"])
            write_json(manifest_path, manifest)

            verify_run = workspace / "runs" / "handoff-verify-manifested-extra"
            verified = run_tool(
                "--mode",
                "verify",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(verify_run),
                "--bundle-dir",
                str(bundle),
            )

            self.assertEqual(verified.returncode, 0, verified.stderr)
            result = json.loads(
                (verify_run / "result.json").read_text(encoding="utf-8")
            )
            self.assertFalse(result["passed"])
            self.assertIn("not permitted", result["error"])

    def test_verify_rejects_a_manifested_extra_file_at_bundle_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            contract = current_contract()
            binding = contract_binding(contract)
            current_spec = workspace / "migration-spec.md"
            next_spec = workspace / "waiting-migration-spec.md"
            golden_run = workspace / "runs" / "golden-001"
            build_run = workspace / "runs" / "handoff-build-001"

            write_spec(current_spec, contract, "status: ACTIVE")
            write_spec(next_spec, contract, "status: WAITING")
            write_sealed_golden_run(golden_run, binding, contract)
            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(build_run),
                "--golden-run",
                str(golden_run),
                "--bundle-spec",
                str(next_spec),
            )
            self.assertEqual(built.returncode, 0, built.stderr)

            bundle = build_run / "bundle"
            extra = bundle / "unrelated-root.txt"
            extra.write_text("manifested but not permitted\n", encoding="utf-8")
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"].append(
                {
                    "path": "unrelated-root.txt",
                    "size": extra.stat().st_size,
                    "sha256": hashlib.sha256(extra.read_bytes()).hexdigest(),
                }
            )
            manifest["files"].sort(key=lambda entry: entry["path"])
            write_json(manifest_path, manifest)

            verify_run = workspace / "runs" / "handoff-verify-root-extra"
            verified = run_tool(
                "--mode",
                "verify",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(verify_run),
                "--bundle-dir",
                str(bundle),
            )

            self.assertEqual(verified.returncode, 0, verified.stderr)
            result = json.loads(
                (verify_run / "result.json").read_text(encoding="utf-8")
            )
            self.assertFalse(result["passed"])
            self.assertIn("not permitted", result["error"])
            self.assertIn("unrelated-root.txt", result["error"])

    def test_skill_keeps_working_state_validation_with_the_agent(self) -> None:
        skill = MODEL_ADAPTATION_SKILL.read_text(encoding="utf-8")
        self.assertIn("临时 Spec", skill)
        self.assertIn("WAITING / HANDOFF", skill)
        self.assertIn("原子", skill)
        self.assertIn("不解析 Working State", skill)


if __name__ == "__main__":
    unittest.main()
