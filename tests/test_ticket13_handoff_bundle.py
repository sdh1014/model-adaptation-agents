import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
HANDOFF_BUNDLE = (
    REPO_ROOT / "model-adaptation" / "scripts" / "handoff_bundle.py"
)
MIGRATION_SPEC = REPO_ROOT / "migration-spec.md"


def current_contract() -> dict:
    text = MIGRATION_SPEC.read_text(encoding="utf-8")
    begin = "<!-- CONTRACT-DATA: BEGIN -->"
    end = "<!-- CONTRACT-DATA: END -->"
    return json.loads(text.split(begin, 1)[1].split(end, 1)[0])


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


def write_sealed_golden_run(path: Path, binding: dict) -> None:
    sample_name = "shape-001.pt"
    (path / "samples").mkdir(parents=True)
    (path / "self-replay").mkdir()
    (path / "capture-config.json").write_text(
        json.dumps({"spec_binding": binding}) + "\n",
        encoding="utf-8",
    )
    (path / "capture.log").write_text("capture=complete\n", encoding="utf-8")
    (path / "samples" / sample_name).write_bytes(b"portable-golden-sample")
    (path / "self-replay" / "result.json").write_text(
        json.dumps({"spec_binding": binding, "passed": True}) + "\n",
        encoding="utf-8",
    )
    (path / "capture-state.json").write_text(
        json.dumps(
            {
                "schema": "kernel-call-capture-state/v1",
                "spec_binding": binding,
                "status": "SEALED",
                "capture_closed": True,
                "saved_shape_count": 1,
                "samples": [
                    {
                        "shape_id": "shape-001",
                        "file": f"samples/{sample_name}",
                    }
                ],
                "self_replay": {
                    "passed": True,
                    "checked_shape_count": 1,
                    "worker_result_sha256": "a" * 64,
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_tool(*args: str) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    environment["PYTHONPYCACHEPREFIX"] = "/tmp/model-adaptation-agents-pyc"
    return subprocess.run(
        [sys.executable, str(HANDOFF_BUNDLE), *args],
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
                    "runs/golden-001/samples/shape-001.pt",
                    "runs/golden-001/self-replay/result.json",
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
            (bundle / "runs" / "golden-001" / "samples" / "shape-001.pt").unlink()
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
            self.assertIn("samples/shape-001.pt", result["error"])
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
                / "shape-001.pt"
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
            changed_contract["contract_revision"] += 1
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


if __name__ == "__main__":
    unittest.main()
