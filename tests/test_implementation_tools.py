from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CREATE_RUN = ROOT / "scripts/implementation/create_run.py"
CREATE_ATTEMPT = ROOT / "scripts/implementation/create_attempt.py"
SNAPSHOT = ROOT / "scripts/implementation/snapshot_repo.py"
CHECK_SCOPE = ROOT / "scripts/implementation/check_scope.py"
CHECK_IMPLEMENTATION = ROOT / "scripts/implementation/check_implementation.sh"
FAILURE_SIGNATURE = ROOT / "scripts/implementation/failure_signature.py"
HISTORY = ROOT / "scripts/implementation/history.py"
COLLECT_CONTEXT = ROOT / "scripts/implementation/collect_context.py"
RUN_BASH = ROOT / "scripts/run_bash.py"


def run(*args: str, check: bool = True, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        check=check,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def init_repo(path: Path) -> str:
    path.mkdir(parents=True)
    run("git", "init", "-q", str(path))
    run("git", "-C", str(path), "config", "user.email", "test@example.com")
    run("git", "-C", str(path), "config", "user.name", "Test")
    (path / "allowed").mkdir()
    (path / "allowed/main.py").write_text("VALUE = 1\n", encoding="utf-8")
    (path / "outside.txt").write_text("base\n", encoding="utf-8")
    run("git", "-C", str(path), "add", ".")
    run("git", "-C", str(path), "commit", "-qm", "initial")
    return run("git", "-C", str(path), "rev-parse", "HEAD").stdout.strip()


class ImplementationToolTests(unittest.TestCase):
    def test_create_run_and_history_include_attempt_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            init_repo(repo)
            runs = base / "runs"
            proc = run(
                sys.executable,
                str(CREATE_RUN),
                "--model-id",
                "demo",
                "--target-id",
                "vllm-kunlun",
                "--item-id",
                "WI-001",
                "--target-repo",
                str(repo),
                "--repository-role",
                "target_repo",
                "--runs-root",
                str(runs),
            )
            run_dir = Path(proc.stdout.strip())
            attempt_proc = run(
                sys.executable,
                str(CREATE_ATTEMPT),
                "--run-dir",
                str(run_dir),
                "--max-attempts",
                "3",
            )
            attempt = Path(attempt_proc.stdout.strip())
            command = attempt / "verification/command"
            command.mkdir(parents=True)
            (attempt / "attempt.md").write_text(
                "# Attempt\n\n- Hypothesis: registry entry is missing.\n",
                encoding="utf-8",
            )
            (command / "command.json").write_text(
                json.dumps({"argv": ["python", "-m", "pytest", "test_registry.py"]}),
                encoding="utf-8",
            )
            (command / "result.json").write_text(
                json.dumps({"exit_code": 1, "timed_out": False}), encoding="utf-8"
            )
            (command / "stdout.log").write_text("collected 1 item\n", encoding="utf-8")
            (command / "stderr.log").write_text("RuntimeError: missing model\n", encoding="utf-8")
            (attempt / "failure-signature.json").write_text(
                json.dumps({"signature_hash": "abc", "signature": {"exit_code": 1}}),
                encoding="utf-8",
            )
            (run_dir / "outcome.json").write_text(
                json.dumps(
                    {
                        "status": "in_progress",
                        "summary": "registry test failed",
                        "next_action": "try a distinct registration hypothesis",
                    }
                ),
                encoding="utf-8",
            )

            output = base / "history.json"
            run(
                sys.executable,
                str(HISTORY),
                "--runs-root",
                str(runs / "demo/vllm-kunlun"),
                "--item-id",
                "WI-001",
                "--output",
                str(output),
            )
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["invocation_count"], 1)
            self.assertEqual(data["observed_attempt_count"], 1)
            self.assertEqual(data["signature_counts"]["abc"], 1)
            attempt_data = data["runs"][0]["attempts"][0]
            self.assertIn("registry entry", attempt_data["attempt_excerpt"])
            self.assertIn("RuntimeError", attempt_data["stderr_tail"])
            self.assertEqual(attempt_data["command"]["argv"][-1], "test_registry.py")

    def test_scope_check_allows_current_item_and_ignores_unchanged_prior_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            head = init_repo(repo)
            (repo / "outside.txt").write_text("prior work item\n", encoding="utf-8")
            attempt_dir = base / "attempt"
            run(
                sys.executable,
                str(SNAPSHOT),
                "--target-repo",
                str(repo),
                "--run-dir",
                str(attempt_dir),
                "--phase",
                "before",
                "--base-ref",
                head,
            )
            (repo / "allowed/main.py").write_text("VALUE = 3\n", encoding="utf-8")
            scope = base / "scope.json"
            run(
                sys.executable,
                str(CHECK_SCOPE),
                "--target-repo",
                str(repo),
                "--base-ref",
                head,
                "--before-snapshot",
                str(attempt_dir / "repo-before.json"),
                "--allow",
                "allowed/**",
                "--output",
                str(scope),
            )
            data = json.loads(scope.read_text(encoding="utf-8"))
            self.assertEqual(data["changed_during_attempt"], ["allowed/main.py"])
            self.assertEqual(data["violations"], [])

    def test_scope_check_detects_deletion_of_clean_tracked_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            head = init_repo(repo)
            attempt_dir = base / "attempt"
            run(
                sys.executable,
                str(SNAPSHOT),
                "--target-repo",
                str(repo),
                "--run-dir",
                str(attempt_dir),
                "--phase",
                "before",
                "--base-ref",
                head,
            )
            (repo / "outside.txt").unlink()
            scope = base / "scope.json"
            failed = run(
                sys.executable,
                str(CHECK_SCOPE),
                "--target-repo",
                str(repo),
                "--base-ref",
                head,
                "--before-snapshot",
                str(attempt_dir / "repo-before.json"),
                "--allow",
                "allowed/**",
                "--output",
                str(scope),
                check=False,
            )
            self.assertEqual(failed.returncode, 4)
            self.assertIn("outside.txt", json.loads(scope.read_text(encoding="utf-8"))["violations"])

    def test_check_implementation_detects_files_created_by_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            head = init_repo(repo)
            attempt_dir = base / "attempt"
            run(
                sys.executable,
                str(SNAPSHOT),
                "--target-repo",
                str(repo),
                "--run-dir",
                str(attempt_dir),
                "--phase",
                "before",
                "--base-ref",
                head,
            )
            (repo / "allowed/main.py").write_text("VALUE = 2\n", encoding="utf-8")
            verification = attempt_dir / "verification"
            proc = run(
                "bash",
                str(CHECK_IMPLEMENTATION),
                "--target-repo",
                str(repo),
                "--run-dir",
                str(verification),
                "--base-ref",
                head,
                "--before-snapshot",
                str(attempt_dir / "repo-before.json"),
                "--allow",
                "allowed/**",
                "--timeout-seconds",
                "10",
                "--",
                sys.executable,
                "-c",
                "from pathlib import Path; Path('outside-generated.txt').write_text('x')",
                check=False,
            )
            self.assertEqual(proc.returncode, 4)
            summary = json.loads((verification / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "scope_violation_after_verification")
            scope = json.loads((verification / "scope-after.json").read_text(encoding="utf-8"))
            self.assertIn("outside-generated.txt", scope["violations"])

    def test_failure_signature_normalizes_path_and_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            result = base / "result.json"
            result.write_text(json.dumps({"exit_code": 1, "timed_out": False}), encoding="utf-8")
            hashes: list[str] = []
            for index, line in enumerate((123, 987), start=1):
                stderr = base / f"stderr-{index}.log"
                stderr.write_text(
                    f"Traceback (most recent call last):\n"
                    f"  File /tmp/run{index}/model.py:{line}\n"
                    "RuntimeError: shape mismatch\n",
                    encoding="utf-8",
                )
                output = base / f"signature-{index}.json"
                run(
                    sys.executable,
                    str(FAILURE_SIGNATURE),
                    "--command-result",
                    str(result),
                    "--stderr",
                    str(stderr),
                    "--output",
                    str(output),
                )
                hashes.append(json.loads(output.read_text(encoding="utf-8"))["signature_hash"])
            self.assertEqual(hashes[0], hashes[1])

    def test_run_bash_uses_requested_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            cwd = base / "cwd"
            cwd.mkdir()
            run_dir = base / "command"
            run(
                sys.executable,
                str(RUN_BASH),
                "--run-dir",
                str(run_dir),
                "--cwd",
                str(cwd),
                "--",
                sys.executable,
                "-c",
                "import os; print(os.getcwd())",
            )
            reported_cwd = Path((run_dir / "stdout.log").read_text(encoding="utf-8").strip())
            self.assertEqual(reported_cwd.resolve(), cwd.resolve())
            command = json.loads((run_dir / "command.json").read_text(encoding="utf-8"))
            self.assertEqual(Path(command["working_directory"]).resolve(), cwd.resolve())

    def test_collect_context_records_module_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "context.json"
            run(
                sys.executable,
                str(COLLECT_CONTEXT),
                "--output",
                str(output),
                "--module",
                "json",
            )
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["python"]["executable"], sys.executable)
            self.assertTrue(data["modules"][0]["found"])


if __name__ == "__main__":
    unittest.main()
