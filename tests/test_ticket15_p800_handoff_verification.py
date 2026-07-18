from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUNDLED_SPEC = (
    ROOT
    / "runs"
    / "handoff-build-r5-001"
    / "bundle"
    / "migration-spec.md"
)
RUNBOOK = (
    ROOT
    / "model-adaptation"
    / "references"
    / "p800-handoff-verification.md"
)


class Ticket15P800HandoffVerificationTest(unittest.TestCase):
    def test_bundled_spec_waits_only_for_p800_manifest_verification(
        self,
    ) -> None:
        spec = BUNDLED_SPEC.read_text(encoding="utf-8")

        self.assertIn("- `state_revision`: `26`", spec)
        self.assertIn("- `status`: `WAITING`", spec)
        self.assertIn("- `phase`: `HANDOFF`", spec)
        self.assertIn("- `execution_site`: `CUDA`", spec)
        self.assertIn("- `bundle_status`: `VALID`", spec)
        self.assertIn("- `p800_verification_run`: `null`", spec)
        self.assertIn("人工复制", spec)
        self.assertIn("manifest verify", spec)

    def test_runbook_only_verifies_and_uploads_p800_evidence(self) -> None:
        runbook = RUNBOOK.read_text(encoding="utf-8")

        checkout = runbook.index(
            "origin/chore/step3p7-p800-baseline"
        )
        verify = runbook.index("--mode verify")
        assert_result = runbook.index('result["manifest_verified"]')
        commit = runbook.index(
            'git commit -m "evidence: verify handoff bundle on P800"'
        )
        push = runbook.index('git push -u origin "$EVIDENCE_BRANCH"')

        self.assertLess(checkout, verify)
        self.assertLess(verify, assert_result)
        self.assertLess(assert_result, commit)
        self.assertLess(commit, push)
        self.assertIn(
            "export VERIFY_RUN=runs/handoff-verify-p800-r5-001",
            runbook,
        )
        self.assertIn(
            "export EVIDENCE_BRANCH=evidence/p800-handoff-verify-r5-001",
            runbook,
        )
        self.assertIn(
            "export BUNDLE_DIR=runs/handoff-build-r5-001/bundle",
            runbook,
        )
        self.assertIn("--spec migration-spec.md", runbook)
        self.assertIn('--run-dir "$VERIFY_RUN"', runbook)
        self.assertIn('--bundle-dir "$BUNDLE_DIR"', runbook)
        self.assertIn(
            '"c7886622083e8516e335df6946321858ef58dfdec8f33d268aed7140eb13a83a"',
            runbook,
        )
        self.assertIn('result["file_count"] == 13', runbook)
        self.assertNotIn("torch.load", runbook)
        self.assertNotIn("replay_compare.py", runbook)
        self.assertNotIn("workspace_guard.py", runbook)
        self.assertNotIn("sglang.launch_server", runbook)
        self.assertNotIn("git add migration-spec.md", runbook)


if __name__ == "__main__":
    unittest.main()
