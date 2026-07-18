import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "migration-spec.md"
VERIFY_RESULT = (
    ROOT
    / "runs"
    / "handoff-verify-p800-r5-001"
    / "result.json"
)
REVIEW_RESULT = (
    ROOT
    / "runs"
    / "p800-handoff-review-r5-001"
    / "result.json"
)
RUNBOOK = (
    ROOT
    / "model-adaptation"
    / "references"
    / "p800-baseline-replay.md"
)


class Ticket16P800BaselineRunbookTest(unittest.TestCase):
    def test_p800_manifest_evidence_is_accepted_without_tensor_loading(
        self,
    ) -> None:
        verification = json.loads(
            VERIFY_RESULT.read_text(encoding="utf-8")
        )
        review = json.loads(REVIEW_RESULT.read_text(encoding="utf-8"))

        self.assertTrue(verification["passed"])
        self.assertTrue(verification["manifest_verified"])
        self.assertEqual(verification["file_count"], 13)
        self.assertEqual(
            verification["manifest_sha256"],
            "c7886622083e8516e335df6946321858ef58dfdec8f33d268aed7140eb13a83a",
        )
        self.assertTrue(review["passed"])
        self.assertEqual(
            review["reviewed_run"],
            "runs/handoff-verify-p800-r5-001",
        )
        self.assertFalse(review["tensor_payloads_deserialized"])
        self.assertEqual(
            review["spec_binding"],
            verification["spec_binding"],
        )

    def test_current_state_enters_p800_repair_before_baseline(self) -> None:
        spec = SPEC.read_text(encoding="utf-8")

        self.assertIn("- `state_revision`: `28`", spec)
        self.assertIn("- `status`: `ACTIVE`", spec)
        self.assertIn("- `phase`: `P800_REPAIR`", spec)
        self.assertIn("- `execution_site`: `SOURCE`", spec)
        self.assertIn(
            "- `last_run`: `runs/adapter-003`",
            spec,
        )
        self.assertIn(
            "- `p800_verification_run`: "
            "`runs/handoff-verify-p800-r5-001`",
            spec,
        )
        self.assertIn(
            "- `bundle_status`: `VERIFIED_ON_P800`",
            spec,
        )
        self.assertIn("- `baseline_run`: `null`", spec)
        self.assertIn("- `attempts_used`: `0`", spec)
        self.assertIn("P800 baseline replay", spec)

    def test_runbook_seals_one_zero_attempt_baseline(self) -> None:
        runbook = RUNBOOK.read_text(encoding="utf-8")

        checkout = runbook.index(
            "origin/chore/step3p7-p800-baseline"
        )
        workspace_check = runbook.index("--mode check-baseline")
        replay = runbook.index("--mode kernel-replay")
        assessment = runbook.index("--mode assess-baseline")
        outcome = runbook.index("P800 baseline gap_observed:")
        commit = runbook.index(
            'git commit -m "evidence: add P800 baseline replay"'
        )
        push = runbook.index('git push -u origin "$EVIDENCE_BRANCH"')

        self.assertLess(checkout, workspace_check)
        self.assertLess(workspace_check, replay)
        self.assertLess(replay, assessment)
        self.assertLess(assessment, outcome)
        self.assertLess(outcome, commit)
        self.assertLess(commit, push)
        self.assertIn(
            "export EVIDENCE_BRANCH=evidence/p800-baseline-r5-001",
            runbook,
        )
        self.assertIn(
            "export CHECK_RUN=runs/p800-baseline-check-r5-001",
            runbook,
        )
        self.assertIn(
            "export REPLAY_RUN=runs/p800-baseline-replay-r5-001",
            runbook,
        )
        self.assertIn(
            "export ASSESS_RUN=runs/p800-baseline-assessment-r5-001",
            runbook,
        )
        self.assertIn(
            "runs/handoff-build-r5-001/bundle/runs/"
            "cuda-golden-r5-001",
            runbook,
        )
        self.assertIn(
            "sglang-kunlun/sglang_kunlun/hooks/layers/"
            "quantization/unquant.py",
            runbook,
        )
        self.assertIn(
            "546ad8c682392922792bbbfe53a8bf575545f118",
            runbook,
        )
        self.assertIn("runs/adapter-003/result.json", runbook)
        self.assertIn("--execution-site p800", runbook)
        self.assertIn('assessment["attempts_used"] == 0', runbook)
        self.assertIn(
            'assessment["gap_observed"] is (not replay["passed"])',
            runbook,
        )
        self.assertIn(
            'allowed_failure_types = {',
            runbook,
        )
        self.assertIn(
            'error["type"] in allowed_failure_types',
            runbook,
        )
        self.assertIn("worker-result.json.errors", runbook)
        self.assertNotIn("--sglang-worktree", runbook)
        self.assertNotIn("sglang.launch_server", runbook)
        self.assertNotIn("torch.save", runbook)
        self.assertNotIn("git add migration-spec.md", runbook)


if __name__ == "__main__":
    unittest.main()
