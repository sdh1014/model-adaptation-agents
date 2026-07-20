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
BASELINE_REVIEW_RESULT = (
    ROOT
    / "runs"
    / "p800-baseline-review-r5-001"
    / "result.json"
)
RUNBOOK = (
    ROOT
    / "model-adaptation"
    / "references"
    / "p800-baseline-replay.md"
)
CLAUDE_RUNBOOK = (
    ROOT
    / "model-adaptation"
    / "references"
    / "claude-p800-repair.md"
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

    def test_p800_baseline_is_accepted_as_an_operator_gap(self) -> None:
        review = json.loads(
            BASELINE_REVIEW_RESULT.read_text(encoding="utf-8")
        )

        self.assertTrue(review["passed"])
        self.assertTrue(review["gap_observed"])
        self.assertEqual(review["attempts_used"], 0)
        self.assertEqual(review["replay"]["checked_shape_count"], 3)
        self.assertEqual(review["replay"]["failed_shape_count"], 2)
        self.assertEqual(
            review["replay"]["error_types"],
            ["AssertionError", "AssertionError"],
        )
        self.assertFalse(review["replay"]["actual_tensors_saved"])
        self.assertFalse(review["tensor_payloads_deserialized"])

    def test_current_state_reopens_under_revision_6_queue_contract(self) -> None:
        spec = SPEC.read_text(encoding="utf-8")

        self.assertIn("- `state_revision`: `48`", spec)
        self.assertIn("- `status`: `ACTIVE`", spec)
        self.assertIn("- `phase`: `CUDA_CAPTURE`", spec)
        self.assertIn("- `execution_site`: `CUDA`", spec)
        self.assertIn(
            "- `last_run`: `runs/formal-cuda-runbook-r6-001`",
            spec,
        )
        self.assertIn(
            "- `active_operator`: "
            "`sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe."
            "_swiglu_silu_clamp_mul`",
            spec,
        )
        self.assertIn(
            "- `scan_run`: `runs/scan-007`",
            spec,
        )
        self.assertIn(
            "- `bundle_status`: `NOT_BUILT`",
            spec,
        )
        self.assertIn(
            "- `baseline_run`: `null`",
            spec,
        )
        self.assertIn("- `attempts_used`: `0`", spec)
        self.assertIn("- `passing_run`: `null`", spec)
        self.assertIn("- `active_hypothesis`: `null`", spec)
        self.assertIn("旧 revision 5 Scan 与 Golden 只作历史线索", spec)
        self.assertIn("任意 callable replay 协议", spec)
        self.assertIn("不符合 Kernel Call Contract", spec)

    def test_claude_runbook_only_launches_the_migration_agent(self) -> None:
        runbook = CLAUDE_RUNBOOK.read_text(encoding="utf-8")

        self.assertIn("SGLANG_KUNLUN_WORKTREE", runbook)
        self.assertIn("/model-adaptation Step-3.7-Flash", runbook)
        self.assertIn("当前 `migration-spec.md`", runbook)
        self.assertIn("不提供 attempt 1 假设或补丁", runbook)
        self.assertIn("自动进入 gap queue 下一项", runbook)
        self.assertIn("不新增生产 helper", runbook)
        self.assertNotIn("git apply", runbook)
        self.assertNotIn("torch.clamp", runbook)
        self.assertNotIn("gemm1_limit =", runbook)

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
        self.assertIn(
            '--sglang-worktree "$SGLANG_KUNLUN_WORKTREE"',
            runbook,
        )
        self.assertNotIn("sglang.launch_server", runbook)
        self.assertNotIn("torch.save", runbook)
        self.assertNotIn("git add migration-spec.md", runbook)


if __name__ == "__main__":
    unittest.main()
