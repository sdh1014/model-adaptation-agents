import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "runs" / "cuda-formal-review-r5-002" / "result.json"
HANDOFF_SPEC = (
    ROOT
    / "runs"
    / "handoff-plan-r5-001"
    / "migration-spec.md"
)
HANDOFF_RESULT = (
    ROOT
    / "runs"
    / "handoff-plan-r5-001"
    / "result.json"
)
RUNBOOK = (
    ROOT
    / "model-adaptation"
    / "references"
    / "formal-cuda-handoff.md"
)
def contract_data(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    begin = "<!-- CONTRACT-DATA: BEGIN -->"
    end = "<!-- CONTRACT-DATA: END -->"
    return json.loads(text.split(begin, 1)[1].split(end, 1)[0])


class Ticket15HandoffPlanTest(unittest.TestCase):
    def test_human_clarification_accepts_the_only_actual_capture(self) -> None:
        review = json.loads(REVIEW.read_text(encoding="utf-8"))

        self.assertTrue(review["passed"])
        self.assertEqual(
            review["supersedes"],
            "runs/cuda-formal-review-r5-001",
        )
        self.assertFalse(
            review["counting_rule"]["model_load_failure_counts_as_session"]
        )
        self.assertTrue(
            review["counting_rule"]["sample_producing_run_counts_as_session"]
        )
        self.assertEqual(
            review["accepted_golden_run"],
            "runs/cuda-golden-r5-001",
        )
        self.assertEqual(review["session_status"], "SEALED")
        self.assertEqual(review["spec_binding"]["contract_revision"], 5)

    def test_candidate_spec_is_the_waiting_handoff_state(self) -> None:
        candidate = HANDOFF_SPEC.read_text(encoding="utf-8")
        result = json.loads(HANDOFF_RESULT.read_text(encoding="utf-8"))

        self.assertTrue(result["passed"])
        self.assertEqual(
            result["candidate_spec"],
            "runs/handoff-plan-r5-001/migration-spec.md",
        )
        self.assertEqual(
            result["candidate_state"],
            {
                "bundle_status": "VALID",
                "execution_site": "CUDA",
                "phase": "HANDOFF",
                "state_revision": 26,
                "status": "WAITING",
            },
        )
        self.assertEqual(contract_data(HANDOFF_SPEC)["contract_revision"], 5)
        self.assertIn("- `state_revision`: `26`", candidate)
        self.assertIn("- `status`: `WAITING`", candidate)
        self.assertIn("- `phase`: `HANDOFF`", candidate)
        self.assertIn("- `execution_site`: `CUDA`", candidate)
        self.assertIn("- `session_status`: `SEALED`", candidate)
        self.assertIn(
            "- `golden_run`: `runs/cuda-golden-r5-001`",
            candidate,
        )
        self.assertIn("- `bundle_status`: `VALID`", candidate)
        self.assertIn(
            "- `bundle_path`: `runs/handoff-build-r5-001/bundle`",
            candidate,
        )
        self.assertIn(
            "- `manifest_path`: "
            "`runs/handoff-build-r5-001/bundle/manifest.json`",
            candidate,
        )
        self.assertIn("人工复制", candidate)

    def test_runbook_builds_and_verifies_without_restarting_model(self) -> None:
        runbook = RUNBOOK.read_text(encoding="utf-8")

        build = runbook.index("--mode build")
        verify = runbook.index("--mode verify")
        replace = runbook.index("migration-spec.md.next")
        upload = runbook.index(
            'git commit -m "evidence: add verified CUDA handoff bundle"'
        )
        self.assertLess(build, verify)
        self.assertLess(verify, replace)
        self.assertLess(replace, upload)
        self.assertIn(
            "--sample-record-result \"$SAMPLE_RECORD_RESULT\"",
            runbook,
        )
        self.assertIn(
            "runs/handoff-plan-r5-001/migration-spec.md",
            runbook,
        )
        self.assertIn('Path(record["golden_run"])', runbook)
        self.assertIn('Path(record["sample_files_path"])', runbook)
        self.assertIn(".resolve()", runbook)
        self.assertNotIn("sglang.launch_server", runbook)
        self.assertNotIn("--mode prepare", runbook)
        self.assertNotIn("--mode kernel-replay", runbook)


if __name__ == "__main__":
    unittest.main()
