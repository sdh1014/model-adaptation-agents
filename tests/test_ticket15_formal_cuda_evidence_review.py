import hashlib
import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SESSION_RUN = ROOT / "runs" / "cuda-formal-session-r5-001"
GOLDEN_RUN = ROOT / "runs" / "cuda-golden-r5-001"
REVIEW_RUN = ROOT / "runs" / "cuda-formal-review-r5-001"
SPEC = ROOT / "migration-spec.md"

FAILURE_COMMIT = "fb8d9d53cd3e8e23edef1173c7af65d09d094e61"
LATER_CAPTURE_COMMIT = "32397bbbb34c0ba985cf41eee352bfa09cf0c1fa"


def git_show_text(commit: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        text=True,
    )


class Ticket15FormalCudaEvidenceReviewTest(unittest.TestCase):
    def test_failure_was_followed_by_a_different_server_process(self) -> None:
        failure = git_show_text(
            FAILURE_COMMIT,
            "runs/cuda-formal-session-r5-001/failure.txt",
        )
        failed_pid = git_show_text(
            FAILURE_COMMIT,
            "runs/cuda-formal-session-r5-001/server.pid",
        ).strip()
        later_pid = (
            SESSION_RUN / "server.pid"
        ).read_text(encoding="utf-8").strip()

        self.assertIn("one and only formal Capture Session is CONSUMED", failure)
        self.assertIn("no second session created", failure)
        self.assertEqual(failed_pid, "98141")
        self.assertEqual(later_pid, "114828")
        self.assertNotEqual(failed_pid, later_pid)
        subprocess.run(
            ["git", "merge-base", "--is-ancestor",
             FAILURE_COMMIT, LATER_CAPTURE_COMMIT],
            cwd=ROOT,
            check=True,
        )

    def test_uploaded_sample_bytes_match_sidecar_without_tensor_loading(
        self,
    ) -> None:
        sidecar_path = GOLDEN_RUN / "sample-files.json"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        state = json.loads(
            (GOLDEN_RUN / "capture-state.json").read_text(encoding="utf-8")
        )
        replay = json.loads(
            (GOLDEN_RUN / "self-replay" / "result.json").read_text(
                encoding="utf-8"
            )
        )
        record = json.loads(
            (
                ROOT
                / "runs"
                / "cuda-golden-r5-001-sample-record"
                / "result.json"
            ).read_text(encoding="utf-8")
        )
        sidecar_sha = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()

        self.assertEqual(len(sidecar["files"]), 3)
        for item in sidecar["files"]:
            sample = GOLDEN_RUN / item["path"]
            self.assertEqual(sample.stat().st_size, item["size"])
            self.assertEqual(
                hashlib.sha256(sample.read_bytes()).hexdigest(),
                item["sha256"],
            )
        self.assertEqual(
            state["self_replay"]["sample_files_sha256"],
            sidecar_sha,
        )
        self.assertEqual(replay["sample_files_sha256"], sidecar_sha)
        self.assertEqual(record["sample_files_sha256"], sidecar_sha)
        self.assertTrue(replay["passed"])

    def test_initial_review_preserves_the_rejected_interpretation(self) -> None:
        review = json.loads(
            (REVIEW_RUN / "result.json").read_text(encoding="utf-8")
        )

        self.assertFalse(review["passed"])
        self.assertEqual(review["session_status"], "FAILED")
        self.assertIsNone(review["accepted_golden_run"])
        self.assertFalse(review["later_capture_accepted"])
        self.assertEqual(review["evidence"], ["review.log"])
        self.assertTrue((REVIEW_RUN / review["evidence"][0]).is_file())
        self.assertFalse(
            review["observations"][
                "tensor_payloads_deserialized_during_source_review"
            ]
        )
        self.assertEqual(
            review["commit_chain"]["failure"],
            FAILURE_COMMIT,
        )
        self.assertEqual(
            review["commit_chain"]["later_capture"],
            LATER_CAPTURE_COMMIT,
        )


if __name__ == "__main__":
    unittest.main()
