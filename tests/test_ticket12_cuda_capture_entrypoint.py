import os
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_INTEGRATION_TEST = (
    REPO_ROOT / "tests" / "test_ticket12_real_sglang_hook_integration.py"
)
RUNBOOK = (
    REPO_ROOT
    / "model-adaptation"
    / "references"
    / "cuda-capture-validation.md"
)


class Ticket12CudaCaptureEntrypointTest(unittest.TestCase):
    def test_runbook_uses_target_only_eager_launch_arguments(self) -> None:
        runbook = RUNBOOK.read_text(encoding="utf-8")

        self.assertIn("--scan-result runs/scan-004/result.json", runbook)
        self.assertIn("--operator-id Step3p5MLP.forward", runbook)
        self.assertIn(
            "49e384ce9d304648e9959666ecb8ce8cd98d0deb",
            runbook,
        )
        self.assertIn("--cuda-graph-backend-decode disabled", runbook)
        self.assertIn("--cuda-graph-backend-prefill disabled", runbook)
        self.assertIn('--revision "$MODEL_REVISION"', runbook)
        self.assertIn(
            "5f6244077ac62e04eec3f320501ff8c2b293373a",
            runbook,
        )
        self.assertNotIn("--speculative-algorithm EAGLE", runbook)
        self.assertNotIn("step_swiglu_with_limit", runbook)

    def test_runbook_command_rejects_empty_sglang_worktree(self) -> None:
        environment = os.environ.copy()
        environment["SGLANG_WORKTREE"] = ""
        completed = subprocess.run(
            [sys.executable, str(HOOK_INTEGRATION_TEST), "-v"],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("SGLANG_WORKTREE must be set", completed.stderr)


if __name__ == "__main__":
    unittest.main()
