import os
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_INTEGRATION_TEST = (
    REPO_ROOT / "tests" / "test_ticket12_real_sglang_hook_integration.py"
)


class Ticket12CudaCaptureEntrypointTest(unittest.TestCase):
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
