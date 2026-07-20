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
    def test_runbook_runs_only_revision_five_kernel_preflight(self) -> None:
        runbook = RUNBOOK.read_text(encoding="utf-8")

        self.assertIn("revision 5", runbook)
        self.assertIn("scan-006", runbook)
        self.assertIn("_swiglu_silu_clamp_mul", runbook)
        self.assertIn(
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe."
            "_swiglu_silu_clamp_mul",
            runbook,
        )
        self.assertIn("Ticket 23 已实现对应 adapter", runbook)
        self.assertIn("--mode capture-adapter-validation", runbook)
        self.assertIn("不加载 checkpoint", runbook)
        self.assertIn("不消耗唯一", runbook)
        self.assertIn("不要继续正式模型 Capture", runbook)
        self.assertNotIn("--scan-result runs/scan-004/result.json", runbook)
        self.assertNotIn("--operator-id Step3p5MLP.forward", runbook)
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
