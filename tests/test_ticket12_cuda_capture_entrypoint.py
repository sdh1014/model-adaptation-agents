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
    def test_runbook_blocks_old_mlp_commands_until_kernel_adapter_exists(self) -> None:
        runbook = RUNBOOK.read_text(encoding="utf-8")

        self.assertIn("revision 5", runbook)
        self.assertIn("scan-006", runbook)
        self.assertIn("_swiglu_silu_clamp_mul", runbook)
        self.assertIn(
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe."
            "_swiglu_silu_clamp_mul",
            runbook,
        )
        self.assertIn("capture/replay adapter 尚未实现", runbook)
        self.assertIn("不能执行 CUDA preflight", runbook)
        self.assertIn("完整 checkpoint", runbook)
        self.assertIn("module `state_dict`", runbook)
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
