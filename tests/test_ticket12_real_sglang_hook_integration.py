import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "model-adaptation" / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from model_adaptation_capture import plugin


FIXED_SGLANG_REVISION = "6274831d9fef7bba04eb59302caac24563a974c9"
DEFAULT_SGLANG_WORKTREE = Path("/tmp/sglang-step3p7-cuda")
MODULE_NAMES = (
    "sglang",
    "sglang.srt",
    "sglang.srt.models",
    "sglang.srt.plugins",
    "sglang.srt.models.step3p5_ops",
    "sglang.srt.models.step3p5",
    "sglang.srt.plugins.hook_registry",
)
MISSING = object()


def install_package(name: str, path: Path) -> None:
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_capture_config(path: Path, run_dir: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "golden-capture-config/v0",
                "spec_binding": {
                    "spec_id": "step3p7-flash-p800-demo",
                    "contract_revision": 2,
                    "contract_data_sha256": "fixture-sha256",
                },
                "operator_id": "activation.step_swiglu_with_limit",
                "model_path": "target",
                "max_samples": 3,
                "serialization": "candidate-torch-save/v1",
                "run_dir": str(run_dir),
                "capture_device_type": "cpu",
                "dtype": "bfloat16",
            }
        )
        + "\n",
        encoding="utf-8",
    )


class Ticket12RealSglangHookIntegrationTest(unittest.TestCase):
    def test_real_registry_captures_only_three_target_signatures(self) -> None:
        worktree_was_configured = "SGLANG_WORKTREE" in os.environ
        configured_worktree = os.environ.get("SGLANG_WORKTREE")
        if worktree_was_configured and not configured_worktree:
            self.fail("SGLANG_WORKTREE must be set to a non-empty path")
        worktree = Path(
            configured_worktree or DEFAULT_SGLANG_WORKTREE
        ).resolve()
        helper_path = worktree / "python/sglang/srt/models/step3p5_ops.py"
        registry_path = worktree / "python/sglang/srt/plugins/hook_registry.py"
        if not helper_path.is_file() or not registry_path.is_file():
            if worktree_was_configured:
                self.fail(
                    "SGLANG_WORKTREE does not contain the required fixed "
                    f"SGLang sources: {worktree}"
                )
            self.skipTest(
                "fixed SGLang checkout unavailable; set SGLANG_WORKTREE to run"
            )

        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=worktree,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(revision, FIXED_SGLANG_REVISION)

        saved_modules = {
            name: sys.modules.get(name, MISSING) for name in MODULE_NAMES
        }
        python_root = worktree / "python"
        install_package("sglang", python_root / "sglang")
        install_package("sglang.srt", python_root / "sglang/srt")
        install_package("sglang.srt.models", python_root / "sglang/srt/models")
        install_package("sglang.srt.plugins", python_root / "sglang/srt/plugins")
        ops = load_module(
            "sglang.srt.models.step3p5_ops",
            helper_path,
        )
        hooks = load_module(
            "sglang.srt.plugins.hook_registry",
            registry_path,
        )

        step3p5 = types.ModuleType("sglang.srt.models.step3p5")
        step3p5.step_swiglu_with_limit = ops.step_swiglu_with_limit

        class Step3p5ForCausalLM:
            def forward(self, input_ids, positions, forward_batch):
                del positions, forward_batch
                return step3p5.step_swiglu_with_limit(input_ids, 16.0)

        step3p5.Step3p5ForCausalLM = Step3p5ForCausalLM
        sys.modules[step3p5.__name__] = step3p5
        original_helper = ops.step_swiglu_with_limit
        original_forward = Step3p5ForCausalLM.forward

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                run_dir = Path(temp_dir) / "cuda-preflight-fixture"
                run_dir.mkdir()
                config_path = run_dir / "capture-config.json"
                write_capture_config(config_path, run_dir)
                plugin._reset_for_tests()

                with patch.dict(
                    os.environ,
                    {"MODEL_ADAPTATION_CAPTURE_CONFIG": str(config_path)},
                    clear=False,
                ):
                    plugin.register()
                    hooks.HookRegistry.apply_hooks()

                self.assertEqual(
                    hooks.HookRegistry._patched,
                    {
                        "sglang.srt.models.step3p5.Step3p5ForCausalLM.forward",
                        "sglang.srt.models.step3p5_ops.step_swiglu_with_limit",
                    },
                )
                self.assertIsNot(ops.step_swiglu_with_limit, original_helper)
                self.assertIs(
                    step3p5.step_swiglu_with_limit,
                    ops.step_swiglu_with_limit,
                )

                forward_batch = types.SimpleNamespace(
                    forward_mode=types.SimpleNamespace(name="DECODE")
                )
                model = Step3p5ForCausalLM()
                for rows in (1, 2, 4):
                    gate_up = torch.arange(
                        rows * 8,
                        dtype=torch.bfloat16,
                    ).reshape(rows, 8)
                    actual = model.forward(gate_up, None, forward_batch)
                    expected = original_helper(gate_up, 16.0)
                    torch.testing.assert_close(actual, expected)

                model.forward(
                    torch.full((2, 8), 2.0, dtype=torch.bfloat16),
                    None,
                    forward_batch,
                )
                model.forward(
                    torch.ones((8, 8), dtype=torch.bfloat16),
                    None,
                    forward_batch,
                )

                state = json.loads(
                    (run_dir / "capture-state.json").read_text(encoding="utf-8")
                )
                self.assertEqual(state["saved_sample_count"], 3)
                self.assertEqual(state["repeated_call_count"], 1)
                self.assertEqual(state["skipped_call_count"], 1)
                self.assertEqual(len(state["skipped_signatures"]), 1)
                self.assertTrue(
                    all(
                        sample["signature"]["execution_phase"] == "decode"
                        for sample in state["samples"]
                    )
                )
        finally:
            ops.step_swiglu_with_limit = original_helper
            step3p5.step_swiglu_with_limit = original_helper
            Step3p5ForCausalLM.forward = original_forward
            hooks.HookRegistry.reset()
            plugin._reset_for_tests()
            for name in reversed(MODULE_NAMES):
                previous = saved_modules[name]
                if previous is MISSING:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous


if __name__ == "__main__":
    unittest.main()
