import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "model-adaptation" / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from model_adaptation_capture import plugin


class FakeHookType:
    AROUND = "around"


class FakeHookRegistry:
    calls = []

    @classmethod
    def register(cls, target, hook, hook_type):
        cls.calls.append((target, hook, hook_type))


def fake_sglang_modules():
    sglang = types.ModuleType("sglang")
    sglang.__path__ = []
    srt = types.ModuleType("sglang.srt")
    srt.__path__ = []
    plugins = types.ModuleType("sglang.srt.plugins")
    plugins.__path__ = []
    hook_registry = types.ModuleType("sglang.srt.plugins.hook_registry")
    hook_registry.HookRegistry = FakeHookRegistry
    hook_registry.HookType = FakeHookType
    return {
        "sglang": sglang,
        "sglang.srt": srt,
        "sglang.srt.plugins": plugins,
        "sglang.srt.plugins.hook_registry": hook_registry,
    }


class Ticket12SglangCapturePluginTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeHookRegistry.calls = []
        plugin._reset_for_tests()

    def test_registers_target_only_context_and_swiglu_hooks_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "capture-config.json"
            config_path.write_text(json.dumps({}) + "\n", encoding="utf-8")
            with patch.dict(sys.modules, fake_sglang_modules()), patch.dict(
                os.environ,
                {"MODEL_ADAPTATION_CAPTURE_CONFIG": str(config_path)},
                clear=False,
            ):
                plugin.register()

        self.assertEqual(
            [call[0] for call in FakeHookRegistry.calls],
            [
                "sglang.srt.models.step3p5.Step3p5ForCausalLM.forward",
                "sglang.srt.models.step3p5_ops.step_swiglu_with_limit",
            ],
        )
        self.assertTrue(all(call[2] == FakeHookType.AROUND for call in FakeHookRegistry.calls))

    def test_does_not_patch_sglang_when_capture_is_not_requested(self) -> None:
        with patch.dict(sys.modules, fake_sglang_modules()), patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            plugin.register()

        self.assertEqual(FakeHookRegistry.calls, [])


if __name__ == "__main__":
    unittest.main()
