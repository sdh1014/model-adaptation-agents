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

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "model-adaptation" / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from model_adaptation_capture import plugin


FIXED_SGLANG_REVISION = "49e384ce9d304648e9959666ecb8ce8cd98d0deb"
DEFAULT_SGLANG_WORKTREE = Path("/tmp/sglang-step3p7-cuda-original")
MODULE_NAMES = (
    "sglang",
    "sglang.srt",
    "sglang.srt.models",
    "sglang.srt.plugins",
    "sglang.srt.distributed",
    "sglang.srt.distributed.parallel_state",
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
                "schema": "golden-capture-config/v1",
                "spec_binding": {
                    "spec_id": "step3p7-flash-p800-demo",
                    "contract_revision": 4,
                    "contract_data_sha256": "fixture-sha256",
                },
                "operator_id": "Step3p5MLP.forward",
                "activation_guard": "self.limit is not None",
                "model_path": "target",
                "max_shapes": 3,
                "tensor_parallel_size": 8,
                "tp_rank": 0,
                "serialization": "candidate-torch-save/v2",
                "run_dir": str(run_dir),
                "capture_device_type": "cpu",
                "dtype": "bfloat16",
                "checkpoint": {
                    "id": "stepfun-ai/Step-3.7-Flash@fixture",
                    "model_path": "stepfun-ai/Step-3.7-Flash",
                    "revision": "fixture",
                    "config_digest": "config-digest",
                },
                "preflight_tp_context": {"rank": 0, "size": 8},
            }
        )
        + "\n",
        encoding="utf-8",
    )


class Ticket12RealSglangHookIntegrationTest(unittest.TestCase):
    def test_real_registry_captures_limited_mlp_without_source_helper(self) -> None:
        worktree_was_configured = "SGLANG_WORKTREE" in os.environ
        configured_worktree = os.environ.get("SGLANG_WORKTREE")
        if worktree_was_configured and not configured_worktree:
            self.fail("SGLANG_WORKTREE must be set to a non-empty path")
        import torch

        worktree = Path(configured_worktree or DEFAULT_SGLANG_WORKTREE).resolve()
        model_path = worktree / "python/sglang/srt/models/step3p5.py"
        registry_path = worktree / "python/sglang/srt/plugins/hook_registry.py"
        helper_path = worktree / "python/sglang/srt/models/step3p5_ops.py"
        if not model_path.is_file() or not registry_path.is_file():
            if worktree_was_configured:
                self.fail(
                    "SGLANG_WORKTREE does not contain the required original "
                    f"SGLang sources: {worktree}"
                )
            self.skipTest(
                "original SGLang checkout unavailable; set SGLANG_WORKTREE to run"
            )

        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=worktree,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(revision, FIXED_SGLANG_REVISION)
        self.assertFalse(helper_path.exists())
        model_source = model_path.read_text(encoding="utf-8")
        self.assertIn("class Step3p5MLP(nn.Module):", model_source)
        self.assertIn("if self.limit is not None:", model_source)
        self.assertIn("output, _ = self.down_proj(gate * up)", model_source)

        saved_modules = {
            name: sys.modules.get(name, MISSING) for name in MODULE_NAMES
        }
        python_root = worktree / "python"
        install_package("sglang", python_root / "sglang")
        install_package("sglang.srt", python_root / "sglang/srt")
        install_package("sglang.srt.models", python_root / "sglang/srt/models")
        install_package("sglang.srt.plugins", python_root / "sglang/srt/plugins")
        install_package(
            "sglang.srt.distributed",
            python_root / "sglang/srt/distributed",
        )
        hooks = load_module(
            "sglang.srt.plugins.hook_registry",
            registry_path,
        )

        parallel_state = types.ModuleType(
            "sglang.srt.distributed.parallel_state"
        )
        parallel_state.get_tp_group = lambda: types.SimpleNamespace(
            rank_in_group=0,
            world_size=8,
        )
        sys.modules[parallel_state.__name__] = parallel_state

        step3p5 = types.ModuleType("sglang.srt.models.step3p5")

        class Step3p5MLP:
            def __init__(self, *, limit, prefix):
                self.limit = limit
                self.gate_up_proj = types.SimpleNamespace(
                    prefix=f"{prefix}.gate_up_proj"
                )

            def forward(self, x):
                return x * 2

        class Step3p5ForCausalLM:
            def __init__(self):
                self.limited = Step3p5MLP(
                    limit=16.0,
                    prefix="model.layers.43.share_expert",
                )
                self.plain = Step3p5MLP(
                    limit=None,
                    prefix="model.layers.42.share_expert",
                )

            def forward(self, input_ids, positions, forward_batch):
                del positions, forward_batch
                return (
                    self.limited.forward(input_ids),
                    self.plain.forward(input_ids),
                )

        step3p5.Step3p5MLP = Step3p5MLP
        step3p5.Step3p5ForCausalLM = Step3p5ForCausalLM
        sys.modules[step3p5.__name__] = step3p5
        original_mlp_forward = Step3p5MLP.forward
        original_model_forward = Step3p5ForCausalLM.forward

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
                        "sglang.srt.models.step3p5.Step3p5MLP.forward",
                    },
                )
                self.assertIsNot(Step3p5MLP.forward, original_mlp_forward)

                forward_batch = types.SimpleNamespace(
                    forward_mode=types.SimpleNamespace(name="DECODE")
                )
                model = Step3p5ForCausalLM()
                for rows in (1, 2, 4):
                    x = torch.arange(
                        rows * 8,
                        dtype=torch.bfloat16,
                    ).reshape(rows, 8)
                    limited, plain = model.forward(x, None, forward_batch)
                    torch.testing.assert_close(limited, x * 2)
                    torch.testing.assert_close(plain, x * 2)

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

                state_path = run_dir / "capture-state.json"
                state = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual(state["saved_shape_count"], 3)
                self.assertEqual(state["repeated_call_count"], 1)
                self.assertEqual(state["skipped_call_count"], 1)
                self.assertTrue(
                    all(
                        sample["signature"]["execution_phase"] == "decode"
                        for sample in state["samples"]
                    )
                )
                self.assertTrue(
                    all(
                        sample["signature"]["model_instance_path"]
                        == "model.layers.43.share_expert"
                        for sample in state["samples"]
                    )
                )
                self.assertNotIn("model.layers.42", repr(state))
        finally:
            Step3p5MLP.forward = original_mlp_forward
            Step3p5ForCausalLM.forward = original_model_forward
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
