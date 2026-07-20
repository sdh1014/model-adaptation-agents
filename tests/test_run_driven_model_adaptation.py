import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "model-adaptation" / "SKILL.md"
CONTEXT = ROOT / "CONTEXT.md"
SPEC = ROOT / "migration-spec.md"
TEMPLATE = ROOT / "model-adaptation" / "references" / "migration-spec-template.md"
ENVIRONMENT = ROOT / "docs" / "p800-environment-and-repair.md"

KNOWN_GAPS = (
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe."
    "_swiglu_silu_clamp_mul",
    "sgl_kernel.gemma_rmsnorm",
    "sgl_kernel.gemma_fused_add_rmsnorm",
    "sgl_kernel.topk_sigmoid",
    "sglang.srt.layers.attention.triton_ops.prefill_attention._fwd_kernel",
)

FAILURE_CATEGORIES = (
    "ENVIRONMENT",
    "ADAPTATION",
    "OPERATOR_MISSING",
    "OPERATOR_CONTRACT",
    "DISTRIBUTED_RUNTIME",
    "ACCURACY",
)

REMOVED_ENTRYPOINTS = (
    "capture_golden.py",
    "replay_compare.py",
    "handoff_bundle.py",
    "workspace_guard.py",
    "model_adaptation_capture",
)

REMOVED_PHASES = ("CUDA_CAPTURE", "HANDOFF", "P800_REPAIR")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _contract_data(spec_text: str) -> dict:
    match = re.search(
        r"<!-- CONTRACT-DATA: BEGIN -->\s*(\{.*?\})\s*"
        r"<!-- CONTRACT-DATA: END -->",
        spec_text,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError("Migration Spec must expose one Contract Data JSON block")
    return json.loads(match.group(1))


class RunDrivenModelAdaptationTest(unittest.TestCase):
    def test_public_skill_exposes_the_run_driven_workflow(self):
        skill = _read(SKILL)

        for phase in (
            "PREFLIGHT",
            "OPERATOR_VERIFICATION",
            "EAGER_BRINGUP",
            "MODEL_ACCURACY",
            "ACCURACY_DEBUG",
        ):
            self.assertIn(phase, skill)

        for category in FAILURE_CATEGORIES:
            self.assertIn(category, skill)

        for gap in KNOWN_GAPS:
            self.assertIn(gap, skill)

        self.assertIn("torch.testing.assert_close", skill)
        self.assertIn("生产调用", skill)
        self.assertIn("CPU reference", skill)
        self.assertIn("dumper", skill)
        self.assertIn("comparator", skill)

    def test_current_spec_queues_every_known_gap_under_fixed_precision(self):
        spec = _read(SPEC)
        contract = _contract_data(spec)

        self.assertEqual(contract["schema"], "model-adaptation/v1")
        self.assertEqual(contract["contract_revision"], 7)
        self.assertEqual(
            contract["precision"]["operator"],
            {
                "comparator": "torch.testing.assert_close",
                "atol": 0.01,
                "rtol": 0.02,
                "require_exact_structure": True,
                "require_declared_dtype": True,
                "require_finite": True,
                "integer_exact": True,
            },
        )

        for gap in KNOWN_GAPS:
            self.assertRegex(
                spec,
                rf"\|\s*`{re.escape(gap)}`\s*\|\s*`PENDING`\s*\|",
            )

        self.assertIn("- `phase`: `PREFLIGHT`", spec)
        self.assertIn("- `model_status`: `NOT_STARTED`", spec)
        self.assertIn("- `accuracy_status`: `NOT_STARTED`", spec)

    def test_old_replay_module_and_live_references_are_removed(self):
        self.assertFalse((ROOT / "model-adaptation" / "scripts").exists())
        self.assertFalse((ROOT / "model-adaptation" / "pyproject.toml").exists())

        for path in (
            ROOT
            / "model-adaptation"
            / "references"
            / "cuda-capture-validation.md",
            ROOT / "model-adaptation" / "references" / "formal-cuda-capture-r6.md",
            ROOT / "model-adaptation" / "references" / "formal-cuda-capture.md",
            ROOT / "model-adaptation" / "references" / "formal-cuda-handoff.md",
            ROOT / "model-adaptation" / "references" / "p800-baseline-replay.md",
            ROOT
            / "model-adaptation"
            / "references"
            / "p800-handoff-verification.md",
            ROOT / "model-adaptation" / "references" / "claude-p800-repair.md",
            ROOT / "outputs" / "step3p7-p800-migration-tool-design.md",
        ):
            self.assertFalse(path.exists(), str(path))

        for path in (SKILL, CONTEXT, SPEC, TEMPLATE, ENVIRONMENT):
            text = _read(path)
            for old_name in REMOVED_ENTRYPOINTS + REMOVED_PHASES:
                self.assertNotIn(old_name, text, f"{old_name} remains in {path}")

    def test_context_and_template_define_the_same_public_categories(self):
        context = _read(CONTEXT)
        template = _read(TEMPLATE)

        for category in FAILURE_CATEGORIES:
            self.assertIn(category, context)
            self.assertIn(category, template)

        for required_rule in (
            "浮点",
            "整数",
            "有限值",
            "不得放宽",
            "真实模型",
        ):
            self.assertIn(required_rule, template)

    def test_environment_guide_keeps_preflight_separate_from_operator_gaps(self):
        guide = _read(ENVIRONMENT)

        self.assertIn("SGLANG_PLATFORM=kunlun", guide)
        self.assertIn("SGLANG_IS_FLASHINFER_AVAILABLE=False", guide)
        self.assertIn("PYTHONPATH", guide)
        self.assertIn("ENVIRONMENT", guide)
        self.assertIn("不能记为", guide)
        self.assertIn("OPERATOR_MISSING", guide)


if __name__ == "__main__":
    unittest.main()
