import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_SPEC = ROOT / ".scratch" / "run-driven-model-adaptation" / "spec.md"
SKILL = ROOT / "model-adaptation" / "SKILL.md"
CONTEXT = ROOT / "CONTEXT.md"
SPEC = ROOT / "migration-spec.md"
TEMPLATE = ROOT / "model-adaptation" / "references" / "migration-spec-template.md"
ENVIRONMENT = ROOT / "docs" / "p800-environment-and-repair.md"

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


def _required_operator_candidates(requirements_text: str) -> tuple[str, ...]:
    match = re.search(
        r"<!-- REQUIRED-OPERATOR-CANDIDATES: BEGIN -->\s*(.*?)\s*"
        r"<!-- REQUIRED-OPERATOR-CANDIDATES: END -->",
        requirements_text,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError("Requirements must expose the operator candidate list")
    return tuple(re.findall(r"^\s*-\s+`([^`]+)`\s*$", match.group(1), re.MULTILINE))


def _operator_queue_statuses(spec_text: str) -> dict[str, str]:
    return dict(
        re.findall(
            r"^\|\s*`([^`]+)`\s*\|\s*`(PENDING|ACTIVE|PASS|BLOCKED)`\s*\|",
            spec_text,
            re.MULTILINE,
        )
    )


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

        self.assertIn("torch.testing.assert_close", skill)
        self.assertIn("生产调用", skill)
        self.assertIn("CPU reference", skill)
        self.assertIn("Operator Verification Queue 中的每一行", skill)
        self.assertIn("dumper", skill)
        self.assertIn("comparator", skill)

    def test_current_spec_queues_every_required_candidate_under_fixed_precision(self):
        requirements = _read(REQUIREMENTS_SPEC)
        spec = _read(SPEC)
        contract = _contract_data(spec)
        required_candidates = _required_operator_candidates(requirements)
        queue_statuses = _operator_queue_statuses(spec)

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

        self.assertEqual(len(required_candidates), 5)
        self.assertEqual(len(set(required_candidates)), len(required_candidates))
        for candidate in required_candidates:
            self.assertEqual(queue_statuses.get(candidate), "PENDING")

        self.assertIn("- `phase`: `PREFLIGHT`", spec)
        self.assertIn("- `model_status`: `NOT_STARTED`", spec)
        self.assertIn("- `accuracy_status`: `NOT_STARTED`", spec)

        last_run_match = re.search(r"- `last_run`: `([^`]+)`", spec)
        self.assertIsNotNone(last_run_match)
        last_run = last_run_match.group(1)
        self.assertNotEqual(last_run, "null")
        self.assertTrue((ROOT / last_run).is_file(), last_run)

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

    def test_environment_guide_keeps_preflight_separate_from_operator_missing(self):
        guide = _read(ENVIRONMENT)

        self.assertIn("SGLANG_PLATFORM=kunlun", guide)
        self.assertIn("SGLANG_IS_FLASHINFER_AVAILABLE=False", guide)
        self.assertIn("PYTHONPATH", guide)
        self.assertIn("ENVIRONMENT", guide)
        self.assertIn("不能记为", guide)
        self.assertIn("OPERATOR_MISSING", guide)


if __name__ == "__main__":
    unittest.main()
