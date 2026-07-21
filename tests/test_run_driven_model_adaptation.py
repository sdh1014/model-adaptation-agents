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
OPERATOR_GAP_ANALYSIS = ROOT / "docs" / "step3p7-p800-operator-gap-analysis.md"
BUG_REPAIR_LOOP = ROOT / "model-adaptation" / "references" / "bug-repair-loop.md"
BUG_LOG = ROOT / "docs" / "step3p7-p800-bug-log.md"

FAILURE_CATEGORIES = (
    "ENVIRONMENT",
    "ADAPTATION",
    "OPERATOR_MISSING",
    "OPERATOR_CONTRACT",
    "DISTRIBUTED_RUNTIME",
)

REMOVED_ENTRYPOINTS = (
    "capture_golden.py",
    "replay_compare.py",
    "handoff_bundle.py",
    "workspace_guard.py",
    "model_adaptation_capture",
)

REMOVED_PHASES = (
    "CUDA_CAPTURE",
    "HANDOFF",
    "P800_REPAIR",
    "MODEL_ACCURACY",
    "ACCURACY_DEBUG",
)


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


def _operator_queue_rows(spec_text: str) -> dict[str, dict[str, str]]:
    rows = re.findall(
        r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*"
        r"`(ROUTE_PENDING|PRODUCTION_REACHABLE|ROUTE_BLOCKED)`\s*\|\s*"
        r"`(PENDING|ACTIVE|PASS|BLOCKED)`\s*\|",
        spec_text,
        re.MULTILINE,
    )
    return {
        historical_symbol: {
            "candidate_id": candidate_id,
            "implementation_kind": implementation_kind,
            "route_state": route_state,
            "status": status,
        }
        for candidate_id, historical_symbol, implementation_kind, route_state, status in rows
    }


class RunDrivenModelAdaptationTest(unittest.TestCase):
    def test_public_skill_is_one_compact_eager_loop(self):
        skill = _read(SKILL)

        self.assertLessEqual(len(skill.splitlines()), 220)
        self.assertIn("disable-model-invocation: true", skill)
        self.assertIn("PREFLIGHT -> OPERATOR_VERIFICATION -> EAGER_BRINGUP -> DONE", skill)
        self.assertIn("Operator Verification Queue", skill)
        self.assertIn("CPU reference", skill)
        self.assertIn("P800 Production Operator", skill)
        self.assertIn("回答正常", skill)
        self.assertIn("references/bug-repair-loop.md", skill)
        self.assertIn("docs/step3p7-p800-bug-log.md", skill)

        for category in FAILURE_CATEGORIES:
            self.assertIn(category, skill)
        for phase in REMOVED_PHASES:
            self.assertNotIn(phase, skill)

    def test_contract_keeps_all_candidates_and_only_operator_precision(self):
        requirements = _read(REQUIREMENTS_SPEC)
        spec = _read(SPEC)
        contract = _contract_data(spec)
        required_candidates = _required_operator_candidates(requirements)
        queue_rows = _operator_queue_rows(spec)

        self.assertEqual(contract["schema"], "model-adaptation/v1")
        self.assertEqual(contract["contract_revision"], 10)
        self.assertEqual(contract["runtime"]["tensor_parallel_size"], 8)
        self.assertEqual(contract["runtime"]["dtype"], "bfloat16")
        self.assertEqual(contract["runtime"]["cuda_graph_backend_decode"], "disabled")
        self.assertEqual(contract["runtime"]["cuda_graph_backend_prefill"], "disabled")
        self.assertEqual(set(contract["precision"]), {"operator"})
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
        self.assertEqual(contract["repair_scope"]["max_attempts_per_bug"], 3)

        self.assertEqual(len(required_candidates), 5)
        self.assertEqual(set(required_candidates), set(queue_rows))
        for candidate in required_candidates:
            row = queue_rows[candidate]
            self.assertEqual(row["implementation_kind"], "NATIVE_IMPLEMENTATION")
            self.assertEqual(row["route_state"], "ROUTE_PENDING")
            self.assertEqual(row["status"], "PENDING")

        self.assertIn("- `phase`: `PREFLIGHT`", spec)
        self.assertIn("- `auto_repair_status`: `NOT_EXERCISED`", spec)
        self.assertIn("- `active_attempt`: `0`", spec)
        self.assertIn("- `model_status`: `NOT_STARTED`", spec)
        self.assertNotIn("accuracy_status", spec)

        last_run = re.search(r"- `last_run`: `([^`]+)`", spec)
        self.assertIsNotNone(last_run)
        self.assertTrue((ROOT / last_run.group(1)).is_file())

    def test_operator_analysis_distinguishes_native_implementation_from_route(self):
        spec = _read(SPEC)
        context = _read(CONTEXT)
        analysis = _read(OPERATOR_GAP_ANALYSIS)
        queue_rows = _operator_queue_rows(spec)

        gemma = queue_rows["sgl_kernel.gemma_rmsnorm"]
        self.assertEqual(gemma["candidate_id"], "step3p7.norm.gemma_rmsnorm")
        self.assertEqual(gemma["implementation_kind"], "NATIVE_IMPLEMENTATION")
        self.assertEqual(gemma["route_state"], "ROUTE_PENDING")

        for required_text in (
            "forward_native",
            "GemmaRMSNorm",
            "546ad8c682392922792bbbfe53a8bf575545f118",
            "sglang_kunlun/platform/device.py",
            "sgl_kernel_stub.py",
        ):
            self.assertIn(required_text, analysis)

        self.assertIn("NATIVE_IMPLEMENTATION", context)
        self.assertIn("ROUTE_BLOCKED", context)
        self.assertIn("head_dim=96", spec)
        self.assertIn("non-causal", spec)

    def test_eager_success_is_done_not_a_planned_accuracy_block(self):
        skill = _read(SKILL)
        spec = _read(SPEC)
        template = _read(TEMPLATE)
        context = _read(CONTEXT)

        for required_transition in (
            "model_status: PASS",
            "status: PASS",
            "phase: DONE",
            "next_action: null",
        ):
            self.assertIn(required_transition, skill)

        for path in (SKILL, SPEC, TEMPLATE, CONTEXT, REQUIREMENTS_SPEC):
            text = _read(path)
            for obsolete in (
                "MODEL_ACCURACY",
                "ACCURACY_DEBUG",
                "block_after_eager_bringup",
                "accuracy_status",
                "reference_runtime",
            ):
                self.assertNotIn(obsolete, text, f"{obsolete} remains in {path}")

    def test_bug_loop_has_three_evidence_backed_attempts_and_archive(self):
        skill = _read(SKILL)
        loop = _read(BUG_REPAIR_LOOP)
        bug_log = _read(BUG_LOG)

        for verdict in ("NOT_EXERCISED", "ACTIVE", "PASS", "BLOCKED"):
            self.assertIn(verdict, skill)

        for required_rule in (
            "同一个 BUG 的 3 次不同",
            "每个 attempt 只验证一个",
            "聚焦复现",
            "最初失败的 P800",
            "不得主动注入故障",
        ):
            self.assertIn(required_rule, skill)

        for completion_signal in (
            "最多允许 3 次",
            "一个根因假设",
            "相邻回归",
            "第 3 次不同 attempt",
            "status: BLOCKED",
        ):
            self.assertIn(completion_signal, loop)

        for record_field in (
            "bug_id",
            "attempt_count",
            "max_attempts: 3",
            "exact_command",
            "confirmed_root_cause",
            "solution",
            "focused_regression",
            "original_p800_path_rerun",
            "required_next_capability",
            "closure_evidence",
        ):
            self.assertIn(record_field, bug_log)

        self.assertIn("当前尚未执行 revision 10 的 P800 运行", bug_log)

    def test_old_replay_module_and_live_references_are_removed(self):
        self.assertFalse((ROOT / "model-adaptation" / "scripts").exists())
        self.assertFalse((ROOT / "model-adaptation" / "pyproject.toml").exists())

        for path in (
            ROOT / "model-adaptation" / "references" / "cuda-capture-validation.md",
            ROOT / "model-adaptation" / "references" / "formal-cuda-capture-r6.md",
            ROOT / "model-adaptation" / "references" / "formal-cuda-capture.md",
            ROOT / "model-adaptation" / "references" / "formal-cuda-handoff.md",
            ROOT / "model-adaptation" / "references" / "p800-baseline-replay.md",
            ROOT / "model-adaptation" / "references" / "p800-handoff-verification.md",
            ROOT / "model-adaptation" / "references" / "claude-p800-repair.md",
            ROOT / "outputs" / "step3p7-p800-migration-tool-design.md",
        ):
            self.assertFalse(path.exists(), str(path))

        for path in (SKILL, CONTEXT, SPEC, TEMPLATE, ENVIRONMENT):
            text = _read(path)
            for old_name in REMOVED_ENTRYPOINTS + REMOVED_PHASES:
                self.assertNotIn(old_name, text, f"{old_name} remains in {path}")

    def test_context_and_template_share_categories_and_stop_rules(self):
        context = _read(CONTEXT)
        template = _read(TEMPLATE)

        for category in FAILURE_CATEGORIES:
            self.assertIn(category, context)
            self.assertIn(category, template)

        for text in (context, template):
            self.assertIn("3 次", text)
            self.assertIn("PASS / DONE", text)
            self.assertIn("BLOCKED", text)

    def test_environment_guide_keeps_preflight_separate_from_operator_missing(self):
        guide = _read(ENVIRONMENT)

        self.assertIn("SGLANG_PLATFORM=kunlun", guide)
        self.assertIn("SGLANG_IS_FLASHINFER_AVAILABLE=False", guide)
        self.assertIn("PYTHONPATH", guide)
        self.assertIn("ENVIRONMENT", guide)
        self.assertIn("不能记为", guide)
        self.assertIn("OPERATOR_MISSING", guide)
        self.assertNotIn("NEEDS_HUMAN", guide)


if __name__ == "__main__":
    unittest.main()
