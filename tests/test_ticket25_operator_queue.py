"""Continue repair attempts without losing previously accepted patches."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_ticket14_repair_loop import (
    ALLOWED_PATHS,
    create_test_repository,
    finish_candidate,
    git,
    guard_command,
    record_candidate,
    replay,
    write_spec,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "model-adaptation" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import capture_golden
import replay_compare
from _lib.spec_contract import load_contract_data, load_spec_binding
from model_adaptation_capture.contracts import (
    SWIGLU_KUNLUN_SOURCE_RELATIVE_PATH,
)

SKILL = ROOT / "model-adaptation" / "SKILL.md"
SPEC = ROOT / "migration-spec.md"
TEMPLATE = (
    ROOT
    / "model-adaptation"
    / "references"
    / "migration-spec-template.md"
)
INLINE_CORRECTION_PATCH = (
    ROOT
    / "runs"
    / "inline-repair-correction-r5-001"
    / "candidate.patch"
)


class CumulativeRepairPatchTest(unittest.TestCase):
    def test_inline_correction_is_a_valid_git_patch(self) -> None:
        parsed = git(
            ROOT,
            "apply",
            "--numstat",
            str(INLINE_CORRECTION_PATCH),
        )
        self.assertIn(
            "5\t1\t"
            "sglang-kunlun/sglang_kunlun/hooks/layers/"
            "quantization/unquant.py",
            parsed.stdout,
        )

    def test_candidate_binding_uses_live_original_callsite_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            worktree = Path(temp_dir) / "sglang-kunlun"
            source = worktree / SWIGLU_KUNLUN_SOURCE_RELATIVE_PATH
            source.parent.mkdir(parents=True)
            source.write_text(
                "def unquantized_fused_moe_apply_kunlun(layer, y):\n"
                "    kunlun_ops.swiglu(x=y, y=y)\n",
                encoding="utf-8",
            )
            (worktree / "README.md").write_text(
                "fixture\n",
                encoding="utf-8",
            )
            git(worktree, "init", "-q")
            git(worktree, "config", "user.email", "queue@example.com")
            git(worktree, "config", "user.name", "Queue Test")
            git(worktree, "add", ".")
            git(worktree, "commit", "-qm", "baseline")
            baseline = git(
                worktree,
                "rev-parse",
                "HEAD",
            ).stdout.strip()

            source.write_text(
                "def unquantized_fused_moe_apply_kunlun(layer, y):\n"
                "    limit = layer.moe_runner_config.gemm1_clamp_limit\n"
                "    if limit is None:\n"
                "        kunlun_ops.swiglu(x=y, y=y)\n"
                "    else:\n"
                "        kunlun_ops.swiglu(x=y, y=y, limit=float(limit))\n",
                encoding="utf-8",
            )
            modified_paths = [SWIGLU_KUNLUN_SOURCE_RELATIVE_PATH]
            recorded_patch = replay_compare.candidate_worktree_patch(
                worktree,
                baseline,
                modified_paths,
            )
            metadata = replay_compare.original_swiglu_call_metadata(
                worktree,
                modified_paths,
                baseline,
            )
            self.assertEqual(
                metadata["original_call_arguments"],
                {
                    "x": "y",
                    "y": "y",
                    "limit": (
                        "float(layer.moe_runner_config.gemm1_clamp_limit)"
                    ),
                },
            )

            source.write_text(
                source.read_text(encoding="utf-8") + "# drift\n",
                encoding="utf-8",
            )
            self.assertNotEqual(
                replay_compare.candidate_worktree_patch(
                    worktree,
                    baseline,
                    modified_paths,
                ),
                recorded_patch,
            )

    def test_unrelated_patch_cannot_claim_original_swiglu_callsite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            worktree = Path(temp_dir) / "sglang-kunlun"
            source = worktree / SWIGLU_KUNLUN_SOURCE_RELATIVE_PATH
            source.parent.mkdir(parents=True)
            source.write_text(
                "def unquantized_fused_moe_apply_kunlun(layer, y):\n"
                "    kunlun_ops.swiglu(x=y, y=y)\n",
                encoding="utf-8",
            )
            (worktree / "README.md").write_text(
                "fixture\n",
                encoding="utf-8",
            )
            git(worktree, "init", "-q")
            git(worktree, "config", "user.email", "queue@example.com")
            git(worktree, "config", "user.name", "Queue Test")
            git(worktree, "add", ".")
            git(worktree, "commit", "-qm", "baseline")
            baseline = git(worktree, "rev-parse", "HEAD").stdout.strip()
            (worktree / "README.md").write_text(
                "unrelated change\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                replay_compare.ToolError,
                "original SwiGLU production call site",
            ):
                replay_compare.original_swiglu_call_metadata(
                    worktree,
                    ["README.md"],
                    baseline,
                )

    def test_original_call_binding_rejects_fake_argument_wiring(self) -> None:
        invalid_sources = {
            "constant limit": (
                "def unquantized_fused_moe_apply_kunlun(layer, y):\n"
                "    limit = layer.moe_runner_config.gemm1_clamp_limit\n"
                "    if limit is None:\n"
                "        kunlun_ops.swiglu(x=y, y=y)\n"
                "    else:\n"
                "        kunlun_ops.swiglu(x=y, y=y, limit=0.0)\n"
            ),
            "wrong x": (
                "def unquantized_fused_moe_apply_kunlun(layer, y):\n"
                "    limit = layer.moe_runner_config.gemm1_clamp_limit\n"
                "    if limit is None:\n"
                "        kunlun_ops.swiglu(x=y, y=y)\n"
                "    else:\n"
                "        kunlun_ops.swiglu(x=limit, y=y, limit=float(limit))\n"
            ),
            "dead branch": (
                "def unquantized_fused_moe_apply_kunlun(layer, y):\n"
                "    limit = layer.moe_runner_config.gemm1_clamp_limit\n"
                "    kunlun_ops.swiglu(x=y, y=y)\n"
                "    if False:\n"
                "        kunlun_ops.swiglu(x=y, y=y, limit=float(limit))\n"
            ),
            "entire correction in dead branch": (
                "def unquantized_fused_moe_apply_kunlun(layer, y):\n"
                "    if False:\n"
                "        limit = layer.moe_runner_config.gemm1_clamp_limit\n"
                "        if limit is None:\n"
                "            kunlun_ops.swiglu(x=y, y=y)\n"
                "        else:\n"
                "            kunlun_ops.swiglu(\n"
                "                x=y, y=y, limit=float(limit)\n"
                "            )\n"
            ),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            worktree = Path(temp_dir) / "sglang-kunlun"
            source = worktree / SWIGLU_KUNLUN_SOURCE_RELATIVE_PATH
            source.parent.mkdir(parents=True)
            source.write_text(
                "def unquantized_fused_moe_apply_kunlun(layer, y):\n"
                "    kunlun_ops.swiglu(x=y, y=y)\n",
                encoding="utf-8",
            )
            git(worktree, "init", "-q")
            git(worktree, "config", "user.email", "queue@example.com")
            git(worktree, "config", "user.name", "Queue Test")
            git(worktree, "add", ".")
            git(worktree, "commit", "-qm", "baseline")
            baseline = git(worktree, "rev-parse", "HEAD").stdout.strip()

            for label, invalid_source in invalid_sources.items():
                with self.subTest(label=label):
                    source.write_text(invalid_source, encoding="utf-8")
                    with self.assertRaisesRegex(
                        replay_compare.ToolError,
                        "preserve the original x/y arguments",
                    ):
                        replay_compare.original_swiglu_call_metadata(
                            worktree,
                            [SWIGLU_KUNLUN_SOURCE_RELATIVE_PATH],
                            baseline,
                        )

    def test_next_operator_keeps_passed_patch_and_resets_its_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree, baseline = create_test_repository(workspace)
            spec = workspace / "migration-spec.md"
            write_spec(spec, baseline)

            first_replay = workspace / "runs" / "first-baseline-replay"
            self.assertEqual(
                replay(
                    spec,
                    first_replay,
                    passed=False,
                    operator_id="operator.one",
                ).returncode,
                0,
            )
            first_baseline = workspace / "runs" / "first-baseline"
            self.assertEqual(
                guard_command(
                    "assess-baseline",
                    spec,
                    first_baseline,
                    worktree,
                    "--operator-id",
                    "operator.one",
                    "--replay-result",
                    str(first_replay / "result.json"),
                    allow_synthetic_replay=True,
                ).returncode,
                0,
            )
            first_attempt = workspace / "runs" / "operator-one-attempt-1"
            self.assertEqual(
                guard_command(
                    "start-attempt",
                    spec,
                    first_attempt,
                    worktree,
                    "--operator-id",
                    "operator.one",
                    "--attempt",
                    "1",
                    "--hypothesis",
                    "repair the first operator",
                    "--previous-result",
                    str(first_baseline / "result.json"),
                ).returncode,
                0,
            )
            (worktree / ALLOWED_PATHS[0]).write_text(
                "def swiglu(x):\n"
                "    return x + 1\n",
                encoding="utf-8",
            )
            self.assertEqual(
                record_candidate(spec, first_attempt, worktree).returncode,
                0,
            )
            self.assertEqual(
                replay(
                    spec,
                    first_attempt / "replay",
                    passed=True,
                    operator_id="operator.one",
                ).returncode,
                0,
            )
            self.assertEqual(
                finish_candidate(spec, first_attempt, worktree).returncode,
                0,
            )
            first_result = first_attempt / "result.json"

            second_replay = workspace / "runs" / "second-baseline-replay"
            self.assertEqual(
                replay(
                    spec,
                    second_replay,
                    passed=False,
                    operator_id="operator.two",
                ).returncode,
                0,
            )
            incomplete_baseline = workspace / "runs" / "incomplete-second-baseline"
            rejected = guard_command(
                "assess-baseline",
                spec,
                incomplete_baseline,
                worktree,
                "--operator-id",
                "operator.two",
                "--accepted-result",
                str(first_result),
                "--replay-result",
                str(second_replay / "result.json"),
                allowed_paths=(ALLOWED_PATHS[1],),
                allow_synthetic_replay=True,
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("previously accepted modified path", rejected.stderr)
            self.assertFalse(incomplete_baseline.exists())

            second_baseline = workspace / "runs" / "second-baseline"
            assessed = guard_command(
                "assess-baseline",
                spec,
                second_baseline,
                worktree,
                "--operator-id",
                "operator.two",
                "--accepted-result",
                str(first_result),
                "--replay-result",
                str(second_replay / "result.json"),
                allow_synthetic_replay=True,
            )
            self.assertEqual(assessed.returncode, 0, assessed.stderr)
            assessment = json.loads(
                (second_baseline / "result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(assessment["operator_id"], "operator.two")
            self.assertEqual(
                assessment["accepted_passing_run"],
                str(first_result.resolve()),
            )
            self.assertEqual(assessment["attempts_used"], 0)
            self.assertFalse(assessment["workspace_clean"])
            self.assertTrue(assessment["workspace_ready"])

            failed_attempt = workspace / "runs" / "operator-two-attempt-1"
            started = guard_command(
                "start-attempt",
                spec,
                failed_attempt,
                worktree,
                "--operator-id",
                "operator.two",
                "--accepted-result",
                str(first_result),
                "--attempt",
                "1",
                "--hypothesis",
                "repair the second operator but accidentally break the first",
                "--previous-result",
                str(second_baseline / "result.json"),
                "--regression-operator-id",
                "operator.one",
            )
            self.assertEqual(started.returncode, 0, started.stderr)
            (worktree / ALLOWED_PATHS[0]).write_text(
                "def swiglu(x):\n"
                "    return x + 99\n",
                encoding="utf-8",
            )
            (worktree / ALLOWED_PATHS[1]).write_text(
                "def test_swiglu():\n"
                "    assert 1 + 1 == 2\n",
                encoding="utf-8",
            )
            self.assertEqual(
                record_candidate(spec, failed_attempt, worktree).returncode,
                0,
            )
            self.assertEqual(
                replay(
                    spec,
                    failed_attempt / "replay",
                    passed=True,
                    operator_id="operator.two",
                ).returncode,
                0,
            )
            missing_regression = finish_candidate(
                spec,
                failed_attempt,
                worktree,
            )
            self.assertEqual(missing_regression.returncode, 2)
            self.assertIn(
                "missing one or more required regressions",
                missing_regression.stderr,
            )
            self.assertFalse((failed_attempt / "outcome.json").exists())
            first_regression = (
                failed_attempt
                / "regressions"
                / "operator-one"
            )
            self.assertEqual(
                replay(
                    spec,
                    first_regression,
                    passed=False,
                    operator_id="operator.one",
                ).returncode,
                0,
            )
            failed = finish_candidate(
                spec,
                failed_attempt,
                worktree,
                regression_results=(
                    first_regression / "result.json",
                ),
            )
            self.assertEqual(failed.returncode, 0, failed.stderr)
            failed_result = json.loads(
                (failed_attempt / "result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(failed_result["active_replay_passed"])
            self.assertFalse(failed_result["passed"])
            self.assertFalse(
                failed_result["regression_results"][0]["passed"]
            )
            self.assertIn(
                "return x + 1",
                (worktree / ALLOWED_PATHS[0]).read_text(encoding="utf-8"),
            )
            self.assertIn(
                "assert True",
                (worktree / ALLOWED_PATHS[1]).read_text(encoding="utf-8"),
            )

            passing_attempt = workspace / "runs" / "operator-two-attempt-2"
            started = guard_command(
                "start-attempt",
                spec,
                passing_attempt,
                worktree,
                "--operator-id",
                "operator.two",
                "--accepted-result",
                str(first_result),
                "--attempt",
                "2",
                "--hypothesis",
                "repair the second operator while preserving the first",
                "--previous-result",
                str(failed_attempt / "result.json"),
                "--regression-operator-id",
                "operator.one",
            )
            self.assertEqual(started.returncode, 0, started.stderr)
            (worktree / ALLOWED_PATHS[1]).write_text(
                "def test_swiglu():\n"
                "    assert 1 + 1 == 2\n",
                encoding="utf-8",
            )
            self.assertEqual(
                record_candidate(spec, passing_attempt, worktree).returncode,
                0,
            )
            self.assertEqual(
                replay(
                    spec,
                    passing_attempt / "replay",
                    passed=True,
                    operator_id="operator.two",
                ).returncode,
                0,
            )
            passing_regression = (
                passing_attempt
                / "regressions"
                / "operator-one"
            )
            self.assertEqual(
                replay(
                    spec,
                    passing_regression,
                    passed=True,
                    operator_id="operator.one",
                ).returncode,
                0,
            )
            passed = finish_candidate(
                spec,
                passing_attempt,
                worktree,
                regression_results=(
                    passing_regression / "result.json",
                ),
            )
            self.assertEqual(passed.returncode, 0, passed.stderr)
            cumulative_patch = (
                passing_attempt / "candidate.patch"
            ).read_text(encoding="utf-8")
            self.assertIn("return x + 1", cumulative_patch)
            self.assertIn("assert 1 + 1 == 2", cumulative_patch)

    def test_baseline_pass_keeps_the_last_patch_anchor_for_later_operator(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree, baseline = create_test_repository(workspace)
            spec = workspace / "migration-spec.md"
            write_spec(spec, baseline)

            first_replay = workspace / "runs" / "one-baseline-replay"
            replay(
                spec,
                first_replay,
                passed=False,
                operator_id="operator.one",
            )
            first_baseline = workspace / "runs" / "one-baseline"
            assessed = guard_command(
                "assess-baseline",
                spec,
                first_baseline,
                worktree,
                "--operator-id",
                "operator.one",
                "--replay-result",
                str(first_replay / "result.json"),
                allow_synthetic_replay=True,
            )
            self.assertEqual(assessed.returncode, 0, assessed.stderr)
            first_attempt = workspace / "runs" / "one-attempt"
            started = guard_command(
                "start-attempt",
                spec,
                first_attempt,
                worktree,
                "--operator-id",
                "operator.one",
                "--attempt",
                "1",
                "--hypothesis",
                "repair operator one",
                "--previous-result",
                str(first_baseline / "result.json"),
            )
            self.assertEqual(started.returncode, 0, started.stderr)
            (worktree / ALLOWED_PATHS[0]).write_text(
                "def swiglu(x):\n"
                "    return x + 1\n",
                encoding="utf-8",
            )
            self.assertEqual(
                record_candidate(spec, first_attempt, worktree).returncode,
                0,
            )
            replay(
                spec,
                first_attempt / "replay",
                passed=True,
                operator_id="operator.one",
            )
            self.assertEqual(
                finish_candidate(
                    spec,
                    first_attempt,
                    worktree,
                ).returncode,
                0,
            )
            accepted = first_attempt / "result.json"

            second_replay = workspace / "runs" / "two-baseline-replay"
            replay(
                spec,
                second_replay,
                passed=True,
                operator_id="operator.two",
            )
            second_baseline = workspace / "runs" / "two-baseline"
            assessed = guard_command(
                "assess-baseline",
                spec,
                second_baseline,
                worktree,
                "--operator-id",
                "operator.two",
                "--accepted-result",
                str(accepted),
                "--replay-result",
                str(second_replay / "result.json"),
                allow_synthetic_replay=True,
            )
            self.assertEqual(assessed.returncode, 0, assessed.stderr)
            self.assertTrue(
                json.loads(
                    (second_baseline / "result.json").read_text(
                        encoding="utf-8"
                    )
                )["passed"]
            )

            third_replay = workspace / "runs" / "three-baseline-replay"
            replay(
                spec,
                third_replay,
                passed=False,
                operator_id="operator.three",
            )
            third_baseline = workspace / "runs" / "three-baseline"
            assessed = guard_command(
                "assess-baseline",
                spec,
                third_baseline,
                worktree,
                "--operator-id",
                "operator.three",
                "--accepted-result",
                str(accepted),
                "--replay-result",
                str(third_replay / "result.json"),
                allow_synthetic_replay=True,
            )
            self.assertEqual(assessed.returncode, 0, assessed.stderr)
            third_attempt = workspace / "runs" / "three-attempt"
            started = guard_command(
                "start-attempt",
                spec,
                third_attempt,
                worktree,
                "--operator-id",
                "operator.three",
                "--accepted-result",
                str(accepted),
                "--regression-operator-id",
                "operator.one",
                "--regression-operator-id",
                "operator.two",
                "--attempt",
                "1",
                "--hypothesis",
                "repair operator three",
                "--previous-result",
                str(third_baseline / "result.json"),
            )
            self.assertEqual(started.returncode, 0, started.stderr)
            attempt = json.loads(
                (third_attempt / "attempt.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                attempt["accepted_passing_run"],
                str(accepted.resolve()),
            )
            self.assertEqual(
                attempt["regression_operator_ids"],
                ["operator.one", "operator.two"],
            )

    def test_third_repair_cannot_drop_first_operator_regression(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree, baseline = create_test_repository(workspace)
            spec = workspace / "migration-spec.md"
            write_spec(spec, baseline)

            accepted = None
            for index, operator_id in enumerate(
                ("operator.one", "operator.two")
            ):
                baseline_replay = (
                    workspace / "runs" / f"{operator_id}-baseline-replay"
                )
                self.assertEqual(
                    replay(
                        spec,
                        baseline_replay,
                        passed=False,
                        operator_id=operator_id,
                    ).returncode,
                    0,
                )
                baseline_run = (
                    workspace / "runs" / f"{operator_id}-baseline"
                )
                baseline_args = [
                    "--operator-id",
                    operator_id,
                    "--replay-result",
                    str(baseline_replay / "result.json"),
                ]
                if accepted is not None:
                    baseline_args.extend(
                        ["--accepted-result", str(accepted)]
                    )
                assessed = guard_command(
                    "assess-baseline",
                    spec,
                    baseline_run,
                    worktree,
                    *baseline_args,
                    allow_synthetic_replay=True,
                )
                self.assertEqual(assessed.returncode, 0, assessed.stderr)

                attempt_run = (
                    workspace / "runs" / f"{operator_id}-attempt"
                )
                attempt_args = [
                    "--operator-id",
                    operator_id,
                    "--attempt",
                    "1",
                    "--hypothesis",
                    f"repair {operator_id}",
                    "--previous-result",
                    str(baseline_run / "result.json"),
                ]
                if accepted is not None:
                    attempt_args.extend(
                        [
                            "--accepted-result",
                            str(accepted),
                            "--regression-operator-id",
                            "operator.one",
                        ]
                    )
                started = guard_command(
                    "start-attempt",
                    spec,
                    attempt_run,
                    worktree,
                    *attempt_args,
                )
                self.assertEqual(started.returncode, 0, started.stderr)
                target = worktree / ALLOWED_PATHS[index]
                target.write_text(
                    (
                        "def swiglu(x):\n"
                        "    return x + 1\n"
                        if index == 0
                        else
                        "def test_swiglu():\n"
                        "    assert 1 + 1 == 2\n"
                    ),
                    encoding="utf-8",
                )
                self.assertEqual(
                    record_candidate(
                        spec,
                        attempt_run,
                        worktree,
                    ).returncode,
                    0,
                )
                self.assertEqual(
                    replay(
                        spec,
                        attempt_run / "replay",
                        passed=True,
                        operator_id=operator_id,
                    ).returncode,
                    0,
                )
                regression_results = ()
                if accepted is not None:
                    regression_run = (
                        attempt_run / "regressions" / "operator-one"
                    )
                    self.assertEqual(
                        replay(
                            spec,
                            regression_run,
                            passed=True,
                            operator_id="operator.one",
                        ).returncode,
                        0,
                    )
                    regression_results = (
                        regression_run / "result.json",
                    )
                finished = finish_candidate(
                    spec,
                    attempt_run,
                    worktree,
                    regression_results=regression_results,
                )
                self.assertEqual(finished.returncode, 0, finished.stderr)
                accepted = attempt_run / "result.json"

            third_replay = workspace / "runs" / "operator-three-replay"
            self.assertEqual(
                replay(
                    spec,
                    third_replay,
                    passed=False,
                    operator_id="operator.three",
                ).returncode,
                0,
            )
            third_baseline = workspace / "runs" / "operator-three-baseline"
            assessed = guard_command(
                "assess-baseline",
                spec,
                third_baseline,
                worktree,
                "--operator-id",
                "operator.three",
                "--accepted-result",
                str(accepted),
                "--replay-result",
                str(third_replay / "result.json"),
                allow_synthetic_replay=True,
            )
            self.assertEqual(assessed.returncode, 0, assessed.stderr)

            incomplete = guard_command(
                "start-attempt",
                spec,
                workspace / "runs" / "operator-three-incomplete",
                worktree,
                "--operator-id",
                "operator.three",
                "--accepted-result",
                str(accepted),
                "--regression-operator-id",
                "operator.two",
                "--attempt",
                "1",
                "--hypothesis",
                "drop the first historical regression",
                "--previous-result",
                str(third_baseline / "result.json"),
            )
            self.assertEqual(incomplete.returncode, 2)
            self.assertIn("operator.one", incomplete.stderr)

            complete_run = workspace / "runs" / "operator-three-complete"
            complete = guard_command(
                "start-attempt",
                spec,
                complete_run,
                worktree,
                "--operator-id",
                "operator.three",
                "--accepted-result",
                str(accepted),
                "--regression-operator-id",
                "operator.one",
                "--regression-operator-id",
                "operator.two",
                "--attempt",
                "1",
                "--hypothesis",
                "preserve the full historical regression list",
                "--previous-result",
                str(third_baseline / "result.json"),
            )
            self.assertEqual(complete.returncode, 0, complete.stderr)


class QueueSkillContractTest(unittest.TestCase):
    def test_skill_continues_to_next_sealed_operator_without_confirmation(
        self,
    ) -> None:
        skill = SKILL.read_text(encoding="utf-8")
        self.assertIn("自动选择下一项", skill)
        self.assertIn("不等待人工确认", skill)
        self.assertIn("全部 gap queue", skill)
        self.assertIn("--accepted-result", skill)
        self.assertIn("--regression-operator-id", skill)
        self.assertIn("--regression-result", skill)
        self.assertIn("最近一份非空 `passing_run`", skill)
        self.assertIn("baseline PASS Run", skill)
        self.assertIn("一个 config", skill)
        self.assertIn("全部 capture plan collector", skill)
        self.assertIn("旧 runbook", skill)
        self.assertIn("工具会从 `--accepted-result` 继承", skill)
        self.assertIn("全部 baseline 直接 PASS", skill)

    def test_template_tracks_each_operator_directly(self) -> None:
        template = TEMPLATE.read_text(encoding="utf-8")
        for field in (
            "golden_run",
            "repair_status",
            "attempts_used",
            "passing_run",
        ):
            self.assertIn(field, template)
        self.assertIn("golden_runs", template)
        self.assertIn("PASS / DONE", template)
        self.assertIn("全部算子都是 baseline 直接 PASS", template)

    def test_revision_6_capture_validation_accepts_every_planned_operator(
        self,
    ) -> None:
        contract = load_contract_data(SPEC)
        binding = load_spec_binding(SPEC).as_result_dict()
        operator_ids = ("operator.one", "operator.two")
        operators = [
            {
                "operator_id": operator_id,
                "verdict": "CAPTURE_REQUIRED",
                "model_path": ["target"],
                "boundary": {
                    "inputs": ["x"],
                    "parameters": [],
                    "non_tensor_args": ["limit"],
                    "outputs": ["output"],
                },
                "cuda_impl": ["cuda source"],
                "kunlun_impl": ["kunlun source"],
                "kernel_call": {
                    "capture_seam": f"existing.{operator_id}",
                },
            }
            for operator_id in operator_ids
        ]
        capture_plan = [
            {
                "operator_id": operator["operator_id"],
                "hook_target": operator["kernel_call"]["capture_seam"],
                "max_distinct_shapes": 3,
                "tp_rank": 0,
                "replay_mode": "standalone-kernel-call",
                "saved_inputs": ["x"],
                "saved_parameters": [],
                "saved_non_tensor_args": ["limit"],
                "saved_outputs": ["output"],
                "save_direct_parameter_tensors": True,
                "save_full_checkpoint": False,
                "save_module_state_dict": False,
            }
            for operator in operators
        ]
        scan = {
            "scan_complete": True,
            "spec_binding": binding,
            "source": {
                "sglang_revision": contract["source"]["sglang_revision"],
                "sglang_kunlun_revision": contract["source"][
                    "sglang_kunlun_revision"
                ],
                "checkpoint_id": contract["checkpoint"]["id"],
                "checkpoint_config_sha256": contract["checkpoint"][
                    "config_digest"
                ],
            },
            "scan_scope": contract["scan_scope"],
            "operators": operators,
            "capture_plan": capture_plan,
            "selection": {
                "evaluated_after_scan": True,
                "active_operator": operator_ids[0],
            },
        }

        for operator_id in operator_ids:
            selected = capture_golden.validate_scan_candidate(
                contract,
                scan,
                operator_id,
                binding,
            )
            self.assertEqual(selected["operator_id"], operator_id)


if __name__ == "__main__":
    unittest.main()
