import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "model-adaptation" / "scripts"


WORKSPACE_GUARD = (
    SCRIPTS / "workspace_guard.py"
)
REPLAY_COMPARE = (
    SCRIPTS / "replay_compare.py"
)
MODEL_ADAPTATION_SKILL = ROOT / "model-adaptation" / "SKILL.md"
CLAUDE_SKILL = ROOT / ".claude" / "skills" / "model-adaptation" / "SKILL.md"
MIGRATION_SPEC = ROOT / "migration-spec.md"
REPAIR_LOOP_RUN = ROOT / "runs" / "repair-replay-adapter-r5-001" / "result.json"
ALLOWED_PATHS = (
    "sglang-kunlun/sglang_kunlun/ops/swiglu.py",
    "tests/test_swiglu.py",
)


def run_command(
    *args: str,
    cwd: Path = ROOT,
) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    environment["PYTHONPYCACHEPREFIX"] = (
        "/tmp/model-adaptation-agents-pyc"
    )
    return subprocess.run(
        list(args),
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def git(worktree: Path, *args: str) -> subprocess.CompletedProcess:
    completed = run_command("git", *args, cwd=worktree)
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    return completed


def create_test_repository(root: Path) -> tuple[Path, str]:
    worktree = root / "sglang-kunlun"
    operator = worktree / ALLOWED_PATHS[0]
    focused_test = worktree / ALLOWED_PATHS[1]
    operator.parent.mkdir(parents=True)
    focused_test.parent.mkdir(parents=True)
    operator.write_text(
        "def swiglu(x):\n"
        "    return x\n",
        encoding="utf-8",
    )
    focused_test.write_text(
        "def test_swiglu():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    (worktree / "README.md").write_text(
        "fixture repository\n",
        encoding="utf-8",
    )
    git(worktree, "init", "-q")
    git(worktree, "config", "user.email", "repair-loop@example.com")
    git(worktree, "config", "user.name", "Repair Loop")
    git(worktree, "add", ".")
    git(worktree, "commit", "-qm", "baseline")
    revision = git(worktree, "rev-parse", "HEAD").stdout.strip()
    return worktree, revision


def write_spec(
    path: Path,
    baseline_revision: str,
    *,
    spec_id: str = "ticket14-repair-loop-test",
) -> None:
    text = MIGRATION_SPEC.read_text(encoding="utf-8")
    begin = "<!-- CONTRACT-DATA: BEGIN -->"
    end = "<!-- CONTRACT-DATA: END -->"
    contract = json.loads(
        text.split(begin, 1)[1].split(end, 1)[0].strip()
    )
    contract["spec_id"] = spec_id
    contract["source"]["sglang_kunlun_revision"] = baseline_revision
    path.write_text(
        "\n".join(
            [
                "# Repair Loop Test Spec",
                begin,
                json.dumps(contract, ensure_ascii=False, indent=2),
                end,
                "",
            ]
        ),
        encoding="utf-8",
    )


def guard_command(
    mode: str,
    spec: Path,
    run_dir: Path,
    worktree: Path,
    *extra: str,
    include_allowed_paths: bool = True,
    allowed_paths: tuple[str, ...] = ALLOWED_PATHS,
    allow_synthetic_replay: bool = False,
) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        str(WORKSPACE_GUARD),
        "--mode",
        mode,
        "--spec",
        str(spec),
        "--run-dir",
        str(run_dir),
        "--worktree",
        str(worktree),
    ]
    if include_allowed_paths:
        for path in allowed_paths:
            command.extend(["--allowed-path", path])
    if allow_synthetic_replay:
        command.append("--allow-synthetic-replay")
    command.extend(extra)
    return run_command(*command)


def replay(
    spec: Path,
    run_dir: Path,
    *,
    passed: bool,
) -> subprocess.CompletedProcess:
    case_path = run_dir.parent / f"{run_dir.name}-case.json"
    case_path.parent.mkdir(parents=True, exist_ok=True)
    case_path.write_text(
        json.dumps(
            {
                "expected": {"value": 1},
                "actual": {"value": 1 if passed else 2},
            }
        ),
        encoding="utf-8",
    )
    return run_command(
        sys.executable,
        str(REPLAY_COMPARE),
        "--mode",
        "synthetic",
        "--spec",
        str(spec),
        "--run-dir",
        str(run_dir),
        "--case",
        str(case_path),
    )


def assess_baseline(
    workspace: Path,
    spec: Path,
    worktree: Path,
    *,
    passed: bool,
    allowed_paths: tuple[str, ...] = ALLOWED_PATHS,
) -> Path:
    replay_run = workspace / "runs" / "baseline-replay"
    replayed = replay(spec, replay_run, passed=passed)
    if replayed.returncode != 0:
        raise AssertionError(replayed.stderr)
    assessment = workspace / "runs" / "baseline-assessment"
    assessed = guard_command(
        "assess-baseline",
        spec,
        assessment,
        worktree,
        "--replay-result",
        str(replay_run / "result.json"),
        allowed_paths=allowed_paths,
        allow_synthetic_replay=True,
    )
    if assessed.returncode != 0:
        raise AssertionError(assessed.stderr)
    return assessment / "result.json"


def record_candidate(
    spec: Path,
    run_dir: Path,
    worktree: Path,
) -> subprocess.CompletedProcess:
    return guard_command(
        "record-candidate",
        spec,
        run_dir,
        worktree,
        include_allowed_paths=False,
    )


def finish_candidate(
    spec: Path,
    run_dir: Path,
    worktree: Path,
) -> subprocess.CompletedProcess:
    return guard_command(
        "finish-attempt",
        spec,
        run_dir,
        worktree,
        "--replay-result",
        str(run_dir / "replay" / "result.json"),
        include_allowed_paths=False,
        allow_synthetic_replay=True,
    )


class Ticket14RepairLoopTest(unittest.TestCase):
    def test_agent_driven_repair_flow_evidence_matches_current_sources(
        self,
    ) -> None:
        result = json.loads(REPAIR_LOOP_RUN.read_text(encoding="utf-8"))

        self.assertTrue(result["passed"])
        self.assertEqual(
            result["supersedes"],
            "runs/repair-loop-tool-002/result.json",
        )
        self.assertTrue(result["behavior"]["agent_selects_hypothesis"])
        self.assertTrue(
            result["behavior"]["agent_selects_allowed_paths_per_attempt"]
        )
        self.assertFalse(
            result["behavior"]["baseline_paths_lock_attempts"]
        )
        self.assertFalse(result["behavior"]["preauthored_attempt_patch"])
        self.assertEqual(result["runtime_validation"]["attempts_used"], 0)
        self.assertEqual(
            result["runtime_validation"]["p800_operator_repair"],
            "NOT_RUN",
        )
        self.assertEqual(
            result["validation"]["repair_replay_adapter"],
            "BUILT",
        )
        self.assertTrue(
            result["delegated_to_p800_agent"]["repair_replay_adapter"]
        )
        for source in result["source_files"]:
            self.assertEqual(
                hashlib.sha256(
                    (ROOT / source["path"]).read_bytes()
                ).hexdigest(),
                source["sha256"],
                source["path"],
            )

    def test_skill_leaves_repair_decisions_to_the_migration_agent(
        self,
    ) -> None:
        skill = MODEL_ADAPTATION_SKILL.read_text(encoding="utf-8")
        claude_skill = CLAUDE_SKILL.read_text(encoding="utf-8")

        self.assertIn("SGLANG_KUNLUN_WORKTREE", skill)
        self.assertIn("具体修复方案由 Migration Agent", skill)
        self.assertIn("不得要求人替 Agent 指定改法", skill)
        self.assertIn("不同 attempt 可以选择不同文件", skill)
        self.assertIn("原始 Kernel Call 边界", skill)
        self.assertNotIn("--p800-call-target", skill)
        self.assertIn("candidate.patch", skill)
        self.assertIn(
            "`active_hypothesis: null` 是人工确认",
            skill,
        )
        self.assertIn("等待人选择补丁", skill)
        self.assertNotIn("p800-repair-attempt-r5-001.patch", skill)
        self.assertIn("主 Skill", claude_skill)
        self.assertIn("唯一 `next_action`", claude_skill)

    def test_start_attempt_accepts_agent_selected_paths_by_round(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree, baseline = create_test_repository(workspace)
            spec = workspace / "migration-spec.md"
            write_spec(spec, baseline)
            previous_result = assess_baseline(
                workspace,
                spec,
                worktree,
                passed=False,
                allowed_paths=(ALLOWED_PATHS[0],),
            )

            attempt_one = workspace / "runs" / "repair-001"
            started = guard_command(
                "start-attempt",
                spec,
                attempt_one,
                worktree,
                "--attempt",
                "1",
                "--hypothesis",
                "change the focused implementation and regression test",
                "--previous-result",
                str(previous_result),
                allowed_paths=ALLOWED_PATHS,
            )
            self.assertEqual(started.returncode, 0, started.stderr)
            (worktree / ALLOWED_PATHS[0]).write_text(
                "def swiglu(x):\n"
                "    return x + 1\n",
                encoding="utf-8",
            )
            recorded = record_candidate(spec, attempt_one, worktree)
            self.assertEqual(recorded.returncode, 0, recorded.stderr)
            replay(spec, attempt_one / "replay", passed=False)
            finished = finish_candidate(spec, attempt_one, worktree)
            self.assertEqual(finished.returncode, 0, finished.stderr)

            attempt_two = workspace / "runs" / "repair-002"
            started = guard_command(
                "start-attempt",
                spec,
                attempt_two,
                worktree,
                "--attempt",
                "2",
                "--hypothesis",
                "revise only the focused implementation",
                "--previous-result",
                str(attempt_one / "result.json"),
                allowed_paths=(ALLOWED_PATHS[0],),
            )
            self.assertEqual(started.returncode, 0, started.stderr)

    def test_baseline_pass_is_not_a_gap_and_uses_zero_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree, baseline = create_test_repository(workspace)
            spec = workspace / "migration-spec.md"
            write_spec(spec, baseline)

            checked = guard_command(
                "check-baseline",
                spec,
                workspace / "runs" / "baseline-check",
                worktree,
            )
            self.assertEqual(checked.returncode, 0, checked.stderr)

            assessment_result = assess_baseline(
                workspace,
                spec,
                worktree,
                passed=True,
            )
            result = json.loads(
                assessment_result.read_text(encoding="utf-8")
            )
            self.assertTrue(result["passed"])
            self.assertFalse(result["gap_observed"])
            self.assertEqual(result["attempts_used"], 0)
            self.assertTrue(result["workspace_clean"])

    def test_failed_attempt_restores_baseline_and_next_pass_is_retained(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree, baseline = create_test_repository(workspace)
            branch = git(
                worktree,
                "branch",
                "--show-current",
            ).stdout.strip()
            spec = workspace / "migration-spec.md"
            write_spec(spec, baseline)

            previous_result = assess_baseline(
                workspace,
                spec,
                worktree,
                passed=False,
            )
            baseline_result = json.loads(
                previous_result.read_text(encoding="utf-8")
            )
            self.assertTrue(baseline_result["gap_observed"])
            self.assertEqual(baseline_result["attempts_used"], 0)

            attempt_one = workspace / "runs" / "repair-001"
            started = guard_command(
                "start-attempt",
                spec,
                attempt_one,
                worktree,
                "--attempt",
                "1",
                "--hypothesis",
                "doubling the fallback is insufficient",
                "--previous-result",
                str(previous_result),
            )
            self.assertEqual(started.returncode, 0, started.stderr)
            (worktree / ALLOWED_PATHS[0]).write_text(
                "def swiglu(x):\n"
                "    return x * 2\n",
                encoding="utf-8",
            )
            recorded = record_candidate(spec, attempt_one, worktree)
            self.assertEqual(recorded.returncode, 0, recorded.stderr)
            replay(spec, attempt_one / "replay", passed=False)
            finished = finish_candidate(
                spec,
                attempt_one,
                worktree,
            )
            self.assertEqual(finished.returncode, 0, finished.stderr)
            failed = json.loads(
                (attempt_one / "result.json").read_text(encoding="utf-8")
            )
            self.assertFalse(failed["passed"])
            self.assertEqual(
                failed["workspace_state"],
                "BASELINE_RESTORED",
            )
            outcome = json.loads(
                (attempt_one / "outcome.json").read_text(encoding="utf-8")
            )
            self.assertFalse(outcome["passed"])
            self.assertEqual(
                outcome["recovery_action"],
                "RESTORE_BASELINE",
            )
            self.assertEqual(
                outcome["patch_sha256"],
                failed["patch_sha256"],
            )
            self.assertEqual(
                outcome["replay_result_sha256"],
                failed["replay_result_sha256"],
            )
            self.assertIn(
                "return x * 2",
                (attempt_one / "candidate.patch").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertEqual(
                git(worktree, "status", "--porcelain").stdout,
                "",
            )

            attempt_two = workspace / "runs" / "repair-002"
            started = guard_command(
                "start-attempt",
                spec,
                attempt_two,
                worktree,
                "--attempt",
                "2",
                "--hypothesis",
                "apply the complete corrected fallback",
                "--previous-result",
                str(attempt_one / "result.json"),
            )
            self.assertEqual(started.returncode, 0, started.stderr)
            (worktree / ALLOWED_PATHS[0]).write_text(
                "def swiglu(x):\n"
                "    return x + 1\n",
                encoding="utf-8",
            )
            recorded = record_candidate(spec, attempt_two, worktree)
            self.assertEqual(recorded.returncode, 0, recorded.stderr)
            replay(spec, attempt_two / "replay", passed=True)
            finished = finish_candidate(
                spec,
                attempt_two,
                worktree,
            )
            self.assertEqual(finished.returncode, 0, finished.stderr)
            passed = json.loads(
                (attempt_two / "result.json").read_text(encoding="utf-8")
            )
            self.assertTrue(passed["passed"])
            self.assertEqual(passed["attempt"], 2)
            self.assertEqual(
                passed["workspace_state"],
                "PATCH_RETAINED",
            )
            passing_patch = (
                attempt_two / "candidate.patch"
            ).read_text(encoding="utf-8")
            self.assertIn("return x + 1", passing_patch)
            self.assertNotIn("return x * 2", passing_patch)
            self.assertEqual(
                git(worktree, "rev-parse", "HEAD").stdout.strip(),
                baseline,
            )
            self.assertEqual(
                git(worktree, "branch", "--show-current").stdout.strip(),
                branch,
            )
            self.assertEqual(
                git(worktree, "diff", "--name-only").stdout.splitlines(),
                [ALLOWED_PATHS[0]],
            )

    def test_fifth_failure_restores_baseline_and_sixth_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree, baseline = create_test_repository(workspace)
            spec = workspace / "migration-spec.md"
            write_spec(spec, baseline)
            previous_result = assess_baseline(
                workspace,
                spec,
                worktree,
                passed=False,
            )

            for attempt in range(1, 6):
                run_dir = workspace / "runs" / f"repair-{attempt:03d}"
                started = guard_command(
                    "start-attempt",
                    spec,
                    run_dir,
                    worktree,
                    "--attempt",
                    str(attempt),
                    "--hypothesis",
                    f"candidate {attempt}",
                    "--previous-result",
                    str(previous_result),
                )
                self.assertEqual(started.returncode, 0, started.stderr)
                (worktree / ALLOWED_PATHS[0]).write_text(
                    "def swiglu(x):\n"
                    f"    return x + {attempt}\n",
                    encoding="utf-8",
                )
                recorded = record_candidate(spec, run_dir, worktree)
                self.assertEqual(recorded.returncode, 0, recorded.stderr)
                replay(spec, run_dir / "replay", passed=False)
                finished = finish_candidate(
                    spec,
                    run_dir,
                    worktree,
                )
                self.assertEqual(
                    finished.returncode,
                    0,
                    finished.stderr,
                )
                result = json.loads(
                    (run_dir / "result.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    result["attempt_limit_reached"],
                    attempt == 5,
                )
                self.assertEqual(
                    git(worktree, "status", "--porcelain").stdout,
                    "",
                )
                previous_result = run_dir / "result.json"

            sixth = workspace / "runs" / "repair-006"
            rejected = guard_command(
                "start-attempt",
                spec,
                sixth,
                worktree,
                "--attempt",
                "6",
                "--hypothesis",
                "forbidden sixth attempt",
                "--previous-result",
                str(previous_result),
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("max_repair_attempts", rejected.stderr)
            self.assertFalse(sixth.exists())

    def test_fifth_pass_retains_patch_without_marking_limit_reached(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree, baseline = create_test_repository(workspace)
            spec = workspace / "migration-spec.md"
            write_spec(spec, baseline)
            previous_result = assess_baseline(
                workspace,
                spec,
                worktree,
                passed=False,
            )

            for number in range(1, 5):
                failed_run = (
                    workspace / "runs" / f"repair-{number:03d}"
                )
                started = guard_command(
                    "start-attempt",
                    spec,
                    failed_run,
                    worktree,
                    "--attempt",
                    str(number),
                    "--hypothesis",
                    f"candidate {number} still mismatches",
                    "--previous-result",
                    str(previous_result),
                )
                self.assertEqual(started.returncode, 0, started.stderr)
                (worktree / ALLOWED_PATHS[0]).write_text(
                    "def swiglu(x):\n"
                    f"    return x + {number}\n",
                    encoding="utf-8",
                )
                recorded = record_candidate(spec, failed_run, worktree)
                self.assertEqual(recorded.returncode, 0, recorded.stderr)
                replay(spec, failed_run / "replay", passed=False)
                finished = finish_candidate(
                    spec,
                    failed_run,
                    worktree,
                )
                self.assertEqual(finished.returncode, 0, finished.stderr)
                previous_result = failed_run / "result.json"

            attempt = workspace / "runs" / "repair-005"

            started = guard_command(
                "start-attempt",
                spec,
                attempt,
                worktree,
                "--attempt",
                "5",
                "--hypothesis",
                "the fifth bounded candidate fixes the mismatch",
                "--previous-result",
                str(previous_result),
            )
            self.assertEqual(started.returncode, 0, started.stderr)
            (worktree / ALLOWED_PATHS[0]).write_text(
                "def swiglu(x):\n"
                "    return x + 5\n",
                encoding="utf-8",
            )
            recorded = record_candidate(spec, attempt, worktree)
            self.assertEqual(recorded.returncode, 0, recorded.stderr)
            replayed = replay(spec, attempt / "replay", passed=True)
            self.assertEqual(replayed.returncode, 0, replayed.stderr)
            finished = finish_candidate(
                spec,
                attempt,
                worktree,
            )
            self.assertEqual(finished.returncode, 0, finished.stderr)

            result = json.loads(
                (attempt / "result.json").read_text(encoding="utf-8")
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["attempt"], 5)
            self.assertFalse(result["attempt_limit_reached"])
            self.assertEqual(result["workspace_state"], "PATCH_RETAINED")
            self.assertEqual(
                git(worktree, "diff", "--name-only").stdout.splitlines(),
                [ALLOWED_PATHS[0]],
            )
            self.assertEqual(
                git(worktree, "rev-parse", "HEAD").stdout.strip(),
                baseline,
            )

    def test_attempt_history_rejects_restarting_at_one(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree, baseline = create_test_repository(workspace)
            spec = workspace / "migration-spec.md"
            write_spec(spec, baseline)
            previous_result = assess_baseline(
                workspace,
                spec,
                worktree,
                passed=False,
            )

            first = guard_command(
                "start-attempt",
                spec,
                workspace / "runs" / "repair-001",
                worktree,
                "--attempt",
                "1",
                "--hypothesis",
                "first bounded candidate",
                "--previous-result",
                str(previous_result),
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            second_replay = workspace / "runs" / "baseline-replay-again"
            replayed = replay(spec, second_replay, passed=False)
            self.assertEqual(replayed.returncode, 0, replayed.stderr)
            second_assessment = (
                workspace / "runs" / "baseline-assessment-again"
            )
            assessed = guard_command(
                "assess-baseline",
                spec,
                second_assessment,
                worktree,
                "--replay-result",
                str(second_replay / "result.json"),
                allow_synthetic_replay=True,
            )
            self.assertEqual(assessed.returncode, 0, assessed.stderr)
            repeated = guard_command(
                "start-attempt",
                spec,
                workspace / "runs" / "repair-001-again",
                worktree,
                "--attempt",
                "1",
                "--hypothesis",
                "incorrectly restart the counter",
                "--previous-result",
                str(second_assessment / "result.json"),
            )
            self.assertEqual(repeated.returncode, 2)
            self.assertIn("already claimed", repeated.stderr)
            self.assertFalse(
                (workspace / "runs" / "repair-001-again").exists()
            )

    def test_finish_rejects_patch_changed_after_recording(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree, baseline = create_test_repository(workspace)
            spec = workspace / "migration-spec.md"
            write_spec(spec, baseline)
            previous_result = assess_baseline(
                workspace,
                spec,
                worktree,
                passed=False,
            )
            attempt = workspace / "runs" / "repair-001"
            started = guard_command(
                "start-attempt",
                spec,
                attempt,
                worktree,
                "--attempt",
                "1",
                "--hypothesis",
                "record one exact candidate",
                "--previous-result",
                str(previous_result),
            )
            self.assertEqual(started.returncode, 0, started.stderr)
            operator = worktree / ALLOWED_PATHS[0]
            operator.write_text(
                "def swiglu(x):\n"
                "    return x + 1\n",
                encoding="utf-8",
            )
            recorded = record_candidate(spec, attempt, worktree)
            self.assertEqual(recorded.returncode, 0, recorded.stderr)
            replay(spec, attempt / "replay", passed=True)
            operator.write_text(
                "def swiglu(x):\n"
                "    return x + 2\n",
                encoding="utf-8",
            )

            rejected = finish_candidate(spec, attempt, worktree)
            self.assertEqual(rejected.returncode, 2)
            self.assertIn(
                "does not match recorded candidate.patch",
                rejected.stderr,
            )
            self.assertFalse((attempt / "outcome.json").exists())
            self.assertFalse((attempt / "result.json").exists())
            self.assertIn(
                "return x + 2",
                operator.read_text(encoding="utf-8"),
            )

    def test_candidate_must_be_recorded_before_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree, baseline = create_test_repository(workspace)
            spec = workspace / "migration-spec.md"
            write_spec(spec, baseline)
            previous_result = assess_baseline(
                workspace,
                spec,
                worktree,
                passed=False,
            )
            attempt = workspace / "runs" / "repair-001"
            started = guard_command(
                "start-attempt",
                spec,
                attempt,
                worktree,
                "--attempt",
                "1",
                "--hypothesis",
                "record this candidate before replay",
                "--previous-result",
                str(previous_result),
            )
            self.assertEqual(started.returncode, 0, started.stderr)
            (worktree / ALLOWED_PATHS[0]).write_text(
                "def swiglu(x):\n"
                "    return x + 1\n",
                encoding="utf-8",
            )
            replayed = replay(spec, attempt / "replay", passed=True)
            self.assertEqual(replayed.returncode, 0, replayed.stderr)

            rejected = record_candidate(spec, attempt, worktree)
            self.assertEqual(rejected.returncode, 2)
            self.assertIn(
                "before creating the replay Run",
                rejected.stderr,
            )
            self.assertFalse((attempt / "candidate.patch").exists())

    def test_production_spec_rejects_synthetic_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree, baseline = create_test_repository(workspace)
            spec = workspace / "migration-spec.md"
            write_spec(
                spec,
                baseline,
                spec_id="step3p7-flash-p800-demo",
            )
            replay_run = workspace / "runs" / "baseline-replay"
            replayed = replay(spec, replay_run, passed=False)
            self.assertEqual(replayed.returncode, 0, replayed.stderr)

            rejected = guard_command(
                "assess-baseline",
                spec,
                workspace / "runs" / "baseline-assessment",
                worktree,
                "--replay-result",
                str(replay_run / "result.json"),
                allow_synthetic_replay=True,
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn(
                "synthetic replay is test-only",
                rejected.stderr,
            )
            self.assertFalse(
                (workspace / "runs" / "baseline-assessment").exists()
            )

    def test_production_spec_rejects_forged_kernel_replay_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree, baseline = create_test_repository(workspace)
            spec = workspace / "migration-spec.md"
            write_spec(
                spec,
                baseline,
                spec_id="step3p7-flash-p800-demo",
            )
            binding_run = workspace / "runs" / "binding"
            replayed = replay(spec, binding_run, passed=True)
            self.assertEqual(replayed.returncode, 0, replayed.stderr)
            binding = json.loads(
                (binding_run / "result.json").read_text(encoding="utf-8")
            )["spec_binding"]

            forged_run = workspace / "runs" / "forged-kernel-replay"
            forged_run.mkdir()
            (forged_run / "replay-config.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            (forged_run / "worker-result.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            (forged_run / "replay.log").write_text(
                "execution_site=p800\n"
                "invocation_target=kunlun_ops.swiglu\n"
                "returncode=0\n",
                encoding="utf-8",
            )
            (forged_run / "result.json").write_text(
                json.dumps(
                    {
                        "tool": "replay_compare.py",
                        "action": "kernel-replay",
                        "spec_binding": binding,
                        "passed": True,
                        "execution_site": "p800",
                        "invocation_target": "kunlun_ops.swiglu",
                        "actual_tensors_saved": False,
                        "evidence": [
                            "replay-config.json",
                            "worker-result.json",
                            "replay.log",
                        ],
                    }
                ),
                encoding="utf-8",
            )

            rejected = guard_command(
                "assess-baseline",
                spec,
                workspace / "runs" / "baseline-assessment",
                worktree,
                "--replay-result",
                str(forged_run / "result.json"),
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("golden_run is invalid", rejected.stderr)
            self.assertFalse(
                (workspace / "runs" / "baseline-assessment").exists()
            )

    def test_unknown_changes_stop_without_overwriting_user_work(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree, baseline = create_test_repository(workspace)
            spec = workspace / "migration-spec.md"
            write_spec(spec, baseline)
            readme = worktree / "README.md"
            readme.write_text("user change before baseline\n", encoding="utf-8")

            rejected = guard_command(
                "check-baseline",
                spec,
                workspace / "runs" / "baseline-check",
                worktree,
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("unknown workspace changes", rejected.stderr)
            self.assertEqual(
                readme.read_text(encoding="utf-8"),
                "user change before baseline\n",
            )
            self.assertFalse(
                (workspace / "runs" / "baseline-check").exists()
            )

            git(worktree, "restore", "README.md")
            previous_result = assess_baseline(
                workspace,
                spec,
                worktree,
                passed=False,
            )
            attempt = workspace / "runs" / "repair-001"
            started = guard_command(
                "start-attempt",
                spec,
                attempt,
                worktree,
                "--attempt",
                "1",
                "--hypothesis",
                "only the selected fallback changes",
                "--previous-result",
                str(previous_result),
            )
            self.assertEqual(started.returncode, 0, started.stderr)
            (worktree / ALLOWED_PATHS[0]).write_text(
                "def swiglu(x):\n"
                "    return x + 1\n",
                encoding="utf-8",
            )
            recorded = record_candidate(spec, attempt, worktree)
            self.assertEqual(recorded.returncode, 0, recorded.stderr)
            readme.write_text(
                "unrelated user change during attempt\n",
                encoding="utf-8",
            )
            replay(spec, attempt / "replay", passed=False)
            rejected = finish_candidate(
                spec,
                attempt,
                worktree,
            )
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("unknown workspace changes", rejected.stderr)
            self.assertFalse((attempt / "result.json").exists())
            self.assertTrue((attempt / "candidate.patch").is_file())
            self.assertIn(
                "return x + 1",
                (worktree / ALLOWED_PATHS[0]).read_text(encoding="utf-8"),
            )
            self.assertEqual(
                readme.read_text(encoding="utf-8"),
                "unrelated user change during attempt\n",
            )

    def test_failed_attempt_patch_includes_and_removes_new_allowed_test(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree, baseline = create_test_repository(workspace)
            spec = workspace / "migration-spec.md"
            write_spec(spec, baseline)
            new_test = "tests/test_new_swiglu.py"
            allowed_paths = (ALLOWED_PATHS[0], new_test)
            attempt = workspace / "runs" / "repair-001"
            previous_result = assess_baseline(
                workspace,
                spec,
                worktree,
                passed=False,
                allowed_paths=allowed_paths,
            )

            started = guard_command(
                "start-attempt",
                spec,
                attempt,
                worktree,
                "--attempt",
                "1",
                "--hypothesis",
                "add one focused regression with the fallback",
                "--previous-result",
                str(previous_result),
                allowed_paths=allowed_paths,
            )
            self.assertEqual(started.returncode, 0, started.stderr)
            (worktree / ALLOWED_PATHS[0]).write_text(
                "def swiglu(x):\n"
                "    return x + 1\n",
                encoding="utf-8",
            )
            (worktree / new_test).write_text(
                "def test_new_swiglu():\n"
                "    assert True\n",
                encoding="utf-8",
            )
            recorded = record_candidate(spec, attempt, worktree)
            self.assertEqual(recorded.returncode, 0, recorded.stderr)
            replay(spec, attempt / "replay", passed=False)
            finished = finish_candidate(
                spec,
                attempt,
                worktree,
            )
            self.assertEqual(finished.returncode, 0, finished.stderr)
            self.assertFalse((worktree / new_test).exists())
            patch = (attempt / "candidate.patch").read_text(
                encoding="utf-8"
            )
            self.assertIn(new_test, patch)
            applicable = run_command(
                "git",
                "apply",
                "--check",
                str(attempt / "candidate.patch"),
                cwd=worktree,
            )
            self.assertEqual(
                applicable.returncode,
                0,
                applicable.stderr,
            )


if __name__ == "__main__":
    unittest.main()
