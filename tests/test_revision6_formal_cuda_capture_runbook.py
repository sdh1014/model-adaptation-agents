from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "migration-spec.md"
SKILL = ROOT / "model-adaptation" / "SKILL.md"
DESIGN = ROOT / "outputs" / "step3p7-p800-migration-tool-design.md"
MAP = ROOT / ".scratch" / "step3p7-p800-migration-tool" / "map.md"
RUNBOOK = (
    ROOT
    / "model-adaptation"
    / "references"
    / "formal-cuda-capture-r6.md"
)
SOURCE_RUN = ROOT / "runs" / "formal-cuda-runbook-r6-001" / "result.json"


class Revision6FormalCudaCaptureRunbookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runbook = RUNBOOK.read_text(encoding="utf-8")
        cls.scan = json.loads(
            (ROOT / "runs" / "scan-007" / "result.json").read_text(
                encoding="utf-8"
            )
        )

    def test_environment_preflight_is_the_first_cuda_action(self) -> None:
        runbook = self.runbook

        self.assertIn("Contract revision 6", runbook)
        self.assertIn("runs/scan-007/result.json", runbook)
        self.assertIn(
            "runs/cuda-environment-preflight-r6-001",
            runbook,
        )
        preflight = runbook.index("--mode preflight")
        prepare = runbook.index("--mode prepare-session")
        reservation = runbook.index("session-start.json")
        launch = runbook.index("python -m sglang.launch_server")
        self.assertLess(preflight, prepare)
        self.assertLess(prepare, reservation)
        self.assertLess(reservation, launch)
        preflight_command = runbook[preflight - 240 : preflight + 240]
        self.assertNotIn("--scan-result", preflight_command)
        self.assertNotIn("--operator-id", preflight_command)
        self.assertNotIn("--mode capture-adapter-validation", runbook)
        self.assertNotIn("runs/cuda-preflight-r6-002", runbook)

    def test_capture_and_requests_are_driven_by_the_session_config(
        self,
    ) -> None:
        runbook = self.runbook

        self.assertIn("--mode prepare-session", runbook)
        self.assertIn('--scan-result "$SCAN_RESULT"', runbook)
        self.assertIn('config["operators"]', runbook)
        self.assertIn('config["requests"]', runbook)
        self.assertIn("operator-map.tsv", runbook)
        self.assertIn("request-01.json", runbook)
        self.assertIn("request-02.json", runbook)
        self.assertIn('"text-only"', runbook)
        self.assertIn('"single-image"', runbook)
        for item in self.scan["gap_queue"]:
            self.assertNotIn(
                f"export OPERATOR_ID={item['operator_id']}",
                runbook,
            )

    def test_launch_is_tp8_bf16_eager_without_backend_or_quantization(
        self,
    ) -> None:
        runbook = self.runbook

        for required in (
            "--model-path \"$MODEL_ID\"",
            "--revision \"$MODEL_REVISION\"",
            "--trust-remote-code",
            "--tp-size 8",
            "--dtype bfloat16",
            "--cuda-graph-backend-decode disabled",
            "--cuda-graph-backend-prefill disabled",
        ):
            with self.subTest(required=required):
                self.assertIn(required, runbook)
        for forbidden in (
            "--quantization",
            "--speculative-algorithm",
            "--speculative-draft-model-path",
            "--attention-backend",
            "--moe-runner-backend",
            "--moe-a2a-backend",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, runbook)

    def test_one_python_interpreter_runs_tools_and_model(self) -> None:
        runbook = self.runbook

        self.assertNotIn("python3 ", runbook)
        self.assertIn(
            "python -m pip install -e ./model-adaptation",
            runbook,
        )
        self.assertIn(
            "python model-adaptation/scripts/capture_golden.py",
            runbook,
        )
        self.assertIn("python -m sglang.launch_server", runbook)
        self.assertIn(
            "python model-adaptation/scripts/replay_compare.py",
            runbook,
        )
        self.assertIn(
            "python model-adaptation/scripts/handoff_bundle.py",
            runbook,
        )

    def test_reservation_is_unique_but_does_not_fake_capture(self) -> None:
        runbook = self.runbook

        self.assertIn(
            "export EVIDENCE_BRANCH=evidence/cuda-formal-capture-r6-001",
            runbook,
        )
        self.assertIn('"reservation_id": uuid.uuid4().hex', runbook)
        self.assertIn('"consumes_capture_session": False', runbook)
        self.assertIn('"capture_session_consumed": False', runbook)
        self.assertIn("LC_ALL=C git push --porcelain", runbook)
        self.assertIn("[new branch]", runbook)
        self.assertIn("[up to date]", runbook)
        self.assertIn("不得创建第二个", runbook)
        self.assertIn("同一个 reservation", runbook)

    def test_every_golden_is_recorded_then_self_replayed_dynamically(
        self,
    ) -> None:
        runbook = self.runbook

        record = runbook.index("--mode record-samples")
        replay = runbook.index("--mode kernel-replay")
        formal = runbook.index("formal-result.json")
        self.assertLess(record, replay)
        self.assertLess(replay, formal)
        self.assertIn('while IFS=$\'\\t\' read -r', runbook)
        self.assertIn('--scan-result "$SCAN_RESULT"', runbook)
        self.assertIn('--operator-id "$operator_id"', runbook)
        self.assertIn('--golden-run "$golden_run"', runbook)
        self.assertIn('"capture_process_id"', runbook)
        self.assertIn(
            '"action": "complete_formal_cuda_capture_session"',
            runbook,
        )
        self.assertIn('"capture_status": "SEALED"', runbook)
        self.assertIn('"formal_session_consumed": True', runbook)

    def test_bundle_is_built_and_verified_before_spec_and_final_push(
        self,
    ) -> None:
        runbook = self.runbook

        formal = runbook.index("formal-result.json")
        candidate = runbook.index('Path(os.environ["BUNDLE_SPEC"])')
        build = runbook.index("--mode build")
        verify = runbook.index("--mode verify")
        replace_spec = runbook.index("mv migration-spec.md.next migration-spec.md")
        final_commit = runbook.index(
            'git commit -m "evidence: add revision 6 CUDA capture and bundle"'
        )
        self.assertLess(formal, candidate)
        self.assertLess(candidate, build)
        self.assertLess(build, verify)
        self.assertLess(verify, replace_spec)
        self.assertLess(replace_spec, final_commit)
        self.assertIn('"handoff-manifest/v2"', runbook)
        self.assertIn("- `state_revision`: `49`", runbook)
        self.assertIn("- `status`: `WAITING`", runbook)
        self.assertIn("- `phase`: `HANDOFF`", runbook)
        self.assertIn('"bundle_status": "VALID"', runbook)
        self.assertIn('"manifest_verified"] is True', runbook)

    def test_size_gate_covers_every_run_that_will_be_pushed(self) -> None:
        runbook = self.runbook

        size_gate = runbook[
            runbook.index("先确认 GitHub 可接收所有待回传文件") :
        ]
        for path in (
            '"$SESSION_RUN"',
            '"$RECORD_ROOT"',
            '"$PLAN_RUN"',
            '"$BUILD_RUN"',
            '"$VERIFY_RUN"',
        ):
            with self.subTest(path=path):
                self.assertIn(path, size_gate)
        self.assertIn("-type f", size_gate)
        self.assertIn("-size +95M", size_gate)

    def test_failure_boundary_separates_environment_and_consumed_session(
        self,
    ) -> None:
        runbook = self.runbook

        self.assertIn("环境失败", runbook)
        self.assertIn("不创建正式 Session", runbook)
        self.assertIn("saved_shape_count", runbook)
        self.assertIn("尚未产生任何 Golden Sample", runbook)
        self.assertIn("不得新建 Run 或 evidence 分支", runbook)
        self.assertIn("已有 Golden Sample", runbook)
        self.assertIn("不得重启模型", runbook)

    def test_all_bash_blocks_parse(self) -> None:
        blocks = re.findall(
            r"```bash\n(.*?)\n```",
            self.runbook,
            flags=re.DOTALL,
        )
        self.assertGreater(len(blocks), 0)
        for index, block in enumerate(blocks, start=1):
            with self.subTest(block=index):
                completed = subprocess.run(
                    ["bash", "-n"],
                    input=block,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(
                    completed.returncode,
                    0,
                    completed.stderr,
                )

    def test_all_embedded_python_blocks_compile(self) -> None:
        blocks = re.findall(
            r"python(?:3)? - <<'PY'\n(.*?)\nPY",
            self.runbook,
            flags=re.DOTALL,
        )
        self.assertGreater(len(blocks), 0)
        for index, block in enumerate(blocks, start=1):
            with self.subTest(block=index):
                compile(block, f"runbook-python-{index}", "exec")

    def test_candidate_spec_transition_executes_against_revision_48(
        self,
    ) -> None:
        blocks = re.findall(
            r"python(?:3)? - <<'PY'\n(.*?)\nPY",
            self.runbook,
            flags=re.DOTALL,
        )
        transition = next(
            block
            for block in blocks
            if "prepare_revision_6_handoff_spec" in block
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "migration-spec.md").write_bytes(SPEC.read_bytes())
            session = workspace / "runs" / "cuda-formal-session-r6-001"
            session.mkdir(parents=True)
            goldens = []
            for ordinal, item in enumerate(
                self.scan["gap_queue"],
                start=1,
            ):
                golden = session / "operators" / f"{ordinal:02d}"
                golden.mkdir(parents=True)
                (golden / "capture-state.json").write_text(
                    json.dumps({"saved_shape_count": min(ordinal, 3)}),
                    encoding="utf-8",
                )
                goldens.append(
                    {
                        "operator_id": item["operator_id"],
                        "golden_run": str(golden),
                    }
                )
            (session / "formal-result.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "capture_status": "SEALED",
                        "spec_binding": self.scan["spec_binding"],
                        "goldens": goldens,
                    }
                ),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "SESSION_RUN": str(session),
                    "ENV_RUN": "runs/cuda-environment-preflight-r6-001",
                    "PLAN_RUN": "runs/handoff-plan-r6-001",
                    "BUNDLE_SPEC": (
                        "runs/handoff-plan-r6-001/migration-spec.md"
                    ),
                    "BUILD_RUN": "runs/handoff-build-r6-001",
                    "VERIFY_RUN": "runs/handoff-verify-cuda-r6-001",
                }
            )
            completed = subprocess.run(
                ["python3", "-c", transition],
                cwd=workspace,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stderr,
            )
            candidate = (
                workspace
                / "runs"
                / "handoff-plan-r6-001"
                / "migration-spec.md"
            ).read_text(encoding="utf-8")
            self.assertIn("- `state_revision`: `49`", candidate)
            self.assertIn("- `status`: `WAITING`", candidate)
            self.assertIn("- `phase`: `HANDOFF`", candidate)
            self.assertIn("- `bundle_status`: `VALID`", candidate)
            for item in self.scan["gap_queue"]:
                self.assertNotIn(
                    f"| `{item['operator_id']}` | "
                    "`CAPTURE_REQUIRED` | `null` |",
                    candidate,
                )

    def test_source_handoff_advances_working_state_to_cuda(self) -> None:
        spec = SPEC.read_text(encoding="utf-8")
        skill = SKILL.read_text(encoding="utf-8")
        design = DESIGN.read_text(encoding="utf-8")
        map_text = MAP.read_text(encoding="utf-8")

        self.assertIn("- `state_revision`: `48`", spec)
        self.assertIn("- `execution_site`: `CUDA`", spec)
        self.assertIn(
            "- `last_completed_action`: "
            "`revision_6_formal_cuda_capture_runbook_generated_and_reviewed`",
            spec,
        )
        self.assertIn(
            "- `last_run`: `runs/formal-cuda-runbook-r6-001`",
            spec,
        )
        self.assertIn(
            "- `next_action`: `在 CUDA 机器严格执行 "
            "model-adaptation/references/formal-cuda-capture-r6.md；"
            "先完成环境 preflight，通过后在同一 reservation 内完成唯一"
            "正式 Session、全部 Golden self-replay 和 Handoff Bundle "
            "build/verify`",
            spec,
        )
        self.assertIn(
            "[references/formal-cuda-capture-r6.md]"
            "(references/formal-cuda-capture-r6.md)",
            skill,
        )
        self.assertIn("runbook 已生成并完成 SOURCE 审查", design)
        self.assertIn("下一步转到 CUDA 机器", design)
        self.assertIn("runbook 已生成并完成 SOURCE 审查", map_text)

        result = json.loads(SOURCE_RUN.read_text(encoding="utf-8"))
        self.assertTrue(result["passed"])
        self.assertEqual(result["execution_site"], "SOURCE")
        self.assertFalse(result["consumes_capture_session"])
        self.assertFalse(result["runtime_execution"]["cuda_preflight"])
        self.assertFalse(result["runtime_execution"]["capture_session"])
        self.assertFalse(result["runtime_execution"]["handoff_bundle"])
        self.assertEqual(
            result["runbook"],
            "model-adaptation/references/formal-cuda-capture-r6.md",
        )
        self.assertEqual(result["scan_result"], "runs/scan-007/result.json")
        for source in result["source_files"]:
            actual = hashlib.sha256(
                (ROOT / source["path"]).read_bytes()
            ).hexdigest()
            self.assertEqual(actual, source["sha256"], source["path"])


if __name__ == "__main__":
    unittest.main()
