from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = (
    ROOT
    / "model-adaptation"
    / "references"
    / "formal-cuda-capture.md"
)


class Ticket15FormalCudaCaptureRunbookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runbook = RUNBOOK.read_text(encoding="utf-8")

    def test_runbook_binds_the_only_formal_session_to_current_evidence(
        self,
    ) -> None:
        for required in (
            "runs/spec-binding-004/result.json",
            "runs/scan-006/result.json",
            "runs/adapter-002/result.json",
            "runs/cuda-preflight-r5-002/result.json",
            "49e384ce9d304648e9959666ecb8ce8cd98d0deb",
            "stepfun-ai/Step-3.7-Flash",
            "5f6244077ac62e04eec3f320501ff8c2b293373a",
            "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe."
            "_swiglu_silu_clamp_mul",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.runbook)

        self.assertIn("test ! -e \"$SESSION_RUN\"", self.runbook)
        self.assertIn("test ! -e \"$GOLDEN_RUN\"", self.runbook)
        self.assertIn("set -euo pipefail", self.runbook)
        self.assertIn('WORKTREE_STATUS="$(git status --short)"', self.runbook)
        self.assertIn('test -z "$WORKTREE_STATUS"', self.runbook)
        self.assertIn(
            'REMOTE_EVIDENCE="$(git ls-remote --heads origin '
            '"refs/heads/$EVIDENCE_BRANCH")"',
            self.runbook,
        )
        self.assertIn('test -z "$REMOTE_EVIDENCE"', self.runbook)
        self.assertIn('"consumes_capture_session": True', self.runbook)
        self.assertIn('"status": "ACTIVE"', self.runbook)
        self.assertNotIn('"status": "RESERVED"', self.runbook)
        self.assertIn("不得创建第二个正式 Session", self.runbook)

    def test_launch_is_tp8_bf16_target_only_eager_without_backend_overrides(
        self,
    ) -> None:
        for required in (
            "--model-path \"$MODEL_ID\"",
            "--revision \"$MODEL_REVISION\"",
            "--tp-size 8",
            "--dtype bfloat16",
            "--cuda-graph-backend-decode disabled",
            "--cuda-graph-backend-prefill disabled",
            "MODEL_ADAPTATION_CAPTURE_CONFIG",
            "SGLANG_PLUGINS=model_adaptation_capture",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.runbook)

        for forbidden in (
            "--quantization",
            "--speculative-algorithm",
            "--speculative-draft-model-path",
            "--attention-backend",
            "--moe-runner-backend",
            "--moe-a2a-backend",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, self.runbook)

    def test_recording_precedes_self_replay_and_stops_before_bundle_build(
        self,
    ) -> None:
        prepare = self.runbook.index("--mode prepare")
        reservation_branch = self.runbook.index(
            'git switch -c "$EVIDENCE_BRANCH"'
        )
        session_start = self.runbook.index('"consumes_capture_session": True')
        reservation_commit = self.runbook.index(
            'git commit -m "evidence: reserve formal CUDA capture session"'
        )
        reservation_push = self.runbook.index(
            "git push --porcelain -u origin"
        )
        launch = self.runbook.index("python -m sglang.launch_server")
        request = self.runbook.index('\"/generate\"')
        record = self.runbook.index("--mode record-samples")
        replay = self.runbook.index("--mode kernel-replay")
        upload = self.runbook.index(
            'git commit -m "evidence: add formal CUDA golden capture"'
        )

        self.assertLess(prepare, reservation_branch)
        self.assertLess(reservation_branch, session_start)
        self.assertLess(session_start, reservation_commit)
        self.assertLess(reservation_commit, reservation_push)
        self.assertLess(reservation_push, launch)
        self.assertLess(launch, request)
        self.assertLess(request, record)
        self.assertLess(record, replay)
        self.assertLess(replay, upload)
        self.assertIn(
            '--run-dir "$GOLDEN_RUN/self-replay"',
            self.runbook,
        )
        self.assertIn('1 <= state["saved_shape_count"] <= 3', self.runbook)
        self.assertIn('"tp_rank"] == 0', self.runbook)
        self.assertIn("参数 Tensor 列表为空", self.runbook)
        self.assertNotIn("--mode build", self.runbook)
        self.assertIn("不要自行编写或修改下一状态 Spec", self.runbook)

    def test_success_and_failure_append_to_the_reserved_branch(self) -> None:
        self.assertIn(
            "export EVIDENCE_BRANCH=evidence/cuda-formal-capture-r5-001",
            self.runbook,
        )
        self.assertIn(
            'git commit -m "evidence: record failed formal CUDA capture"',
            self.runbook,
        )
        self.assertIn('if test -e "$RECORD_RUN"; then', self.runbook)
        self.assertGreaterEqual(
            self.runbook.count('git push origin "$EVIDENCE_BRANCH"'),
            2,
        )
        self.assertNotIn(
            "evidence/cuda-formal-capture-r5-001-failed",
            self.runbook,
        )
        self.assertIn('"reservation_id": uuid.uuid4().hex', self.runbook)
        self.assertIn("[new branch]", self.runbook)
        self.assertIn("[up to date]", self.runbook)
        self.assertIn("LC_ALL=C git push --porcelain", self.runbook)
        self.assertIn(
            'grep -Fx "$EXPECTED_RESERVATION_LINE"',
            self.runbook,
        )

    def test_git_reservation_accepts_only_the_first_remote_branch_creation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote = root / "remote.git"
            checkout = root / "checkout"
            subprocess.run(
                ["git", "init", "--quiet", "--bare", str(remote)],
                check=True,
            )
            subprocess.run(
                ["git", "init", "--quiet", str(checkout)],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(checkout), "config", "user.email",
                 "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(checkout), "config", "user.name",
                 "Ticket 15 Test"],
                check=True,
            )
            (checkout / "session-start.json").write_text(
                '{"reservation_id":"unique"}\n',
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "-C", str(checkout), "add", "session-start.json"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(checkout), "commit", "--quiet",
                 "-m", "reserve"],
                check=True,
            )
            branch = "evidence/test-formal-capture"
            subprocess.run(
                ["git", "-C", str(checkout), "branch", branch],
                check=True,
            )
            command = [
                "git",
                "-C",
                str(checkout),
                "push",
                "--porcelain",
                str(remote),
                f"{branch}:{branch}",
            ]
            expected = (
                f"*\trefs/heads/{branch}:refs/heads/{branch}\t[new branch]"
            )
            first = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            second = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn(expected, first.stdout.splitlines())
            self.assertNotIn(expected, second.stdout.splitlines())
            self.assertIn("[up to date]", second.stdout)

    def test_runtime_metadata_is_filtered_before_it_is_uploaded(self) -> None:
        self.assertIn('curl -fsS "$BASE_URL/server_info"', self.runbook)
        self.assertIn("runtime-profile.json", self.runbook)
        self.assertIn("原始 `server_info`", self.runbook)
        self.assertIn("不得提交", self.runbook)
        for field in (
            '"attention_backend"',
            '"moe_runner_backend"',
            '"cuda_graph_backend_decode"',
            '"cuda_graph_backend_prefill"',
        ):
            with self.subTest(field=field):
                self.assertIn(field, self.runbook)
        self.assertIn('"moe_runner_backend_argument"', self.runbook)
        self.assertIn('"actual_backends"', self.runbook)
        self.assertIn('"moe_runner": "triton"', self.runbook)
        self.assertIn('"kind": "captured-hook-plus-fixed-source"', self.runbook)
        self.assertNotIn('"schema": "cuda-runtime-profile/v1"', self.runbook)

    def test_runbook_marks_the_sealed_capture_as_complete(self) -> None:
        self.assertIn("采集已经完成，不要再次执行", self.runbook)
        self.assertIn("不得重启模型", self.runbook)
        self.assertIn("runs/cuda-golden-r5-001", self.runbook)
        self.assertIn(
            "model-adaptation/references/formal-cuda-handoff.md",
            self.runbook,
        )
        self.assertIn("当前阶段不要构建 bundle", self.runbook)


if __name__ == "__main__":
    unittest.main()
