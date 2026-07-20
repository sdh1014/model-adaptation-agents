from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "model-adaptation" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import capture_golden
import replay_compare
from _lib.spec_contract import load_contract_data, load_spec_binding
from model_adaptation_capture.contracts import (
    checkpoint_metadata,
    shape_id_for,
)


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
SOURCE_RUN = ROOT / "runs" / "formal-cuda-runbook-r6-002" / "result.json"


class Revision6FormalCudaCaptureRunbookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runbook = RUNBOOK.read_text(encoding="utf-8")
        cls.scan = json.loads(
            (ROOT / "runs" / "scan-007" / "result.json").read_text(
                encoding="utf-8"
            )
        )

    def _python_block(self, marker: str) -> str:
        return next(
            block
            for block in re.findall(
                r"python(?:3)? - <<'PY'\n(.*?)\nPY",
                self.runbook,
                flags=re.DOTALL,
            )
            if marker in block
        )

    def _run_reservation_transition(self, workspace: Path) -> None:
        session = workspace / "runs" / "cuda-formal-session-r6-001"
        session.mkdir(parents=True, exist_ok=True)
        (session / "session-start.json").write_text(
            json.dumps(
                {
                    "status": "ACTIVE",
                    "spec_binding": self.scan["spec_binding"],
                    "session_id": "cuda-formal-session-r6-001",
                    "reservation_id": "fixture-reservation",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        environment["SESSION_RUN"] = "runs/cuda-formal-session-r6-001"
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                self._python_block("reserve_revision_6_working_state"),
            ],
            cwd=workspace,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def _run_environment_failure_transition(
        self,
        workspace: Path,
    ) -> None:
        environment_run = (
            workspace
            / "runs"
            / "cuda-environment-preflight-r6-001"
        )
        environment_run.mkdir(parents=True)
        (environment_run / "result.json").write_text(
            json.dumps(
                {
                    "action": "environment-preflight",
                    "passed": False,
                    "preflight_status": "PREFLIGHT_FAILED",
                    "consumes_capture_session": False,
                    "spec_binding": self.scan["spec_binding"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        environment["ENV_RUN"] = (
            "runs/cuda-environment-preflight-r6-001"
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                self._python_block(
                    "record_revision_6_environment_failure"
                ),
            ],
            cwd=workspace,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

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

    def test_each_model_launch_uses_fresh_evidence_paths(self) -> None:
        runbook = self.runbook

        self.assertIn(
            'export LAUNCH_EVIDENCE_DIR="$SESSION_RUN/launch-$LAUNCH_STATE_REVISION"',
            runbook,
        )
        self.assertIn('mkdir "$LAUNCH_EVIDENCE_DIR"', runbook)
        self.assertIn(
            '>"$LAUNCH_EVIDENCE_DIR/server.log"',
            runbook,
        )
        self.assertNotIn('>"$SESSION_RUN/server.log"', runbook)
        self.assertIn(
            'launch / f"request-{index:02d}.json"',
            runbook,
        )
        self.assertIn(
            '"launch_evidence_dir": os.environ["LAUNCH_EVIDENCE_DIR"]',
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

    def test_replay_wrapper_accepts_every_revision6_cuda_operator(
        self,
    ) -> None:
        contract = load_contract_data(SPEC)
        binding = load_spec_binding(SPEC).as_result_dict()
        checkpoint = checkpoint_metadata(
            contract["checkpoint"]["id"],
            contract["checkpoint"]["config_digest"],
        )
        expected_revision = contract["source"]["sglang_revision"]

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            worktree = workspace / "sglang"
            worktree.mkdir()
            for ordinal, operator_id in enumerate(
                [
                    item["operator_id"]
                    for item in self.scan["gap_queue"]
                    if item["verdict"] == "CAPTURE_REQUIRED"
                ],
                start=1,
            ):
                with self.subTest(operator_id=operator_id):
                    operator = next(
                        item
                        for item in self.scan["operators"]
                        if item["operator_id"] == operator_id
                    )
                    plan = next(
                        item
                        for item in self.scan["capture_plan"]
                        if item["operator_id"] == operator_id
                    )
                    sample_fields = {
                        "inputs": plan["saved_inputs"],
                        "parameters": plan["saved_parameters"],
                        "non_tensor_args": plan["saved_non_tensor_args"],
                        "outputs": plan["saved_outputs"],
                    }
                    golden = workspace / f"golden-{ordinal:02d}"
                    samples = golden / "samples"
                    samples.mkdir(parents=True)
                    input_shapes = {
                        name: [ordinal, 4]
                        for name in sample_fields["inputs"]
                    }
                    tensor_metadata = {
                        "kind": "tensor",
                        "shape": [ordinal, 4],
                        "dtype": "bfloat16",
                        "layout": "strided",
                        "stride": [4, 1],
                    }
                    shape_id = shape_id_for(input_shapes)
                    signature = {
                        "operator_id": operator_id,
                        "model_path": "target",
                        "tensor_parallel_size": 8,
                        "tp_rank": 0,
                        "inputs": {
                            name: {
                                "kind": "tensor",
                                "shape": shape,
                                "dtype": "bfloat16",
                                "layout": "strided",
                                "stride": [4, 1],
                            }
                            for name, shape in input_shapes.items()
                        },
                        "parameters": {},
                        "non_tensor_args": {
                            name: (
                                None
                                if name == "sm_scale"
                                else False
                                if name == "is_causal"
                                else True
                                if name == "renormalize"
                                else 1.0
                            )
                            for name in sample_fields["non_tensor_args"]
                        },
                        "outputs": {
                            name: dict(tensor_metadata)
                            for name in sample_fields["outputs"]
                        },
                    }
                    signature["parameters"] = {
                        name: dict(tensor_metadata)
                        for name in sample_fields["parameters"]
                    }
                    sample_path = samples / f"{shape_id}.pt"
                    sample_path.write_bytes(f"sample-{ordinal}".encode())
                    session_config = (
                        workspace
                        / "runs"
                        / "cuda-formal-session-r6-001"
                        / "capture-config.json"
                    )
                    capture_config = {
                        "schema": "golden-capture-config/v1",
                        "session_config": str(session_config),
                        "adapter": capture_golden.ADAPTER_NAMES[operator_id],
                        "spec_binding": binding,
                        "operator_id": operator_id,
                        "activation_guard": operator["activation_guard"],
                        "model_path": "target",
                        "max_shapes": 3,
                        "tensor_parallel_size": 8,
                        "tp_rank": 0,
                        "serialization": "kernel-call-torch-save/v1",
                        "capture_device_type": "cuda",
                        "dtype": "bfloat16",
                        "checkpoint": checkpoint,
                        "precision_gate": contract["precision_gate"],
                        "run_dir": str(golden),
                        "hook_target": plan["hook_target"],
                        "boundary": operator["boundary"],
                        "sample_fields": sample_fields,
                        "replay": {
                            "mode": "standalone-kernel-call",
                            "cuda_target": plan["hook_target"],
                            "p800_target": (
                                "kunlun_ops.swiglu"
                                if ordinal == 1
                                else None
                            ),
                            "weights_in_golden_sample": False,
                        },
                        "source": {
                            "capture_module_path": "/fixed/source.py",
                            "capture_module_sha256": "0" * 64,
                        },
                    }
                    (golden / "capture-config.json").write_text(
                        json.dumps(capture_config),
                        encoding="utf-8",
                    )
                    state = {
                        "schema": "kernel-call-capture-state/v1",
                        "spec_binding": binding,
                        "operator_id": operator_id,
                        "tp_rank": 0,
                        "tensor_parallel_size": 8,
                        "checkpoint": checkpoint,
                        "loaded_checkpoint": {
                            "model_path": checkpoint["model_path"],
                            "revision": checkpoint["revision"],
                        },
                        "capture_session_config": str(session_config),
                        "capture_process_id": 1234,
                        "status": "ACTIVE",
                        "capture_closed": False,
                        "saved_shape_count": 1,
                        "repeated_call_count": 0,
                        "skipped_call_count": 0,
                        "samples": [
                            {
                                "shape_id": shape_id,
                                "file": f"samples/{shape_id}.pt",
                                "signature": signature,
                                "repeat_count": 0,
                            }
                        ],
                        "skipped_signatures": [],
                    }
                    (golden / "capture-state.json").write_text(
                        json.dumps(state),
                        encoding="utf-8",
                    )
                    sample_sidecar = {
                        "schema": "golden-sample-files/v1",
                        "spec_binding": binding,
                        "operator_id": operator_id,
                        "files": [
                            {
                                "shape_id": shape_id,
                                "path": f"samples/{shape_id}.pt",
                                "size": sample_path.stat().st_size,
                                "sha256": hashlib.sha256(
                                    sample_path.read_bytes()
                                ).hexdigest(),
                            }
                        ],
                    }
                    (golden / "sample-files.json").write_text(
                        json.dumps(sample_sidecar),
                        encoding="utf-8",
                    )

                    def worker(command, **kwargs):
                        del kwargs
                        config = json.loads(
                            Path(command[-1]).read_text(encoding="utf-8")
                        )
                        result = {
                            "schema": "kernel-call-replay-result/v1",
                            "spec_binding": binding,
                            "operator_id": operator_id,
                            "execution_site": "cuda",
                            "invocation_target": plan["hook_target"],
                            "tp_rank": 0,
                            "tensor_parallel_size": 8,
                            "precision_gate": contract["precision_gate"],
                            "sample_files_sha256": config[
                                "sample_files_sha256"
                            ],
                            "passed": True,
                            "checked_shape_count": 1,
                            "failed_shape_count": 0,
                            "checked_shapes": [
                                {"shape_id": shape_id, "passed": True}
                            ],
                            "errors": [],
                            "actual_tensors_saved": False,
                        }
                        Path(config["run_dir"], "worker-result.json").write_text(
                            json.dumps(result),
                            encoding="utf-8",
                        )
                        return subprocess.CompletedProcess(
                            command,
                            0,
                            stdout="",
                            stderr="",
                        )

                    replay_run = workspace / f"replay-{ordinal:02d}"
                    with patch.object(
                        replay_compare,
                        "resolve_git_revision",
                        return_value=expected_revision,
                    ), patch.object(
                        replay_compare.subprocess,
                        "run",
                        side_effect=worker,
                    ):
                        replay_compare.run_kernel_replay(
                            SPEC,
                            replay_run,
                            ROOT / "runs" / "scan-007" / "result.json",
                            operator_id,
                            golden,
                            execution_site="cuda",
                            sglang_worktree=worktree,
                        )
                    replay_config = json.loads(
                        (replay_run / "replay-config.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(
                        replay_config["invocation_target"],
                        plan["hook_target"],
                    )
                    self.assertEqual(
                        replay_config["sample_fields"],
                        sample_fields,
                    )
                    self.assertEqual(
                        replay_config["adapter"],
                        capture_golden.ADAPTER_NAMES[operator_id],
                    )

    def test_legacy_and_named_single_input_shape_ids_are_equivalent(
        self,
    ) -> None:
        self.assertEqual(
            shape_id_for([1, 4]),
            shape_id_for({"x": [1, 4]}),
        )

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
        self.assertIn('"target_revision"', runbook)
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
        self.assertIn('glob("operators/*/samples/*.pt")', runbook)
        self.assertIn("sample_files_on_disk", runbook)
        self.assertIn("archived_zero_sample_states", runbook)
        self.assertIn("尚未产生任何 Golden Sample", runbook)
        self.assertIn("不得新建 Run 或 evidence 分支", runbook)
        self.assertIn("已有 Golden Sample", runbook)
        self.assertIn("不得重启模型", runbook)
        self.assertIn('"status": "BLOCKED"', runbook)
        self.assertIn('"phase": "CUDA_CAPTURE"', runbook)

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

    def test_reservation_updates_working_state_before_first_push(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "migration-spec.md").write_bytes(SPEC.read_bytes())
            self._run_reservation_transition(workspace)
            spec = (workspace / "migration-spec.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("- `state_revision`: `50`", spec)
            self.assertIn("- `execution_site`: `CUDA`", spec)
            self.assertIn(
                "- `last_completed_action`: "
                "`revision_6_cuda_capture_session_reserved`",
                spec,
            )
            self.assertIn(
                "- `capture_session_id`: "
                "`runs/cuda-formal-session-r6-001`",
                spec,
            )
            self.assertIn("- `session_status`: `ACTIVE`", spec)

    def test_environment_failure_updates_state_and_allocates_fresh_run(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "migration-spec.md").write_bytes(SPEC.read_bytes())
            self._run_environment_failure_transition(workspace)
            spec = (workspace / "migration-spec.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("- `state_revision`: `50`", spec)
            self.assertIn("- `status`: `ACTIVE`", spec)
            self.assertIn("- `phase`: `CUDA_CAPTURE`", spec)
            self.assertIn("- `execution_site`: `CUDA`", spec)
            self.assertIn(
                "- `last_completed_action`: "
                "`revision_6_cuda_environment_preflight_failed`",
                spec,
            )
            self.assertIn(
                "- `last_run`: "
                "`runs/cuda-environment-preflight-r6-001`",
                spec,
            )
            self.assertIn(
                "ENV_RUN=runs/cuda-environment-preflight-r6-002",
                spec,
            )
            self.assertIn("- `capture_session_id`: `null`", spec)
            self.assertIn("- `session_status`: `NOT_STARTED`", spec)

    def test_reservation_accepts_a_prior_environment_failure_revision(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "migration-spec.md").write_bytes(SPEC.read_bytes())
            self._run_environment_failure_transition(workspace)
            self._run_reservation_transition(workspace)
            spec = (workspace / "migration-spec.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("- `state_revision`: `51`", spec)
            self.assertIn(
                "- `last_completed_action`: "
                "`revision_6_cuda_capture_session_reserved`",
                spec,
            )
            self.assertIn("- `session_status`: `ACTIVE`", spec)

    def test_candidate_spec_transition_executes_after_reservation(
        self,
    ) -> None:
        transition = self._python_block(
            "prepare_revision_6_handoff_spec"
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "migration-spec.md").write_bytes(SPEC.read_bytes())
            self._run_reservation_transition(workspace)
            session = workspace / "runs" / "cuda-formal-session-r6-001"
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
                    "SESSION_RUN": "runs/cuda-formal-session-r6-001",
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
            self.assertIn("- `state_revision`: `51`", candidate)
            self.assertIn("- `status`: `WAITING`", candidate)
            self.assertIn("- `phase`: `HANDOFF`", candidate)
            self.assertIn("- `execution_site`: `CUDA`", candidate)
            self.assertIn("- `bundle_status`: `VALID`", candidate)
            for item in self.scan["gap_queue"]:
                self.assertNotIn(
                    f"| `{item['operator_id']}` | "
                    "`CAPTURE_REQUIRED` | `null` |",
                    candidate,
                )

    def test_failure_detects_sample_file_before_state_update(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "migration-spec.md").write_bytes(SPEC.read_bytes())
            self._run_reservation_transition(workspace)
            sample = (
                workspace
                / "runs"
                / "cuda-formal-session-r6-001"
                / "operators"
                / "01"
                / "samples"
                / "orphan.pt"
            )
            sample.parent.mkdir(parents=True)
            sample.write_bytes(b"already-captured")
            launch = (
                workspace
                / "runs"
                / "cuda-formal-session-r6-001"
                / "launch-50"
            )
            launch.mkdir()
            environment = os.environ.copy()
            environment.update(
                {
                    "SESSION_RUN": "runs/cuda-formal-session-r6-001",
                    "LAUNCH_EVIDENCE_DIR": (
                        "runs/cuda-formal-session-r6-001/launch-50"
                    ),
                    "FAILED_COMMAND": "curl /generate",
                    "FAILED_EXIT_CODE": "1",
                    "FAILURE_SUMMARY": "request failed after sample write",
                }
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    self._python_block("record_revision_6_cuda_failure"),
                ],
                cwd=workspace,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            failure = json.loads(
                (
                    workspace
                    / "runs"
                    / "cuda-formal-session-r6-001"
                    / "failure-51.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(failure["sample_files_on_disk"], 1)
            self.assertEqual(failure["saved_shape_count"], 0)
            self.assertTrue(failure["formal_session_consumed"])
            spec = (workspace / "migration-spec.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("- `state_revision`: `51`", spec)
            self.assertIn("- `status`: `BLOCKED`", spec)
            self.assertIn("- `phase`: `CUDA_CAPTURE`", spec)
            self.assertIn("- `session_status`: `FAILED`", spec)
            self.assertIn("- `next_action`: `none`", spec)

    def test_zero_sample_failure_keeps_the_same_reservation_active(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "migration-spec.md").write_bytes(SPEC.read_bytes())
            self._run_reservation_transition(workspace)
            launch = (
                workspace
                / "runs"
                / "cuda-formal-session-r6-001"
                / "launch-50"
            )
            launch.mkdir()
            (launch / "server.log").write_text(
                "first launch failed\n",
                encoding="utf-8",
            )
            state = (
                workspace
                / "runs"
                / "cuda-formal-session-r6-001"
                / "operators"
                / "01"
                / "capture-state.json"
            )
            state.parent.mkdir(parents=True)
            contract = load_contract_data(SPEC)
            checkpoint = checkpoint_metadata(
                contract["checkpoint"]["id"],
                contract["checkpoint"]["config_digest"],
            )
            session_config = (
                workspace
                / "runs"
                / "cuda-formal-session-r6-001"
                / "capture-config.json"
            ).resolve()
            operator_id = self.scan["gap_queue"][0]["operator_id"]
            (state.parent / "capture-config.json").write_text(
                json.dumps(
                    {
                        "schema": "golden-capture-config/v1",
                        "spec_binding": self.scan["spec_binding"],
                        "operator_id": operator_id,
                        "tp_rank": 0,
                        "tensor_parallel_size": 8,
                        "checkpoint": checkpoint,
                        "session_config": str(session_config),
                        "run_dir": str(state.parent.resolve()),
                    }
                ),
                encoding="utf-8",
            )
            state.write_text(
                json.dumps(
                    {
                        "schema": "kernel-call-capture-state/v1",
                        "spec_binding": self.scan["spec_binding"],
                        "operator_id": operator_id,
                        "tp_rank": 0,
                        "tensor_parallel_size": 8,
                        "checkpoint": checkpoint,
                        "loaded_checkpoint": {
                            "model_path": checkpoint["model_path"],
                            "revision": checkpoint["revision"],
                        },
                        "capture_session_config": str(session_config),
                        "status": "ACTIVE",
                        "capture_closed": False,
                        "saved_shape_count": 0,
                        "repeated_call_count": 0,
                        "skipped_call_count": 0,
                        "samples": [],
                        "skipped_signatures": [],
                        "capture_process_id": 1234,
                    }
                ),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "SESSION_RUN": "runs/cuda-formal-session-r6-001",
                    "LAUNCH_EVIDENCE_DIR": (
                        "runs/cuda-formal-session-r6-001/launch-50"
                    ),
                    "FAILED_COMMAND": "python -m sglang.launch_server",
                    "FAILED_EXIT_CODE": "1",
                    "FAILURE_SUMMARY": "checkpoint path was wrong",
                }
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    self._python_block("record_revision_6_cuda_failure"),
                ],
                cwd=workspace,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            failure = json.loads(
                (
                    workspace
                    / "runs"
                    / "cuda-formal-session-r6-001"
                    / "failure-51.json"
                ).read_text(encoding="utf-8")
            )
            self.assertFalse(failure["formal_session_consumed"])
            spec = (workspace / "migration-spec.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("- `state_revision`: `51`", spec)
            self.assertIn("- `status`: `ACTIVE`", spec)
            self.assertIn("- `session_status`: `ACTIVE`", spec)
            self.assertIn("同一个 reservation", spec)
            self.assertNotIn("- `status`: `BLOCKED`", spec)
            self.assertEqual(
                (launch / "server.log").read_text(encoding="utf-8"),
                "first launch failed\n",
            )
            self.assertIn(
                "launch-50",
                failure["launch_evidence_dir"],
            )
            self.assertFalse(state.exists())
            archived_state = (
                launch
                / "zero-sample-capture-states"
                / "01"
                / "capture-state.json"
            )
            self.assertTrue(archived_state.is_file())
            self.assertEqual(
                failure["archived_zero_sample_states"],
                [str(archived_state.relative_to(workspace))],
            )

    def test_incomplete_zero_sample_state_blocks_instead_of_archiving(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "migration-spec.md").write_bytes(SPEC.read_bytes())
            self._run_reservation_transition(workspace)
            launch = (
                workspace
                / "runs"
                / "cuda-formal-session-r6-001"
                / "launch-50"
            )
            launch.mkdir()
            operator = (
                workspace
                / "runs"
                / "cuda-formal-session-r6-001"
                / "operators"
                / "01"
            )
            operator.mkdir(parents=True)
            contract = load_contract_data(SPEC)
            checkpoint = checkpoint_metadata(
                contract["checkpoint"]["id"],
                contract["checkpoint"]["config_digest"],
            )
            session_config = (
                workspace
                / "runs"
                / "cuda-formal-session-r6-001"
                / "capture-config.json"
            ).resolve()
            operator_id = self.scan["gap_queue"][0]["operator_id"]
            (operator / "capture-config.json").write_text(
                json.dumps(
                    {
                        "schema": "golden-capture-config/v1",
                        "spec_binding": self.scan["spec_binding"],
                        "operator_id": operator_id,
                        "tp_rank": 0,
                        "tensor_parallel_size": 8,
                        "checkpoint": checkpoint,
                        "session_config": str(session_config),
                        "run_dir": str(operator.resolve()),
                    }
                ),
                encoding="utf-8",
            )
            state = operator / "capture-state.json"
            state.write_text(
                json.dumps(
                    {
                        "status": "ACTIVE",
                        "capture_closed": False,
                        "saved_shape_count": 0,
                        "samples": [],
                        "capture_process_id": 1234,
                    }
                ),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "SESSION_RUN": "runs/cuda-formal-session-r6-001",
                    "LAUNCH_EVIDENCE_DIR": (
                        "runs/cuda-formal-session-r6-001/launch-50"
                    ),
                    "FAILED_COMMAND": "python -m sglang.launch_server",
                    "FAILED_EXIT_CODE": "1",
                    "FAILURE_SUMMARY": "collector state was incomplete",
                }
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    self._python_block("record_revision_6_cuda_failure"),
                ],
                cwd=workspace,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            failure = json.loads(
                (
                    workspace
                    / "runs"
                    / "cuda-formal-session-r6-001"
                    / "failure-51.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(failure["formal_session_consumed"])
            self.assertTrue(failure["capture_state_errors"])
            self.assertEqual(
                failure["archived_zero_sample_states"],
                [],
            )
            self.assertTrue(state.is_file())
            spec = (workspace / "migration-spec.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("- `status`: `BLOCKED`", spec)
            self.assertIn("- `session_status`: `FAILED`", spec)

    def test_final_summary_uses_the_actual_environment_run(self) -> None:
        final_section = self.runbook[
            self.runbook.index("回传下面五项后停止") :
            self.runbook.index("## 失败边界")
        ]
        self.assertIn("'environment run: %s\\n' \"$ENV_RUN\"", final_section)
        self.assertNotIn(
            "environment run: runs/cuda-environment-preflight-r6-001",
            final_section,
        )

    def test_source_review_keeps_execution_site_at_source(self) -> None:
        spec = SPEC.read_text(encoding="utf-8")
        skill = SKILL.read_text(encoding="utf-8")
        design = DESIGN.read_text(encoding="utf-8")
        map_text = MAP.read_text(encoding="utf-8")

        self.assertIn("- `state_revision`: `49`", spec)
        self.assertIn("- `execution_site`: `SOURCE`", spec)
        self.assertIn(
            "- `last_completed_action`: "
            "`revision_6_formal_cuda_capture_runbook_review_corrected`",
            spec,
        )
        self.assertIn(
            "- `last_run`: `runs/formal-cuda-runbook-r6-002`",
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
        self.assertIn("runbook 已完成 SOURCE 修正复审", design)
        self.assertIn("下一步转到 CUDA 机器", design)
        self.assertIn("runbook 已完成 SOURCE 修正复审", map_text)

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
