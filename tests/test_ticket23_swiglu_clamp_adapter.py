import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "model-adaptation" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import capture_golden
from model_adaptation_capture import plugin
from model_adaptation_capture import preflight


SPEC = ROOT / "migration-spec.md"
SCAN = ROOT / "runs" / "scan-006" / "result.json"
OPERATOR_ID = (
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe."
    "_swiglu_silu_clamp_mul"
)
CUDA_SOURCE = (
    "python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py"
)
ADAPTER_RUN = ROOT / "runs" / "adapter-002"


class Ticket23SwiGLUClampAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        plugin._reset_for_tests()

    def test_revision_five_prepare_uses_selected_kernel_call_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree = workspace / "sglang"
            worktree.mkdir()
            source_path = worktree / CUDA_SOURCE
            source_path.parent.mkdir(parents=True)
            source_path.write_text("# fixed source fixture\n", encoding="utf-8")
            run_dir = workspace / "golden-001"

            with patch.object(
                capture_golden,
                "resolve_git_revision",
                return_value="49e384ce9d304648e9959666ecb8ce8cd98d0deb",
            ):
                _, config = capture_golden.prepare_capture_config(
                    SPEC,
                    run_dir,
                    SCAN,
                    OPERATOR_ID,
                    worktree,
                    preflight=True,
                )

            self.assertEqual(config["operator_id"], OPERATOR_ID)
            self.assertEqual(config["hook_target"], OPERATOR_ID)
            self.assertEqual(
                config["boundary"],
                {
                    "inputs": ["x"],
                    "parameters": [],
                    "non_tensor_args": ["gemm1_limit"],
                    "outputs": ["output"],
                },
            )
            self.assertEqual(
                config["sample_fields"],
                {
                    "inputs": ["x"],
                    "parameters": [],
                    "non_tensor_args": ["gemm1_limit"],
                    "outputs": ["output"],
                },
            )
            self.assertEqual(config["replay"]["mode"], "standalone-kernel-call")
            self.assertEqual(config["replay"]["cuda_target"], OPERATOR_ID)
            self.assertEqual(
                config["replay"]["p800_target"],
                "kunlun_ops.swiglu",
            )
            self.assertFalse(config["replay"]["weights_in_golden_sample"])
            self.assertEqual(
                config["preflight_tp_context"],
                {"rank": 0, "size": 8},
            )
            self.assertEqual(
                config["source"]["capture_module_path"],
                str((worktree / CUDA_SOURCE).resolve()),
            )
            self.assertNotIn("state_dependency", config)
            self.assertNotIn("operator_boundary", json.dumps(config))

    def test_prepare_rejects_a_worktree_without_the_selected_source_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree = workspace / "sglang"
            worktree.mkdir()
            run_dir = workspace / "golden-001"

            with patch.object(
                capture_golden,
                "resolve_git_revision",
                return_value="49e384ce9d304648e9959666ecb8ce8cd98d0deb",
            ), self.assertRaisesRegex(capture_golden.ToolError, "source file"):
                capture_golden.prepare_capture_config(
                    SPEC,
                    run_dir,
                    SCAN,
                    OPERATOR_ID,
                    worktree,
                    preflight=True,
                )

            self.assertFalse(run_dir.exists())

    def test_latest_adapter_run_records_source_only_hardening(
        self,
    ) -> None:
        result = json.loads(
            (ADAPTER_RUN / "result.json").read_text(encoding="utf-8")
        )

        self.assertTrue(result["passed"])
        self.assertEqual(
            result["action"],
            "harden_kernel_replay_evidence_binding",
        )
        self.assertEqual(result["adapter_status"], "IMPLEMENTED")
        self.assertEqual(result["active_operator"], OPERATOR_ID)
        self.assertEqual(result["supersedes"], "runs/adapter-001")
        self.assertEqual(
            result["runtime_validation"],
            {
                "consumes_capture_session": False,
                "cuda_available": False,
                "cuda_preflight": "NOT_RERUN_FOR_THIS_SOURCE_REVISION",
                "local_test_device": "none",
                "p800_baseline": "NOT_RERUN_FOR_THIS_SOURCE_REVISION",
            },
        )
        self.assertEqual(
            result["sample_binding"],
            {
                "recorded_before_self_replay": True,
                "replay_config_bound": True,
                "worker_result_bound": True,
                "golden_state_bound": True,
                "handoff_manifest_bound": True,
            },
        )
        self.assertEqual(result["capture_boundary"]["parameters"], [])
        self.assertEqual(result["replay_targets"]["cuda"], OPERATOR_ID)
        self.assertEqual(
            result["replay_targets"]["p800"],
            "kunlun_ops.swiglu",
        )
        for source in result["source_files"]:
            self.assertEqual(
                hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest(),
                source["sha256"],
                source["path"],
            )

    def test_plugin_registers_only_the_existing_swiglu_kernel_call(self) -> None:
        calls = []

        class FakeHookType:
            AROUND = "around"

        class FakeHookRegistry:
            @classmethod
            def register(cls, target, hook, hook_type):
                calls.append((target, hook, hook_type))

        sglang = types.ModuleType("sglang")
        sglang.__path__ = []
        srt = types.ModuleType("sglang.srt")
        srt.__path__ = []
        plugins = types.ModuleType("sglang.srt.plugins")
        plugins.__path__ = []
        hook_registry = types.ModuleType("sglang.srt.plugins.hook_registry")
        hook_registry.HookRegistry = FakeHookRegistry
        hook_registry.HookType = FakeHookType
        modules = {
            "sglang": sglang,
            "sglang.srt": srt,
            "sglang.srt.plugins": plugins,
            "sglang.srt.plugins.hook_registry": hook_registry,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "capture-config.json"
            config_path.write_text(
                json.dumps({"operator_id": OPERATOR_ID}) + "\n",
                encoding="utf-8",
            )
            with patch.dict(sys.modules, modules), patch.dict(
                os.environ,
                {"MODEL_ADAPTATION_CAPTURE_CONFIG": str(config_path)},
                clear=True,
            ):
                plugin.register()

        self.assertEqual(
            [(target, hook_type) for target, _, hook_type in calls],
            [(OPERATOR_ID, FakeHookType.AROUND)],
        )
        self.assertEqual(calls[0][1].__name__, "around_swiglu_clamp")

    def test_preflight_requires_capture_and_new_process_cuda_self_replay(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree = workspace / "sglang"
            worktree.mkdir()
            run_dir = workspace / "cuda-preflight-r5-001"
            run_dir.mkdir()
            config_path = run_dir / "capture-config.json"
            config = {
                "schema": "golden-capture-config/v1",
                "spec_binding": {
                    "spec_id": "step3p7-flash-p800-demo",
                    "contract_revision": 5,
                    "contract_data_sha256": "fixture-sha256",
                },
                "operator_id": OPERATOR_ID,
                "hook_target": OPERATOR_ID,
                "run_dir": str(run_dir),
                "tensor_parallel_size": 8,
                "tp_rank": 0,
                "precision_gate": {
                    "comparator": "torch.testing.assert_close",
                    "atol": 0.01,
                    "rtol": 0.02,
                    "require_exact_structure": True,
                    "require_same_dtype": True,
                    "require_finite": True,
                    "check_stride": False,
                },
            }
            config_path.write_text(
                json.dumps(config) + "\n",
                encoding="utf-8",
            )
            calls = []
            environments = []

            def worker(command, **kwargs):
                calls.append(command)
                environments.append(kwargs["env"])
                config_path = Path(command[-1])
                if config_path.name == "capture-config.json":
                    (run_dir / "capture-state.json").write_text(
                        '{"capture": "rank-zero"}\n',
                        encoding="utf-8",
                    )
                    (run_dir / "samples").mkdir()
                    for index in range(3):
                        (run_dir / "samples" / f"shape-{index}.pt").write_bytes(
                            f"sample-{index}".encode()
                        )
                    (run_dir / "preflight-capture-summary.json").write_text(
                        '{"tp_rank": 0}\n',
                        encoding="utf-8",
                    )
                elif config_path.name == "preflight-rank-filter-config.json":
                    (run_dir / "preflight-rank-filter-summary.json").write_text(
                        json.dumps(
                            {
                                "applied_hooks": [OPERATOR_ID],
                                "runtime_tp_rank": 1,
                                "capture_tp_rank": 0,
                                "tensor_parallel_size": 8,
                                "hook_call_count": 5,
                                "checkpoint_loaded": False,
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                elif "model_adaptation_capture.kernel_replay" in command:
                    replay_config = json.loads(
                        config_path.read_text(encoding="utf-8")
                    )
                    (run_dir / "worker-result.json").write_text(
                        json.dumps(
                            {
                                "schema": "kernel-call-replay-result/v1",
                                "spec_binding": config["spec_binding"],
                                "operator_id": OPERATOR_ID,
                                "execution_site": "cuda",
                                "invocation_target": OPERATOR_ID,
                                "tp_rank": 0,
                                "tensor_parallel_size": 8,
                                "precision_gate": config["precision_gate"],
                                "sample_files_sha256": replay_config[
                                    "sample_files_sha256"
                                ],
                                "passed": True,
                                "checked_shape_count": 3,
                                "failed_shape_count": 0,
                                "checked_shapes": [
                                    {"shape_id": "shape-1", "passed": True},
                                    {"shape_id": "shape-2", "passed": True},
                                    {"shape_id": "shape-3", "passed": True},
                                ],
                                "errors": [],
                                "actual_tensors_saved": False,
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="",
                    stderr="",
                )

            with patch.object(preflight.subprocess, "run", side_effect=worker):
                passed, evidence = preflight.run_capture_preflight(
                    config_path,
                    worktree,
                )

            self.assertTrue(passed)
            self.assertEqual(len(calls), 3)
            rank_filter_config = json.loads(
                (run_dir / "preflight-rank-filter-config.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                rank_filter_config["preflight_tp_context"],
                {"rank": 1, "size": 8},
            )
            self.assertIn(
                "model_adaptation_capture.kernel_replay",
                calls[2],
            )
            self.assertIn("preflight-rank-filter-config.json", evidence)
            self.assertIn("preflight-rank-filter.log", evidence)
            self.assertIn("preflight-rank-filter-summary.json", evidence)
            self.assertIn("preflight-replay-config.json", evidence)
            self.assertIn("preflight-replay.log", evidence)
            self.assertIn("worker-result.json", evidence)
            self.assertIn(
                "MODEL_ADAPTATION_CAPTURE_CONFIG",
                environments[0],
            )
            self.assertIn(
                "MODEL_ADAPTATION_CAPTURE_CONFIG",
                environments[1],
            )
            self.assertNotIn(
                "MODEL_ADAPTATION_CAPTURE_CONFIG",
                environments[2],
            )
            self.assertNotIn("SGLANG_PLUGINS", environments[2])

    def test_preflight_fails_if_rank_filter_worker_changes_rank_zero_capture(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree = workspace / "sglang"
            worktree.mkdir()
            run_dir = workspace / "cuda-preflight-r5-001"
            run_dir.mkdir()
            config_path = run_dir / "capture-config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "operator_id": OPERATOR_ID,
                        "tensor_parallel_size": 8,
                        "tp_rank": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            calls = []

            def worker(command, **kwargs):
                del kwargs
                calls.append(command)
                worker_config = Path(command[-1])
                if worker_config.name == "capture-config.json":
                    (run_dir / "capture-state.json").write_text(
                        '{"capture": "rank-zero"}\n',
                        encoding="utf-8",
                    )
                    samples = run_dir / "samples"
                    samples.mkdir()
                    for index in range(3):
                        (samples / f"shape-{index}.pt").write_bytes(
                            f"sample-{index}".encode()
                        )
                else:
                    (run_dir / "capture-state.json").write_text(
                        '{"capture": "rank-one-regression"}\n',
                        encoding="utf-8",
                    )
                    (run_dir / "preflight-rank-filter-summary.json").write_text(
                        json.dumps(
                            {
                                "applied_hooks": [OPERATOR_ID],
                                "runtime_tp_rank": 1,
                                "capture_tp_rank": 0,
                                "tensor_parallel_size": 8,
                                "hook_call_count": 5,
                                "checkpoint_loaded": False,
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout="",
                    stderr="",
                )

            with patch.object(preflight.subprocess, "run", side_effect=worker):
                passed, evidence = preflight.run_capture_preflight(
                    config_path,
                    worktree,
                )

            self.assertFalse(passed)
            self.assertEqual(len(calls), 2)
            self.assertIn("preflight-rank-filter.log", evidence)
            self.assertNotIn("preflight-replay-config.json", evidence)


if __name__ == "__main__":
    unittest.main()
