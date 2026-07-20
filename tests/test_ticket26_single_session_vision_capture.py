import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "model-adaptation" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import capture_golden
from _lib.spec_contract import load_contract_data, load_spec_binding
from model_adaptation_capture import capture_adapter_validation
from model_adaptation_capture import plugin
from model_adaptation_capture import preflight
from model_adaptation_capture.contracts import (
    ATTENTION_CAPTURE_SEAM,
    ATTENTION_OPERATOR_ID,
    GEMMA_FUSED_ADD_RMSNORM_OPERATOR_ID,
    GEMMA_RMSNORM_OPERATOR_ID,
    SESSION_CONFIG_SCHEMA,
    SWIGLU_CLAMP_OPERATOR_ID,
    TOPK_SIGMOID_OPERATOR_ID,
)


SPEC = ROOT / "migration-spec.md"
SKILL = ROOT / "model-adaptation" / "SKILL.md"
DESIGN = ROOT / "outputs" / "step3p7-p800-migration-tool-design.md"
OPERATOR_DEFINITIONS = (
    (
        SWIGLU_CLAMP_OPERATOR_ID,
        SWIGLU_CLAMP_OPERATOR_ID,
        "text-only",
        {
            "inputs": ["x"],
            "parameters": [],
            "non_tensor_args": ["gemm1_limit"],
            "outputs": ["output"],
        },
    ),
    (
        GEMMA_RMSNORM_OPERATOR_ID,
        "sglang.srt.layers.layernorm.gemma_rmsnorm",
        "text-only",
        {
            "inputs": ["x"],
            "parameters": ["weight"],
            "non_tensor_args": ["eps"],
            "outputs": ["output"],
        },
    ),
    (
        GEMMA_FUSED_ADD_RMSNORM_OPERATOR_ID,
        "sglang.srt.layers.layernorm.gemma_fused_add_rmsnorm",
        "text-only",
        {
            "inputs": ["x", "residual"],
            "parameters": ["weight"],
            "non_tensor_args": ["eps"],
            "outputs": ["mutated_x", "mutated_residual"],
        },
    ),
    (
        TOPK_SIGMOID_OPERATOR_ID,
        "sglang.srt.layers.moe.topk.topk_sigmoid",
        "text-only",
        {
            "inputs": ["gating_output"],
            "parameters": ["correction_bias"],
            "non_tensor_args": ["renormalize"],
            "outputs": ["topk_weights", "topk_ids"],
        },
    ),
    (
        ATTENTION_OPERATOR_ID,
        ATTENTION_CAPTURE_SEAM,
        "single-image",
        {
            "inputs": ["q", "k", "v", "b_start_loc", "b_seq_len"],
            "parameters": [],
            "non_tensor_args": ["max_input_len", "is_causal", "sm_scale"],
            "outputs": ["output"],
        },
    ),
)


def make_scan(image_sha256: str) -> dict:
    contract = load_contract_data(SPEC)
    binding = load_spec_binding(SPEC).as_result_dict()
    operators = []
    capture_plan = []
    gap_queue = []
    for operator_id, hook_target, input_mode, boundary in OPERATOR_DEFINITIONS:
        operators.append(
            {
                "operator_id": operator_id,
                "verdict": "CAPTURE_REQUIRED",
                "model_path": ["target"],
                "input_mode": input_mode,
                "activation_guard": "fixed fixture guard",
                "boundary": boundary,
                "cuda_impl": ["fixed CUDA source"],
                "kunlun_impl": ["fixed Kunlun source"],
                "kernel_call": {
                    "kind": "existing-call",
                    "capture_seam": hook_target,
                },
            }
        )
        capture_plan.append(
            {
                "operator_id": operator_id,
                "hook_target": hook_target,
                "max_distinct_shapes": 3,
                "tp_rank": 0,
                "replay_mode": "standalone-kernel-call",
                "saved_inputs": boundary["inputs"],
                "saved_parameters": boundary["parameters"],
                "saved_non_tensor_args": boundary["non_tensor_args"],
                "saved_outputs": boundary["outputs"],
                "save_direct_parameter_tensors": True,
                "save_full_checkpoint": False,
                "save_module_state_dict": False,
            }
        )
        gap_queue.append(
            {
                "operator_id": operator_id,
                "verdict": "CAPTURE_REQUIRED",
                "runtime_status": "STATIC_GAP_CANDIDATE",
            }
        )
    return {
        "scan_complete": True,
        "spec_binding": binding,
        "source": {
            "sglang_revision": contract["source"]["sglang_revision"],
            "sglang_kunlun_revision": contract["source"][
                "sglang_kunlun_revision"
            ],
            "checkpoint_id": contract["checkpoint"]["id"],
            "checkpoint_config_sha256": contract["checkpoint"]["config_digest"],
        },
        "scan_scope": contract["scan_scope"],
        "operators": operators,
        "gap_queue": gap_queue,
        "capture_plan": capture_plan,
        "selection": {
            "evaluated_after_scan": True,
            "active_operator": SWIGLU_CLAMP_OPERATOR_ID,
        },
        "request_set": [
            {
                "input_mode": "text-only",
                "request": {
                    "text": "Write one word.",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 1,
                    },
                },
            },
            {
                "input_mode": "single-image",
                "image_sha256": image_sha256,
                "request": {
                    "text": "<im_patch>\nDescribe this image in one short sentence.",
                    "image_data": "examples/assets/example_image.png",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 1,
                    },
                },
            },
        ],
    }


class Ticket26SingleSessionVisionCaptureTest(unittest.TestCase):
    def setUp(self) -> None:
        plugin._reset_for_tests()

    def test_revision_six_preflight_reports_only_environment_readiness(
        self,
    ) -> None:
        environment = {
            "python_version": "3.12.0",
            "torch_version": "2.11.0+cu129",
            "cuda_available": True,
            "cuda_device_count": 8,
            "bfloat16_supported": True,
            "sglang_module_path": "/fixed/sglang/python/sglang/__init__.py",
            "plugin_entry_point": "model_adaptation_capture",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree = workspace / "sglang"
            worktree.mkdir()
            run_dir = workspace / "cuda-environment-preflight"

            with patch.object(
                capture_golden,
                "resolve_git_revision",
                return_value=load_contract_data(SPEC)["source"]["sglang_revision"],
            ), patch.object(
                capture_golden,
                "run_environment_preflight",
                return_value=(
                    True,
                    environment,
                    [
                        "environment-config.json",
                        "environment.log",
                        "environment-result.json",
                    ],
                ),
            ):
                capture_golden.run_cuda_environment_preflight(
                    SPEC,
                    run_dir,
                    worktree,
                )

            result = json.loads(
                (run_dir / "result.json").read_text(encoding="utf-8")
            )
            config = json.loads(
                (run_dir / "environment-config.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(result["action"], "environment-preflight")
            self.assertTrue(result["passed"])
            self.assertEqual(
                result["preflight_status"],
                "PREFLIGHT_PASSED",
            )
            self.assertFalse(result["consumes_capture_session"])
            self.assertEqual(result["environment"], environment)
            self.assertNotIn("capture_status", result)
            self.assertNotIn("operator_ids", result)
            self.assertNotIn("request_modes", result)
            self.assertNotIn("scan_result", config)
            self.assertNotIn("operators", config)
            self.assertNotIn("requests", config)
            self.assertNotIn("preflight_tp_context", config)
            self.assertFalse((run_dir / "operators").exists())
            self.assertFalse((run_dir / "samples").exists())

    def test_cuda_environment_preflight_rejects_p800_only_flags(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree = workspace / "sglang"
            worktree.mkdir()
            config_path = workspace / "environment-config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema": preflight.ENVIRONMENT_CONFIG_SCHEMA,
                        "spec_binding": {
                            "spec_id": "step3p7-flash-p800-demo",
                            "contract_revision": 6,
                            "contract_data_sha256": "a" * 64,
                        },
                        "run_dir": str(workspace),
                        "source": {
                            "sglang_revision": "b" * 40,
                            "sglang_worktree": str(worktree),
                        },
                        "runtime": {
                            "tensor_parallel_size": 8,
                            "dtype": "bfloat16",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"SGLANG_IS_FLASHINFER_AVAILABLE": "False"},
                clear=True,
            ), self.assertRaisesRegex(
                preflight.EnvironmentPreflightError,
                "P800-only",
            ):
                preflight.run_environment_worker(config_path)

    def test_cuda_environment_worker_checks_tp8_bf16_and_fixed_sglang(
        self,
    ) -> None:
        class FakeCuda:
            @staticmethod
            def device_count():
                return 8

            @staticmethod
            def is_bf16_supported():
                return True

            @staticmethod
            def get_device_name(index):
                return f"CUDA-{index}"

        fake_torch = types.SimpleNamespace(
            __version__="2.11.0+cu129",
            cuda=FakeCuda(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree = workspace / "sglang"
            package = worktree / "python" / "sglang"
            package.mkdir(parents=True)
            module_file = package / "__init__.py"
            module_file.write_text("", encoding="utf-8")
            config_path = workspace / "environment-config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema": preflight.ENVIRONMENT_CONFIG_SCHEMA,
                        "spec_binding": {
                            "spec_id": "step3p7-flash-p800-demo",
                            "contract_revision": 6,
                            "contract_data_sha256": "a" * 64,
                        },
                        "run_dir": str(workspace),
                        "source": {
                            "sglang_revision": "b" * 40,
                            "sglang_worktree": str(worktree),
                        },
                        "runtime": {
                            "tensor_parallel_size": 8,
                            "dtype": "bfloat16",
                        },
                    }
                ),
                encoding="utf-8",
            )

            loaded_plugins = []
            modules = {
                "sglang": types.ModuleType("sglang"),
                "sglang.srt": types.ModuleType("sglang.srt"),
                "sglang.srt.plugins": types.ModuleType("sglang.srt.plugins"),
            }
            modules["sglang"].__file__ = str(module_file)
            modules["sglang"].__path__ = []
            modules["sglang.srt"].__path__ = []
            modules["sglang.srt.plugins"].load_plugins = (
                lambda: loaded_plugins.append("loaded")
            )

            with patch.object(
                preflight,
                "_require_cuda_torch",
                return_value=fake_torch,
            ), patch.object(
                preflight,
                "_require_capture_entry_point",
            ), patch.dict(
                sys.modules,
                modules,
            ), patch.dict(
                os.environ,
                {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
                clear=True,
            ):
                preflight.run_environment_worker(config_path)

            result = json.loads(
                (workspace / "environment-result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["tensor_parallel_size"], 8)
            self.assertEqual(result["dtype"], "bfloat16")
            self.assertEqual(result["environment"]["cuda_device_count"], 8)
            self.assertEqual(
                result["environment"]["cuda_devices"],
                [f"CUDA-{index}" for index in range(8)],
            )
            self.assertTrue(result["environment"]["bfloat16_supported"])
            self.assertEqual(
                result["environment"]["sglang_module_path"],
                str(module_file.resolve()),
            )
            self.assertEqual(loaded_plugins, ["loaded"])
            serialized = json.dumps(result)
            self.assertNotIn("operator", serialized)
            self.assertNotIn("shape", serialized)
            self.assertNotIn("sample", serialized)
            self.assertNotIn("replay", serialized)

    def test_environment_preflight_cli_rejects_scan_and_operator_inputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "environment-preflight"
            with patch.object(
                sys,
                "argv",
                [
                    "capture_golden.py",
                    "--spec",
                    str(SPEC),
                    "--run-dir",
                    str(run_dir),
                    "--mode",
                    "preflight",
                    "--scan-result",
                    "runs/scan-007/result.json",
                    "--sglang-worktree",
                    "/fixed/sglang",
                ],
            ):
                self.assertEqual(capture_golden.main(), 2)
            self.assertFalse(run_dir.exists())

    def test_revision_six_rejects_historical_capture_adapter_validation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            run_dir = workspace / "capture-adapter-validation"

            with self.assertRaisesRegex(
                capture_golden.ToolError,
                "Contract revisions 1 through 5",
            ):
                capture_golden.run_capture_adapter_validation(
                    SPEC,
                    run_dir,
                    ROOT / "runs" / "scan-007" / "result.json",
                    SWIGLU_CLAMP_OPERATOR_ID,
                    workspace / "sglang",
                )

            self.assertFalse(run_dir.exists())

    def test_revision_six_requires_every_gap_including_vision_in_capture_plan(
        self,
    ) -> None:
        image_bytes = b"fixed-image-fixture"
        scan = make_scan(hashlib.sha256(image_bytes).hexdigest())
        scan["capture_plan"] = [
            item
            for item in scan["capture_plan"]
            if item["operator_id"] != ATTENTION_OPERATOR_ID
        ]
        contract = load_contract_data(SPEC)
        binding = load_spec_binding(SPEC).as_result_dict()

        with self.assertRaisesRegex(
            capture_golden.ToolError,
            "every CAPTURE_REQUIRED operator",
        ):
            capture_golden.validate_capture_session_plan(
                contract,
                scan,
                binding,
            )

    def test_spec_skill_and_design_keep_vision_in_the_same_session(self) -> None:
        spec = SPEC.read_text(encoding="utf-8")
        skill = SKILL.read_text(encoding="utf-8")
        design = DESIGN.read_text(encoding="utf-8")

        self.assertIn("- `scan_run`: `runs/scan-007`", spec)
        self.assertIn(f"| `{ATTENTION_OPERATOR_ID}` |", spec)
        self.assertIn("同一 capture plan", spec)
        for text in (skill, design):
            self.assertIn("examples/assets/example_image.png", text)
            self.assertIn("context_attention_fwd", text)
            self.assertIn("同一", text)

    def test_prepare_session_uses_one_config_for_text_and_fixed_image(
        self,
    ) -> None:
        image_bytes = b"fixed-image-fixture"
        image_digest = hashlib.sha256(image_bytes).hexdigest()
        scan = make_scan(image_digest)

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree = workspace / "sglang"
            for relative_path in capture_golden.SUPPORTED_SOURCE_PATHS.values():
                source = worktree / relative_path
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("# fixed source fixture\n", encoding="utf-8")
            image_path = worktree / "examples/assets/example_image.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(image_bytes)
            scan_path = workspace / "scan.json"
            scan_path.write_text(
                json.dumps(scan, ensure_ascii=False),
                encoding="utf-8",
            )
            run_dir = workspace / "cuda-session-r6-001"

            with patch.object(
                capture_golden,
                "resolve_git_revision",
                return_value=load_contract_data(SPEC)["source"]["sglang_revision"],
            ):
                _, config = capture_golden.prepare_capture_session_config(
                    SPEC,
                    run_dir,
                    scan_path,
                    worktree,
                )

            self.assertEqual(config["schema"], SESSION_CONFIG_SCHEMA)
            self.assertEqual(
                [item["input_mode"] for item in config["requests"]],
                ["text-only", "single-image"],
            )
            self.assertEqual(
                Path(config["requests"][1]["request"]["image_data"]),
                image_path.resolve(),
            )
            self.assertEqual(config["requests"][1]["image_sha256"], image_digest)
            self.assertEqual(
                [item["operator_id"] for item in config["operators"]],
                [item[0] for item in OPERATOR_DEFINITIONS],
            )
            self.assertEqual(
                len({item["session_config"] for item in config["operators"]}),
                1,
            )
            for item in config["operators"]:
                operator_config = json.loads(
                    (
                        Path(item["run_dir"]) / "capture-config.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(operator_config, item)
            self.assertEqual(
                config["operators"][-1]["hook_target"],
                ATTENTION_CAPTURE_SEAM,
            )
            self.assertEqual(
                config["operators"][-1]["sample_fields"]["outputs"],
                ["output"],
            )
            self.assertEqual(
                config["operators"][0]["replay"]["p800_target"],
                "kunlun_ops.swiglu",
            )
            self.assertTrue(
                all(
                    item["replay"]["p800_target"] is None
                    for item in config["operators"][1:]
                )
            )

    def test_session_plugin_registers_all_existing_call_boundaries(self) -> None:
        calls = []

        class FakeHookType:
            AROUND = "around"

        class FakeHookRegistry:
            @classmethod
            def register(cls, target, hook, hook_type):
                calls.append((target, hook.__name__, hook_type))

        modules = {
            "sglang": types.ModuleType("sglang"),
            "sglang.srt": types.ModuleType("sglang.srt"),
            "sglang.srt.plugins": types.ModuleType("sglang.srt.plugins"),
            "sglang.srt.plugins.hook_registry": types.ModuleType(
                "sglang.srt.plugins.hook_registry"
            ),
        }
        modules["sglang"].__path__ = []
        modules["sglang.srt"].__path__ = []
        modules["sglang.srt.plugins"].__path__ = []
        modules[
            "sglang.srt.plugins.hook_registry"
        ].HookRegistry = FakeHookRegistry
        modules["sglang.srt.plugins.hook_registry"].HookType = FakeHookType

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "capture-config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema": SESSION_CONFIG_SCHEMA,
                        "operators": [
                            {"operator_id": item[0]} for item in OPERATOR_DEFINITIONS
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(sys.modules, modules), patch.dict(
                os.environ,
                {"MODEL_ADAPTATION_CAPTURE_CONFIG": str(config_path)},
                clear=True,
            ):
                plugin.register()

        self.assertEqual(
            [target for target, _, _ in calls],
            [item[1] for item in OPERATOR_DEFINITIONS],
        )
        self.assertEqual(
            calls[-1][1],
            "around_vision_prefill_attention",
        )

    def test_attention_hook_records_mutated_output_buffer(self) -> None:
        records = []

        class FakeCollector:
            def record_call(self, **kwargs):
                records.append(kwargs)

        output = {"value": "before"}

        def original(
            q,
            k,
            v,
            o,
            b_start_loc,
            b_seq_len,
            max_input_len,
            is_causal=True,
            sm_scale=None,
        ):
            del q, k, v, b_start_loc, b_seq_len, max_input_len, is_causal, sm_scale
            o["value"] = "after"
            return None

        with patch.object(plugin, "_get_collector", return_value=FakeCollector()):
            result = plugin.around_vision_prefill_attention(
                original,
                "q",
                "k",
                "v",
                output,
                "start",
                "length",
                169,
                is_causal=False,
                sm_scale=None,
            )

        self.assertIsNone(result)
        self.assertEqual(records[0]["outputs"], {"output": output})
        self.assertEqual(output["value"], "after")
        self.assertEqual(
            records[0]["inputs"],
            {
                "q": "q",
                "k": "k",
                "v": "v",
                "b_start_loc": "start",
                "b_seq_len": "length",
            },
        )
        self.assertEqual(
            records[0]["non_tensor_args"],
            {
                "max_input_len": 169,
                "is_causal": False,
                "sm_scale": None,
            },
        )

    def test_session_capture_adapter_validation_covers_every_operator(
        self,
    ) -> None:
        image_bytes = b"fixed-image-fixture"
        scan = make_scan(hashlib.sha256(image_bytes).hexdigest())

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            worktree = workspace / "sglang"
            for relative_path in capture_golden.SUPPORTED_SOURCE_PATHS.values():
                source = worktree / relative_path
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("# fixed source fixture\n", encoding="utf-8")
            image_path = worktree / "examples/assets/example_image.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(image_bytes)
            scan_path = workspace / "scan.json"
            scan_path.write_text(json.dumps(scan), encoding="utf-8")
            run_dir = workspace / "capture-adapter-validation"
            with patch.object(
                capture_golden,
                "resolve_git_revision",
                return_value=load_contract_data(SPEC)["source"]["sglang_revision"],
            ):
                _, config = capture_golden.prepare_capture_session_config(
                    SPEC,
                    run_dir,
                    scan_path,
                    worktree,
                )
            config_path = run_dir / "capture-config.json"
            commands = []

            def worker(command, **kwargs):
                del kwargs
                commands.append(command)
                worker_config_path = Path(command[-1])
                worker_config = json.loads(
                    worker_config_path.read_text(encoding="utf-8")
                )
                if worker_config_path.name == "capture-config.json":
                    for item in config["operators"]:
                        operator_run = Path(item["run_dir"])
                        (operator_run / "capture-state.json").write_text(
                            json.dumps({"operator_id": item["operator_id"]}),
                            encoding="utf-8",
                        )
                        for index in range(3):
                            (operator_run / "samples" / f"shape-{index}.pt").write_bytes(
                                f"{item['operator_id']}-{index}".encode()
                            )
                    (run_dir / "preflight-capture-summary.json").write_text(
                        json.dumps(
                            {
                                "applied_hooks": [
                                    item["hook_target"]
                                    for item in config["operators"]
                                ],
                                "tp_rank": 0,
                                "tensor_parallel_size": 8,
                                "checkpoint_loaded": False,
                                "saved_shape_counts": {
                                    item["operator_id"]: 3
                                    for item in config["operators"]
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                elif worker_config_path.name == "preflight-rank-filter-config.json":
                    (run_dir / "preflight-rank-filter-summary.json").write_text(
                        json.dumps(
                            {
                                "applied_hooks": [
                                    item["hook_target"]
                                    for item in config["operators"]
                                ],
                                "runtime_tp_rank": 1,
                                "capture_tp_rank": 0,
                                "tensor_parallel_size": 8,
                                "checkpoint_loaded": False,
                            }
                        ),
                        encoding="utf-8",
                    )
                else:
                    operator_run = Path(worker_config["golden_run"])
                    (Path(worker_config["run_dir"]) / "worker-result.json").write_text(
                        json.dumps(
                            {
                                "schema": "kernel-call-replay-result/v1",
                                "spec_binding": config["spec_binding"],
                                "operator_id": worker_config["operator_id"],
                                "execution_site": "cuda",
                                "invocation_target": worker_config[
                                    "invocation_target"
                                ],
                                "tp_rank": 0,
                                "tensor_parallel_size": 8,
                                "precision_gate": worker_config["precision_gate"],
                                "sample_files_sha256": worker_config[
                                    "sample_files_sha256"
                                ],
                                "passed": True,
                                "checked_shape_count": 3,
                                "failed_shape_count": 0,
                                "checked_shapes": [
                                    {
                                        "shape_id": f"shape-{index}",
                                        "passed": True,
                                    }
                                    for index in range(3)
                                ],
                                "errors": [],
                                "actual_tensors_saved": False,
                                "golden_run": str(operator_run),
                            }
                        ),
                        encoding="utf-8",
                    )
                return types.SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr="",
                )

            with patch.object(
                capture_adapter_validation.subprocess,
                "run",
                side_effect=worker,
            ):
                passed, evidence = (
                    capture_adapter_validation.run_capture_adapter_validation(
                        config_path,
                        worktree,
                    )
                )

            self.assertTrue(passed)
            self.assertEqual(len(commands), 7)
            replay_configs = list(
                (run_dir / "operators").glob("*/preflight-replay-config.json")
            )
            self.assertEqual(len(replay_configs), len(OPERATOR_DEFINITIONS))
            self.assertIn("preflight-capture-summary.json", evidence)
            self.assertIn("preflight-rank-filter-summary.json", evidence)
            self.assertEqual(
                sum(item.endswith("/worker-result.json") for item in evidence),
                len(OPERATOR_DEFINITIONS),
            )


if __name__ == "__main__":
    unittest.main()
