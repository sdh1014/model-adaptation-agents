import os
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "model-adaptation" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import replay_compare


SPEC = ROOT / "migration-spec.md"
SKILL = ROOT / "model-adaptation" / "SKILL.md"
DESIGN = ROOT / "outputs" / "step3p7-p800-migration-tool-design.md"
ENVIRONMENT_DOC = ROOT / "docs" / "p800-environment-and-repair.md"
BASELINE_RUNBOOK = (
    ROOT / "model-adaptation" / "references" / "p800-baseline-replay.md"
)
CLAUDE_RUNBOOK = (
    ROOT / "model-adaptation" / "references" / "claude-p800-repair.md"
)
SOURCE_RUN = ROOT / "runs" / "p800-launch-environment-tool-001" / "result.json"

OPTIONAL_ENVIRONMENT_VARIABLES = (
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
    "XSHMEM_SYMMETRIC_SIZE",
    "XSHMEM_QP_NUM_PER_RANK",
    "ENABLE_CONTROL_THINK",
    "DEFAULT_ENABLE_THINKING",
    "XPU_HYBRID_ATTN_USE_GATHER_MULTISTREAM",
    "SGLANG_HEALTH_CHECK_TIMEOUT",
    "MODEL_PATH",
    "SGLANG_HACK_FLASHMLA_BACKEND",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM",
    "SGLANG_OPT_USE_TILELANG_MHC_POST",
    "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP",
    "USE_FAST_ALLOC_EXTEND_KUNLUN",
    "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH",
    "SGLANG_OPT_USE_JIT_NORM",
    "SGLANG_OPT_USE_FUSED_STORE_CACHE",
    "SGLANG_OPT_FP8_WO_A_GEMM",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO",
    "SGLANG_TOPK_TRANSFORM_512_TORCH",
    "SGLANG_FIX_DSV4_BASE_MODEL_LOAD",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE",
    "SGLANG_DSV4_FP4_EXPERTS",
    "SGLANG_PREP_IN_CUDA_GRAPH",
    "SGLANG_OPT_SWIGLU_CLAMP_FUSION",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK",
    "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK",
    "SGLANG_OPT_CP_REARRANGE_TRITON",
    "SGLANG_ENABLE_THINKING",
    "SGLANG_TOOL_STRICT_LEVEL",
    "BKCL_RDMA_VERBS",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT",
)


class Ticket27P800KunlunEnvironmentTest(unittest.TestCase):
    def test_source_run_keeps_its_historical_environment_source_inventory(
        self,
    ) -> None:
        result = json.loads(SOURCE_RUN.read_text(encoding="utf-8"))

        self.assertTrue(result["passed"])
        self.assertEqual(result["execution_site"], "SOURCE")
        self.assertFalse(result["p800_runtime_executed"])
        self.assertEqual(
            result["required_environment"],
            {
                "SGLANG_PLATFORM": "kunlun",
                "SGLANG_IS_FLASHINFER_AVAILABLE": "False",
            },
        )
        for source in result["source_files"]:
            self.assertTrue((ROOT / source["path"]).is_file(), source["path"])
            self.assertEqual(len(source["sha256"]), 64, source["path"])
            self.assertTrue(
                all(character in "0123456789abcdef" for character in source["sha256"]),
                source["path"],
            )

    def test_p800_worker_forces_platform_and_worktree_import_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            worktree = Path(temp_dir) / "sglang"
            existing_pythonpath = os.pathsep.join(
                [
                    str((worktree / "python").resolve()),
                    str((worktree / "sglang-kunlun").resolve()),
                    "/existing/pythonpath",
                ]
            )
            with patch.dict(
                os.environ,
                {
                    "PYTHONPATH": existing_pythonpath,
                    "SGLANG_PLATFORM": "wrong-platform",
                    "SGLANG_IS_FLASHINFER_AVAILABLE": "True",
                },
                clear=True,
            ):
                (
                    environment,
                    launch_environment,
                ) = replay_compare._kernel_worker_environment(
                    worktree,
                    "p800",
                )

        python_paths = environment["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(
            python_paths[:2],
            [
                str((worktree / "python").resolve()),
                str((worktree / "sglang-kunlun").resolve()),
            ],
        )
        self.assertEqual(python_paths[-1], "/existing/pythonpath")
        self.assertEqual(
            python_paths.count(str((worktree / "python").resolve())),
            1,
        )
        self.assertEqual(
            python_paths.count(
                str((worktree / "sglang-kunlun").resolve())
            ),
            1,
        )
        self.assertEqual(environment["SGLANG_PLATFORM"], "kunlun")
        self.assertEqual(
            environment["SGLANG_IS_FLASHINFER_AVAILABLE"],
            "False",
        )
        self.assertEqual(
            launch_environment,
            {
                "SGLANG_PLATFORM": "kunlun",
                "SGLANG_IS_FLASHINFER_AVAILABLE": "False",
                "PYTHONPATH_PREFIX": [
                    str((worktree / "python").resolve()),
                    str((worktree / "sglang-kunlun").resolve()),
                ],
                "selected_optional": {},
            },
        )

    def test_p800_worker_requires_an_explicit_kunlun_worktree(self) -> None:
        with self.assertRaisesRegex(
            replay_compare.ToolError,
            "P800.*worktree",
        ):
            replay_compare._kernel_worker_environment(None, "p800")

    def test_cuda_worker_does_not_inherit_forced_p800_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SGLANG_PLATFORM": "kunlun",
                "SGLANG_IS_FLASHINFER_AVAILABLE": "False",
            },
            clear=True,
        ):
            environment, launch_environment = (
                replay_compare._kernel_worker_environment(None, "cuda")
            )

        self.assertNotIn("SGLANG_PLATFORM", environment)
        self.assertNotIn("SGLANG_IS_FLASHINFER_AVAILABLE", environment)
        self.assertEqual(launch_environment, {})

    def test_selected_optional_environment_requires_and_records_reasons(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            worktree = Path(temp_dir) / "sglang"
            with patch.dict(
                os.environ,
                {"MODEL_PATH": "/models/step-3.7-flash"},
                clear=True,
            ):
                with self.assertRaisesRegex(
                    replay_compare.ToolError,
                    "MODEL_PATH.*reason",
                ):
                    replay_compare._kernel_worker_environment(
                        worktree,
                        "p800",
                    )
                _, launch_environment = (
                    replay_compare._kernel_worker_environment(
                        worktree,
                        "p800",
                        {
                            "MODEL_PATH": (
                                "load the fixed Step-3.7-Flash checkpoint"
                            )
                        },
                    )
                )

        self.assertEqual(
            launch_environment["selected_optional"],
            {
                "MODEL_PATH": {
                    "value": "/models/step-3.7-flash",
                    "reason": "load the fixed Step-3.7-Flash checkpoint",
                }
            },
        )

    def test_skill_and_spec_distinguish_required_and_agent_selected_env(
        self,
    ) -> None:
        for text in (
            SPEC.read_text(encoding="utf-8"),
            SKILL.read_text(encoding="utf-8"),
            DESIGN.read_text(encoding="utf-8"),
        ):
            self.assertIn("SGLANG_PLATFORM=kunlun", text)
            self.assertIn("SGLANG_IS_FLASHINFER_AVAILABLE=False", text)
            self.assertIn("Agent", text)
            self.assertIn("按需", text)

        spec = SPEC.read_text(encoding="utf-8")
        self.assertIn("- `state_revision`: `48`", spec)
        self.assertIn(
            "| `41` | 固定 P800 Kunlun 启动基线",
            spec,
        )

    def test_full_environment_catalog_is_preserved_as_agent_input(self) -> None:
        environment_doc = ENVIRONMENT_DOC.read_text(encoding="utf-8")
        self.assertEqual(
            set(replay_compare.P800_OPTIONAL_ENVIRONMENT),
            set(OPTIONAL_ENVIRONMENT_VARIABLES),
        )
        for variable in OPTIONAL_ENVIRONMENT_VARIABLES:
            self.assertIn(variable, environment_doc)
        self.assertIn("D 节点：`32`", environment_doc)
        self.assertIn("P 节点：`256`", environment_doc)
        self.assertIn("D 节点：`2147483648`", environment_doc)
        self.assertIn("P 节点：`8589934592`", environment_doc)
        self.assertIn("不能整表无条件导出", environment_doc)
        for path in (SKILL, BASELINE_RUNBOOK, CLAUDE_RUNBOOK):
            self.assertIn(
                "--p800-environment-reason",
                path.read_text(encoding="utf-8"),
            )

    def test_p800_runbooks_export_confirmed_baseline(self) -> None:
        exact_pythonpath = (
            'export PYTHONPATH="$SGLANG_KUNLUN_WORKTREE/python:'
            '$SGLANG_KUNLUN_WORKTREE/sglang-kunlun'
            '${PYTHONPATH:+:$PYTHONPATH}"'
        )
        for runbook in (BASELINE_RUNBOOK, CLAUDE_RUNBOOK):
            text = runbook.read_text(encoding="utf-8")
            self.assertIn("export SGLANG_PLATFORM=kunlun", text)
            self.assertIn(
                "export SGLANG_IS_FLASHINFER_AVAILABLE=False",
                text,
            )
            self.assertIn(exact_pythonpath, text)
        self.assertIn(
            '--sglang-worktree "$SGLANG_KUNLUN_WORKTREE"',
            BASELINE_RUNBOOK.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
