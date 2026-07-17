import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "model-adaptation" / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from model_adaptation_capture.replay import LoadedModelReplay, ModelReplayError
import replay_compare


REPLAY_COMPARE = SCRIPTS_ROOT / "replay_compare.py"
OPERATOR_ID = "Step3p5MLP.forward"
MODEL_INSTANCE_PATH = "model.layers.43.share_expert"


def contract_data() -> dict:
    return {
        "schema": "migration-spec/v0",
        "spec_id": "step3p7-flash-p800-demo",
        "contract_revision": 4,
        "model": "Step-3.7-Flash",
        "source": {
            "sglang_revision": "49e384ce9d304648e9959666ecb8ce8cd98d0deb",
            "sglang_kunlun_revision": "546ad8c682392922792bbbfe53a8bf575545f118",
        },
        "checkpoint": {
            "id": "stepfun-ai/Step-3.7-Flash@fixture",
            "config_digest": "config-digest",
        },
        "model_path": {
            "target_entry": "Step3p7ForConditionalGeneration.forward",
            "draft_entry": None,
        },
        "operator_boundary": {
            "id": OPERATOR_ID,
            "activation_guard": "self.limit is not None",
            "tp_rank": 0,
        },
        "runtime": {
            "tensor_parallel_size": 8,
            "dtype": "bfloat16",
            "quantization": None,
            "speculative_algorithm": None,
            "cuda_graph_backend_decode": "disabled",
            "cuda_graph_backend_prefill": "disabled",
            "attention_backend": None,
        },
        "demo_input_mode": "text-only",
        "limits": {
            "max_shapes_per_operator": 3,
            "max_repair_attempts": 5,
        },
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


def binding_for(contract: dict) -> dict:
    canonical = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "spec_id": contract["spec_id"],
        "contract_revision": contract["contract_revision"],
        "contract_data_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def write_spec(path: Path, contract: dict) -> None:
    path.write_text(
        "\n".join(
            (
                "# Migration Spec",
                "<!-- CONTRACT-DATA: BEGIN -->",
                json.dumps(contract, ensure_ascii=False, indent=2),
                "<!-- CONTRACT-DATA: END -->",
                "",
            )
        ),
        encoding="utf-8",
    )


def signature(rows: int) -> dict:
    return {
        "operator_id": OPERATOR_ID,
        "model_path": "target",
        "model_instance_path": MODEL_INSTANCE_PATH,
        "execution_phase": "decode",
        "tensor_parallel_size": 8,
        "tp_rank": 0,
        "inputs": {
            "x": {
                "kind": "tensor",
                "shape": [rows, 8],
                "dtype": "bfloat16",
                "layout": "strided",
                "stride": [8, 1],
            }
        },
        "parameters": {"limit": 16.0},
    }


def shape_id(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            {"x": value["inputs"]["x"]["shape"]},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def write_golden_run(run_dir: Path, binding: dict) -> None:
    state_samples = []
    for rows in (1, 2, 4):
        shape_signature = signature(rows)
        current_shape_id = shape_id(shape_signature)
        sample_path = run_dir / "samples" / f"{current_shape_id}.pt"
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        x = torch.arange(rows * 8, dtype=torch.bfloat16).reshape(rows, 8)
        torch.save(
            {
                "schema": "capture-candidate/v1",
                "spec_binding": binding,
                "operator_id": OPERATOR_ID,
                "model_instance_path": MODEL_INSTANCE_PATH,
                "tp_rank": 0,
                "tensor_parallel_size": 8,
                "signature": shape_signature,
                "inputs": {"x": x},
                "parameters": {"limit": 16.0},
                "outputs": {"output": x * 2},
            },
            sample_path,
        )
        state_samples.append(
            {
                "shape_id": current_shape_id,
                "file": str(sample_path.relative_to(run_dir)),
                "signature": shape_signature,
                "repeat_count": 0,
            }
        )

    (run_dir / "capture-state.json").write_text(
        json.dumps(
            {
                "schema": "golden-capture-state/v2",
                "spec_binding": binding,
                "operator_id": OPERATOR_ID,
                "tp_rank": 0,
                "tensor_parallel_size": 8,
                "checkpoint": {
                    "id": "stepfun-ai/Step-3.7-Flash@fixture",
                    "model_path": "stepfun-ai/Step-3.7-Flash",
                    "revision": "fixture",
                    "config_digest": "config-digest",
                },
                "loaded_checkpoint": {
                    "model_path": "stepfun-ai/Step-3.7-Flash",
                    "revision": "fixture",
                },
                "status": "ACTIVE",
                "capture_closed": False,
                "saved_shape_count": 3,
                "repeated_call_count": 0,
                "skipped_call_count": 0,
                "samples": state_samples,
                "skipped_signatures": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class Ticket20LoadedModelReplayTest(unittest.TestCase):
    def test_replays_rank_zero_shapes_inside_loaded_tp8_mlp_instance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            contract = contract_data()
            binding = binding_for(contract)
            spec = workspace / "migration-spec.md"
            golden_run = workspace / "golden-001"
            replay_run = workspace / "cuda-self-replay-001"
            write_spec(spec, contract)
            write_golden_run(golden_run, binding)

            prepared = subprocess.run(
                [
                    sys.executable,
                    str(REPLAY_COMPARE),
                    "--spec",
                    str(spec),
                    "--run-dir",
                    str(replay_run),
                    "--mode",
                    "prepare-model-replay",
                    "--golden-run",
                    str(golden_run),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            config_path = replay_run / "replay-config.json"
            capture_state = json.loads(
                (golden_run / "capture-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(capture_state["status"], "ACTIVE")
            self.assertTrue(capture_state["capture_closed"])
            prepared_config = json.loads(
                config_path.read_text(encoding="utf-8")
            )
            loaded_checkpoint = {
                "model_path": "stepfun-ai/Step-3.7-Flash",
                "revision": "fixture",
            }

            with self.assertRaisesRegex(
                ModelReplayError,
                "loaded checkpoint",
            ):
                LoadedModelReplay.from_config_path(
                    config_path,
                    tp_rank=0,
                    tp_size=8,
                    loaded_checkpoint={
                        "model_path": "stepfun-ai/Another-Model",
                        "revision": "fixture",
                    },
                )

            relaxed_config = copy.deepcopy(prepared_config)
            relaxed_config["precision_gate"]["atol"] = 1.0
            config_path.write_text(
                json.dumps(relaxed_config) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ModelReplayError, "precision gate"):
                LoadedModelReplay.from_config_path(
                    config_path,
                    tp_rank=0,
                    tp_size=8,
                    loaded_checkpoint=loaded_checkpoint,
                )

            incomplete_config = copy.deepcopy(prepared_config)
            incomplete_config["shape_groups"].pop()
            config_path.write_text(
                json.dumps(incomplete_config) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ModelReplayError, "shape groups"):
                LoadedModelReplay.from_config_path(
                    config_path,
                    tp_rank=0,
                    tp_size=8,
                    loaded_checkpoint=loaded_checkpoint,
                )
            config_path.write_text(
                json.dumps(prepared_config) + "\n",
                encoding="utf-8",
            )

            module = types.SimpleNamespace(
                limit=16.0,
                gate_up_proj=types.SimpleNamespace(
                    prefix=f"{MODEL_INSTANCE_PATH}.gate_up_proj"
                ),
            )

            def loaded_mlp_forward(_module, x):
                return x * 2

            non_replay_rank_calls = 0

            def must_not_replay_on_rank_one(_module, x):
                nonlocal non_replay_rank_calls
                non_replay_rank_calls += 1
                return x * 2

            non_replay_rank = LoadedModelReplay.from_config_path(
                config_path,
                tp_rank=1,
                tp_size=8,
                loaded_checkpoint=loaded_checkpoint,
            )
            non_replay_rank.run_for_instance(
                must_not_replay_on_rank_one,
                module,
                model_instance_path=MODEL_INSTANCE_PATH,
                device=torch.device("cpu"),
            )
            self.assertEqual(non_replay_rank_calls, 0)

            replay = LoadedModelReplay.from_config_path(
                config_path,
                tp_rank=0,
                tp_size=8,
                loaded_checkpoint=loaded_checkpoint,
            )
            replay.run_for_instance(
                loaded_mlp_forward,
                module,
                model_instance_path=MODEL_INSTANCE_PATH,
                device=torch.device("cpu"),
            )
            replay_result_path = replay_run / "replay-result.json"
            valid_replay_result = json.loads(
                replay_result_path.read_text(encoding="utf-8")
            )

            contradictory_result = copy.deepcopy(valid_replay_result)
            contradictory_result["checked_shapes"][0]["passed"] = False
            replay_result_path.write_text(
                json.dumps(contradictory_result) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                replay_compare.ToolError,
                "per-shape pass results",
            ):
                replay_compare.run_finalize_model_replay(spec, replay_run)

            wrong_instance_result = copy.deepcopy(valid_replay_result)
            wrong_instance_result["checked_shapes"][0][
                "model_instance_path"
            ] = "model.layers.44.share_expert"
            replay_result_path.write_text(
                json.dumps(wrong_instance_result) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                replay_compare.ToolError,
                "checked shape groups",
            ):
                replay_compare.run_finalize_model_replay(spec, replay_run)

            extra_field_result = copy.deepcopy(valid_replay_result)
            extra_field_result["checked_shapes"][0]["unexpected"] = True
            replay_result_path.write_text(
                json.dumps(extra_field_result) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                replay_compare.ToolError,
                "checked_shapes entry",
            ):
                replay_compare.run_finalize_model_replay(spec, replay_run)
            replay_result_path.write_text(
                json.dumps(valid_replay_result) + "\n",
                encoding="utf-8",
            )

            config_path.write_text(
                json.dumps(relaxed_config) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                replay_compare.ToolError,
                "precision gate",
            ):
                replay_compare.run_finalize_model_replay(spec, replay_run)

            config_path.write_text(
                json.dumps(incomplete_config) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                replay_compare.ToolError,
                "shape groups",
            ):
                replay_compare.run_finalize_model_replay(spec, replay_run)

            config_path.write_text(
                json.dumps(prepared_config) + "\n",
                encoding="utf-8",
            )
            real_atomic_write = replay_compare.atomic_write_json

            def fail_capture_state(path, value):
                if Path(path).name == "capture-state.json":
                    raise OSError("simulated capture-state write failure")
                return real_atomic_write(path, value)

            with patch.object(
                replay_compare,
                "atomic_write_json",
                side_effect=fail_capture_state,
            ):
                with self.assertRaisesRegex(OSError, "capture-state"):
                    replay_compare.run_finalize_model_replay(spec, replay_run)
            self.assertFalse((replay_run / "result.json").exists())
            capture_state = json.loads(
                (golden_run / "capture-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(capture_state["status"], "ACTIVE")

            def fail_result(path, value):
                if Path(path).name == "result.json":
                    raise OSError("simulated result write failure")
                return real_atomic_write(path, value)

            with patch.object(
                replay_compare,
                "atomic_write_json",
                side_effect=fail_result,
            ):
                with self.assertRaisesRegex(OSError, "result"):
                    replay_compare.run_finalize_model_replay(spec, replay_run)
            self.assertFalse((replay_run / "result.json").exists())
            capture_state = json.loads(
                (golden_run / "capture-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(capture_state["status"], "SEALED")

            replay_compare.run_finalize_model_replay(spec, replay_run)
            result = json.loads(
                (replay_run / "result.json").read_text(encoding="utf-8")
            )
            self.assertTrue(result["passed"])
            self.assertEqual(result["checked_shape_count"], 3)
            self.assertEqual(result["tp_rank"], 0)
            self.assertFalse(list(replay_run.rglob("*actual*.pt")))
            capture_state = json.loads(
                (golden_run / "capture-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(capture_state["status"], "SEALED")
            self.assertTrue(capture_state["capture_closed"])

            p800_run = workspace / "p800-baseline-001"
            replay_compare.run_prepare_model_replay(
                spec,
                p800_run,
                golden_run,
            )
            p800_config_path = p800_run / "replay-config.json"
            p800_config = json.loads(
                p800_config_path.read_text(encoding="utf-8")
            )
            self.assertFalse(p800_config["seal_golden_on_success"])
            p800_replay = LoadedModelReplay.from_config_path(
                p800_config_path,
                tp_rank=0,
                tp_size=8,
                loaded_checkpoint=loaded_checkpoint,
            )
            p800_replay.run_for_instance(
                loaded_mlp_forward,
                module,
                model_instance_path=MODEL_INSTANCE_PATH,
                device=torch.device("cpu"),
            )
            replay_compare.run_finalize_model_replay(spec, p800_run)
            p800_result = json.loads(
                (p800_run / "result.json").read_text(encoding="utf-8")
            )
            self.assertTrue(p800_result["passed"])
            self.assertEqual(
                json.loads(
                    (golden_run / "capture-state.json").read_text(
                        encoding="utf-8"
                    )
                ),
                capture_state,
            )


if __name__ == "__main__":
    unittest.main()
