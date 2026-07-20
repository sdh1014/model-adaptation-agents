"""Repair replay adapter: route P800 replay through the repaired boundary.

These tests prove the model-adaptation flow can generate and accept a P800
Kernel Call replay that runs through an Agent-selected repaired boundary living
inside the pinned SGLang-Kunlun worktree, without presetting the module or
function name and without letting a flow-code shortcut stand in for the real
repair. They exercise the deterministic tools only; the actual source repair is
claimed as a bounded attempt on hardware.
"""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "model-adaptation" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from model_adaptation_capture.contracts import (
    KUNLUN_SWIGLU_TARGET,
    normalize_repair_entry,
    parse_repair_invocation_target,
    repair_invocation_target,
)
from model_adaptation_capture.kernel_replay import (
    KernelReplayError,
    _resolve_repair_call,
    run_kernel_replay_worker,
)
from test_ticket23_kernel_replay import (
    BINDING,
    OPERATOR_ID,
    PRECISION,
    write_golden_run,
)


REPAIR_ENTRY = "repaired_boundary:apply_swiglu_clamp"
REPAIR_TARGET = f"repair-kernel-call/v1:{REPAIR_ENTRY}"
KUNLUN_REVISION = "546ad8c682392922792bbbfe53a8bf575545f118"


def _clamped(x: torch.Tensor, limit: float) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = torch.nn.functional.silu(x[..., :d].float()).clamp(max=limit)
    linear = x[..., d:].float().clamp(min=-limit, max=limit)
    return (gate * linear).to(x.dtype)


def write_clamp_golden_run(path: Path, *, limit: float = 7.0) -> None:
    """A SEALED, self-replayed Golden Run whose output is the true clamp."""

    from model_adaptation_capture.contracts import shape_id_for

    samples = path / "samples"
    samples.mkdir(parents=True)
    x = torch.tensor([[1.0, 2.0, 9.0, 4.0]], dtype=torch.bfloat16)
    expected = _clamped(x, limit)
    shape_id = shape_id_for(x.shape)
    signature = {
        "operator_id": OPERATOR_ID,
        "model_path": "target",
        "tensor_parallel_size": 8,
        "tp_rank": 0,
        "inputs": {
            "x": {
                "kind": "tensor",
                "shape": list(x.shape),
                "dtype": "bfloat16",
                "layout": "strided",
                "stride": list(x.stride()),
            }
        },
        "parameters": {},
        "non_tensor_args": {"gemm1_limit": float(limit)},
    }
    sample_path = samples / f"{shape_id}.pt"
    torch.save(
        {
            "schema": "kernel-call-sample/v1",
            "spec_binding": BINDING,
            "operator_id": OPERATOR_ID,
            "tp_rank": 0,
            "tensor_parallel_size": 8,
            "signature": signature,
            "inputs": {"x": x},
            "parameters": {},
            "non_tensor_args": {"gemm1_limit": float(limit)},
            "outputs": {"output": expected},
        },
        sample_path,
    )
    (path / "sample-files.json").write_text(
        json.dumps(
            {
                "schema": "golden-sample-files/v1",
                "spec_binding": BINDING,
                "operator_id": OPERATOR_ID,
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
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "capture-state.json").write_text(
        json.dumps(
            {
                "schema": "kernel-call-capture-state/v1",
                "spec_binding": BINDING,
                "operator_id": OPERATOR_ID,
                "tp_rank": 0,
                "tensor_parallel_size": 8,
                "status": "SEALED",
                "capture_closed": True,
                "saved_shape_count": 1,
                "samples": [
                    {
                        "shape_id": shape_id,
                        "file": f"samples/{shape_id}.pt",
                        "signature": signature,
                        "repeat_count": 0,
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def make_worktree(root: Path) -> Path:
    """A stand-in for the fixed SGLang-Kunlun worktree with a repaired entry."""

    worktree = root / "sglang-kunlun"
    package = worktree / "python"
    package.mkdir(parents=True)
    module = package / "repaired_boundary.py"
    module.write_text(
        "import torch\n"
        "\n"
        "\n"
        "def apply_swiglu_clamp(x, gemm1_limit):\n"
        "    d = x.shape[-1] // 2\n"
        "    gate = torch.nn.functional.silu(x[..., :d].float())"
        ".clamp(max=gemm1_limit)\n"
        "    linear = x[..., d:].float().clamp(min=-gemm1_limit, "
        "max=gemm1_limit)\n"
        "    return (gate * linear).to(x.dtype)\n",
        encoding="utf-8",
    )
    return worktree


def repair_config(golden_run: Path, run_dir: Path, worktree: Path) -> dict:
    return {
        "schema": "kernel-call-replay-config/v1",
        "spec_binding": BINDING,
        "operator_id": OPERATOR_ID,
        "execution_site": "p800",
        "invocation_target": REPAIR_TARGET,
        "repair_entry": REPAIR_ENTRY,
        "sglang_kunlun_worktree": str(worktree.resolve()),
        "sglang_kunlun_revision": KUNLUN_REVISION,
        "golden_run": str(golden_run),
        "run_dir": str(run_dir),
        "tensor_parallel_size": 8,
        "tp_rank": 0,
        "precision_gate": PRECISION,
        "sample_files_sha256": hashlib.sha256(
            (golden_run / "sample-files.json").read_bytes()
        ).hexdigest(),
        "allow_active_capture": False,
    }


class ContractHelperTest(unittest.TestCase):
    def test_repair_target_round_trips(self) -> None:
        target = repair_invocation_target(REPAIR_ENTRY)
        self.assertEqual(target, REPAIR_TARGET)
        self.assertEqual(parse_repair_invocation_target(target), REPAIR_ENTRY)

    def test_repair_entry_must_be_module_and_callable(self) -> None:
        for bad in ("nocolon", "a:b:c", "1bad:apply", "mod:1apply", "", ":x"):
            with self.assertRaises(ValueError):
                normalize_repair_entry(bad)

    def test_baseline_target_is_not_a_repair_target(self) -> None:
        with self.assertRaises(ValueError):
            parse_repair_invocation_target(KUNLUN_SWIGLU_TARGET)


class ResolveRepairCallTest(unittest.TestCase):
    def test_resolves_callable_inside_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            worktree = make_worktree(Path(temp_dir))
            sys.path.insert(0, str((worktree / "python").resolve()))
            try:
                call = _resolve_repair_call(
                    {
                        "invocation_target": REPAIR_TARGET,
                        "repair_entry": REPAIR_ENTRY,
                        "sglang_kunlun_worktree": str(worktree.resolve()),
                    }
                )
                x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
                out = call(x, 7.0)
                torch.testing.assert_close(out, _clamped(x, 7.0))
            finally:
                sys.path.remove(str((worktree / "python").resolve()))
                sys.modules.pop("repaired_boundary", None)

    def test_rejects_entry_outside_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            worktree = make_worktree(root)
            outside = root / "outside"
            outside.mkdir()
            (outside / "flow_shortcut.py").write_text(
                "def apply(x, gemm1_limit):\n    return x[..., : x.shape[-1] // 2]\n",
                encoding="utf-8",
            )
            sys.path.insert(0, str(outside.resolve()))
            try:
                with self.assertRaises(KernelReplayError):
                    _resolve_repair_call(
                        {
                            "invocation_target": "repair-kernel-call/v1:flow_shortcut:apply",
                            "repair_entry": "flow_shortcut:apply",
                            "sglang_kunlun_worktree": str(worktree.resolve()),
                        }
                    )
            finally:
                sys.path.remove(str(outside.resolve()))
                sys.modules.pop("flow_shortcut", None)


class RepairWorkerTest(unittest.TestCase):
    def test_repaired_boundary_clears_the_clamp_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            golden_run = workspace / "golden-001"
            write_clamp_golden_run(golden_run)
            worktree = make_worktree(workspace)
            run_dir = workspace / "repair-replay"

            python_path = str((worktree / "python").resolve())
            sys.path.insert(0, python_path)
            try:
                result = run_kernel_replay_worker(
                    repair_config(golden_run, run_dir, worktree),
                    device=torch.device("cpu"),
                )
            finally:
                sys.path.remove(python_path)
                sys.modules.pop("repaired_boundary", None)

            self.assertTrue(result["passed"])
            self.assertEqual(result["invocation_target"], REPAIR_TARGET)
            self.assertFalse(list(workspace.rglob("*actual*.pt")))

    def test_baseline_config_must_not_carry_repair_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            golden_run = workspace / "golden-001"
            write_golden_run(golden_run)
            worktree = make_worktree(workspace)
            run_dir = workspace / "replay"
            config = repair_config(golden_run, run_dir, worktree)
            config["invocation_target"] = KUNLUN_SWIGLU_TARGET
            with self.assertRaises(KernelReplayError):
                run_kernel_replay_worker(
                    config,
                    p800_call=lambda x, limit: _clamped(x, limit),
                    device=torch.device("cpu"),
                )

    def test_repair_config_requires_pinned_revision_and_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            golden_run = workspace / "golden-001"
            write_golden_run(golden_run)
            worktree = make_worktree(workspace)
            for field in ("sglang_kunlun_revision", "sglang_kunlun_worktree"):
                config = repair_config(golden_run, workspace / "r", worktree)
                del config[field]
                with self.assertRaises(KernelReplayError):
                    run_kernel_replay_worker(
                        config,
                        p800_call=lambda x, limit: _clamped(x, limit),
                        device=torch.device("cpu"),
                    )


if __name__ == "__main__":
    unittest.main()
