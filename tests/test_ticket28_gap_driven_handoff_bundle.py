import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Optional
import unittest


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "model-adaptation" / "scripts" / "handoff_bundle.py"
SPEC = ROOT / "migration-spec.md"
SOURCE_RUN = ROOT / "runs" / "gap-driven-handoff-tool-002" / "result.json"


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def read_contract() -> dict:
    text = SPEC.read_text(encoding="utf-8")
    return json.loads(
        text.split("<!-- CONTRACT-DATA: BEGIN -->", 1)[1].split(
            "<!-- CONTRACT-DATA: END -->",
            1,
        )[0]
    )


def binding_for(contract: dict) -> dict:
    return {
        "spec_id": contract["spec_id"],
        "contract_revision": contract["contract_revision"],
        "contract_data_sha256": hashlib.sha256(
            canonical_json_bytes(contract)
        ).hexdigest(),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_spec(path: Path, contract: dict, state: str) -> None:
    path.write_text(
        "\n".join(
            [
                "# Migration Spec",
                "<!-- HUMAN-OWNED CONTRACT: BEGIN -->",
                "<!-- CONTRACT-DATA: BEGIN -->",
                json.dumps(contract, ensure_ascii=False, indent=2),
                "<!-- CONTRACT-DATA: END -->",
                "<!-- HUMAN-OWNED CONTRACT: END -->",
                "<!-- AGENT-WRITABLE WORKING STATE: BEGIN -->",
                state,
                "<!-- AGENT-WRITABLE WORKING STATE: END -->",
                "",
            ]
        ),
        encoding="utf-8",
    )


def operator_definitions() -> list[dict]:
    return [
        {
            "operator_id": "example.kernel.alpha",
            "hook_target": "example.calls.alpha",
            "inputs": ["x"],
            "parameters": ["weight"],
            "non_tensor_args": ["eps"],
            "outputs": ["output"],
        },
        {
            "operator_id": "example.kernel.beta",
            "hook_target": "example.calls.beta",
            "inputs": ["left", "right"],
            "parameters": [],
            "non_tensor_args": ["causal"],
            "outputs": ["value", "index"],
        },
    ]


def make_scan(contract: dict, binding: dict, definitions: list[dict]) -> dict:
    operators = []
    gap_queue = []
    capture_plan = []
    for index, item in enumerate(definitions, start=1):
        boundary = {
            "inputs": item["inputs"],
            "parameters": item["parameters"],
            "non_tensor_args": item["non_tensor_args"],
            "outputs": item["outputs"],
        }
        operators.append(
            {
                "operator_id": item["operator_id"],
                "model_path": ["target"],
                "boundary": boundary,
                "cuda_impl": ["synthetic CUDA evidence"],
                "kunlun_impl": ["synthetic Kunlun evidence"],
                "verdict": "CAPTURE_REQUIRED",
            }
        )
        gap_queue.append(
            {
                "operator_id": item["operator_id"],
                "verdict": "CAPTURE_REQUIRED",
                "demo_role": f"QUEUE_{index}",
            }
        )
        capture_plan.append(
            {
                "operator_id": item["operator_id"],
                "hook_target": item["hook_target"],
                "max_distinct_shapes": contract["limits"][
                    "max_shapes_per_operator"
                ],
                "tp_rank": contract["sample_policy"]["capture_tp_rank"],
                "replay_mode": "standalone-kernel-call",
                "saved_inputs": item["inputs"],
                "saved_parameters": item["parameters"],
                "saved_non_tensor_args": item["non_tensor_args"],
                "saved_outputs": item["outputs"],
                "save_direct_parameter_tensors": True,
                "save_full_checkpoint": False,
                "save_module_state_dict": False,
            }
        )
    return {
        "tool": "migration-agent",
        "action": "scan",
        "scan_id": "synthetic-gap-scan",
        "spec_binding": binding,
        "scan_complete": True,
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
        "operators": operators,
        "gap_queue": gap_queue,
        "capture_plan": capture_plan,
    }


def tensor_metadata(shape: list[int]) -> dict:
    stride = []
    running = 1
    for size in reversed(shape):
        stride.insert(0, running)
        running *= size
    return {
        "kind": "tensor",
        "shape": shape,
        "dtype": "bfloat16",
        "layout": "strided",
        "stride": stride,
    }


def shape_id_for(input_shapes: dict[str, list[int]]) -> str:
    return hashlib.sha256(canonical_json_bytes(input_shapes)).hexdigest()


def write_golden(
    root: Path,
    *,
    contract: dict,
    binding: dict,
    definition: dict,
    ordinal: int,
) -> tuple[Path, Path]:
    session_dir = root / "runs" / "cuda-session"
    session_config = session_dir / "capture-config.json"
    golden = session_dir / "operators" / f"{ordinal:02d}"
    replay = golden / "self-replay"
    samples = golden / "samples"
    replay.mkdir(parents=True)
    samples.mkdir()

    input_shapes = {
        name: [ordinal, position + 2]
        for position, name in enumerate(definition["inputs"])
    }
    shape_id = shape_id_for(input_shapes)
    sample_bytes = f"sample:{definition['operator_id']}".encode()
    sample_path = samples / f"{shape_id}.pt"
    sample_path.write_bytes(sample_bytes)
    checkpoint_id = contract["checkpoint"]["id"]
    model_path, revision = checkpoint_id.rsplit("@", 1)
    checkpoint = {
        "id": checkpoint_id,
        "model_path": model_path,
        "revision": revision,
        "config_digest": contract["checkpoint"]["config_digest"],
    }
    boundary = {
        "inputs": definition["inputs"],
        "parameters": definition["parameters"],
        "non_tensor_args": definition["non_tensor_args"],
        "outputs": definition["outputs"],
    }
    signature = {
        "operator_id": definition["operator_id"],
        "model_path": "target",
        "tensor_parallel_size": contract["runtime"]["tensor_parallel_size"],
        "tp_rank": contract["sample_policy"]["capture_tp_rank"],
        "inputs": {
            name: tensor_metadata(shape)
            for name, shape in input_shapes.items()
        },
        "parameters": {
            name: tensor_metadata([4])
            for name in definition["parameters"]
        },
        "non_tensor_args": {
            name: (0.00001 if name == "eps" else True)
            for name in definition["non_tensor_args"]
        },
        "outputs": {
            name: tensor_metadata([ordinal, 2])
            for name in definition["outputs"]
        },
    }
    write_json(
        golden / "capture-config.json",
        {
            "schema": "golden-capture-config/v1",
            "session_config": str(session_config.resolve()),
            "spec_binding": binding,
            "operator_id": definition["operator_id"],
            "activation_guard": "synthetic-call-reached",
            "model_path": "target",
            "max_shapes": contract["limits"]["max_shapes_per_operator"],
            "tensor_parallel_size": contract["runtime"]["tensor_parallel_size"],
            "tp_rank": contract["sample_policy"]["capture_tp_rank"],
            "serialization": "kernel-call-torch-save/v1",
            "capture_device_type": "cuda",
            "dtype": contract["runtime"]["dtype"],
            "checkpoint": checkpoint,
            "precision_gate": contract["precision_gate"],
            "run_dir": str(golden.resolve()),
            "hook_target": definition["hook_target"],
            "boundary": boundary,
            "sample_fields": boundary,
            "adapter": f"adapter-{ordinal}",
            "replay": {
                "mode": "standalone-kernel-call",
                "cuda_target": definition["hook_target"],
                "p800_target": None,
                "weights_in_golden_sample": False,
            },
            "source": {
                "capture_module_path": f"/synthetic/{ordinal}.py",
                "capture_module_sha256": "0" * 64,
            },
        },
    )
    sidecar = {
        "schema": "golden-sample-files/v1",
        "spec_binding": binding,
        "operator_id": definition["operator_id"],
        "files": [
            {
                "shape_id": shape_id,
                "path": f"samples/{shape_id}.pt",
                "size": len(sample_bytes),
                "sha256": hashlib.sha256(sample_bytes).hexdigest(),
            }
        ],
    }
    write_json(golden / "sample-files.json", sidecar)
    sidecar_sha = hashlib.sha256(
        (golden / "sample-files.json").read_bytes()
    ).hexdigest()
    replay_config = {
        "schema": "kernel-call-replay-config/v1",
        "spec_binding": binding,
        "operator_id": definition["operator_id"],
        "adapter": f"adapter-{ordinal}",
        "sample_fields": boundary,
        "execution_site": "cuda",
        "invocation_target": definition["hook_target"],
        "golden_run": str(golden.resolve()),
        "run_dir": str(replay.resolve()),
        "tensor_parallel_size": contract["runtime"]["tensor_parallel_size"],
        "tp_rank": contract["sample_policy"]["capture_tp_rank"],
        "precision_gate": contract["precision_gate"],
        "sample_files_sha256": sidecar_sha,
        "allow_active_capture": True,
    }
    worker = {
        "schema": "kernel-call-replay-result/v1",
        "spec_binding": binding,
        "operator_id": definition["operator_id"],
        "execution_site": "cuda",
        "invocation_target": definition["hook_target"],
        "tp_rank": contract["sample_policy"]["capture_tp_rank"],
        "tensor_parallel_size": contract["runtime"]["tensor_parallel_size"],
        "precision_gate": contract["precision_gate"],
        "sample_files_sha256": sidecar_sha,
        "actual_tensors_saved": False,
        "checked_shapes": [{"shape_id": shape_id, "passed": True}],
        "checked_shape_count": 1,
        "failed_shape_count": 0,
        "passed": True,
        "errors": [],
    }
    write_json(replay / "replay-config.json", replay_config)
    write_json(replay / "worker-result.json", worker)
    write_json(
        replay / "result.json",
        {
            "tool": "replay_compare.py",
            "action": "kernel-replay",
            "spec_binding": binding,
            "operator_id": definition["operator_id"],
            "execution_site": "cuda",
            "invocation_target": definition["hook_target"],
            "passed": True,
            "checked_shape_count": 1,
            "failed_shape_count": 0,
            "precision_gate": contract["precision_gate"],
            "sample_files_sha256": sidecar_sha,
            "actual_tensors_saved": False,
            "evidence": [
                "replay-config.json",
                "worker-result.json",
                "replay.log",
            ],
        },
    )
    (replay / "replay.log").write_text("execution_site=cuda\n", encoding="utf-8")
    write_json(
        golden / "capture-state.json",
        {
            "schema": "kernel-call-capture-state/v1",
            "spec_binding": binding,
            "operator_id": definition["operator_id"],
            "tp_rank": contract["sample_policy"]["capture_tp_rank"],
            "tensor_parallel_size": contract["runtime"]["tensor_parallel_size"],
            "checkpoint": checkpoint,
            "loaded_checkpoint": {
                "model_path": model_path,
                "revision": revision,
            },
            "capture_session_config": str(session_config.resolve()),
            "capture_process_id": 4242,
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
            "self_replay": {
                "passed": True,
                "checked_shape_count": 1,
                "sample_files_sha256": sidecar_sha,
                "worker_result_sha256": hashlib.sha256(
                    canonical_json_bytes(worker)
                ).hexdigest(),
            },
        },
    )

    record_run = root / "runs" / f"record-{ordinal:02d}"
    record_run.mkdir()
    write_json(
        record_run / "result.json",
        {
            "tool": "handoff_bundle.py",
            "action": "record-samples",
            "spec_binding": binding,
            "operator_id": definition["operator_id"],
            "passed": True,
            "actual_tensors_saved": False,
            "golden_run": str(golden.resolve()),
            "golden_status": "ACTIVE",
            "recorded_before_self_replay": True,
            "sample_file_count": 1,
            "sample_files_path": str((golden / "sample-files.json").resolve()),
            "sample_files_sha256": sidecar_sha,
            "evidence": ["handoff.log"],
        },
    )
    (record_run / "handoff.log").write_text(
        "action=record-samples\npassed=true\n",
        encoding="utf-8",
    )
    return golden, record_run / "result.json"


def run_tool(*arguments: str) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    environment["PYTHONPYCACHEPREFIX"] = "/tmp/model-adaptation-agents-pyc"
    return subprocess.run(
        [sys.executable, str(HANDOFF), *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


class GapDrivenHandoffBundleTest(unittest.TestCase):
    def test_source_run_is_bound_to_the_current_gap_bundle_sources(self) -> None:
        result = json.loads(SOURCE_RUN.read_text(encoding="utf-8"))
        self.assertTrue(result["passed"])
        self.assertEqual(result["execution_site"], "SOURCE")
        self.assertFalse(result["consumes_capture_session"])
        self.assertEqual(result["manifest_schema"], "handoff-manifest/v2")
        self.assertFalse(result["gap_inventory"]["production_operator_constants_in_v2"])
        self.assertTrue(result["capture_session_binding"]["one_session_config"])
        self.assertTrue(result["capture_session_binding"]["one_process"])
        self.assertTrue(result["capture_session_binding"]["bundled"])
        self.assertTrue(result["sample_validation"]["output_tensor_metadata"])
        self.assertTrue(result["compatibility"]["revision_6_requires_scan"])
        for source in result["source_files"]:
            actual = hashlib.sha256(
                (ROOT / source["path"]).read_bytes()
            ).hexdigest()
            self.assertEqual(actual, source["sha256"], source["path"])

    def prepare_workspace(
        self,
        workspace: Path,
        definitions: list[dict],
    ) -> tuple[Path, Path, Path, list[tuple[Path, Path]]]:
        contract = read_contract()
        binding = binding_for(contract)
        current_spec = workspace / "migration-spec.md"
        bundle_spec = workspace / "waiting-spec.md"
        scan_result = workspace / "scan-result.json"
        write_spec(current_spec, contract, "status: ACTIVE")
        write_spec(bundle_spec, contract, "status: WAITING")
        write_json(
            scan_result,
            make_scan(contract, binding, definitions),
        )
        evidence = [
            write_golden(
                workspace,
                contract=contract,
                binding=binding,
                definition=definition,
                ordinal=index,
            )
            for index, definition in enumerate(definitions, start=1)
        ]
        checkpoint_id = contract["checkpoint"]["id"]
        model_path, revision = checkpoint_id.rsplit("@", 1)
        session_dir = workspace / "runs" / "cuda-session"
        session_config = session_dir / "capture-config.json"
        write_json(
            session_config,
            {
                "schema": "golden-capture-session-config/v1",
                "spec_binding": binding,
                "run_dir": str(session_dir.resolve()),
                "scan_result": {
                    "path": str(scan_result.resolve()),
                    "sha256": hashlib.sha256(
                        scan_result.read_bytes()
                    ).hexdigest(),
                },
                "source": {
                    "sglang_revision": contract["source"]["sglang_revision"],
                    "sglang_worktree": "/synthetic/sglang",
                },
                "checkpoint": {
                    "id": checkpoint_id,
                    "model_path": model_path,
                    "revision": revision,
                    "config_digest": contract["checkpoint"]["config_digest"],
                },
                "tensor_parallel_size": contract["runtime"][
                    "tensor_parallel_size"
                ],
                "tp_rank": contract["sample_policy"]["capture_tp_rank"],
                "requests": [],
                "operators": [
                    json.loads(
                        (golden / "capture-config.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    for golden, _ in evidence
                ],
            },
        )
        return current_spec, bundle_spec, scan_result, evidence

    def build(
        self,
        workspace: Path,
        definitions: list[dict],
        *,
        evidence_override: Optional[list[tuple[Path, Path]]] = None,
    ) -> tuple[subprocess.CompletedProcess, Path]:
        current_spec, bundle_spec, scan_result, evidence = (
            self.prepare_workspace(workspace, definitions)
        )
        if evidence_override is not None:
            evidence = evidence_override
        build_run = workspace / "runs" / "handoff-build"
        arguments = [
            "--mode",
            "build",
            "--spec",
            str(current_spec),
            "--run-dir",
            str(build_run),
            "--scan-result",
            str(scan_result),
            "--bundle-spec",
            str(bundle_spec),
        ]
        for golden, record in evidence:
            arguments.extend(["--golden-run", str(golden)])
            arguments.extend(["--sample-record-result", str(record)])
        return run_tool(*arguments), build_run

    def test_build_and_verify_follow_the_scan_gap_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            definitions = list(reversed(operator_definitions()))
            built, build_run = self.build(workspace, definitions)

            self.assertEqual(built.returncode, 0, built.stderr)
            manifest = json.loads(
                (build_run / "bundle" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["schema"], "handoff-manifest/v2")
            self.assertEqual(
                [item["operator_id"] for item in manifest["operators"]],
                [item["operator_id"] for item in definitions],
            )
            self.assertEqual(manifest["scan_result"]["path"], "scan-result.json")
            self.assertTrue(
                (build_run / "bundle" / "scan-result.json").is_file()
            )
            self.assertEqual(
                manifest["capture_session"]["path"],
                "capture-session.json",
            )
            self.assertEqual(
                manifest["capture_session"]["process_id"],
                4242,
            )
            self.assertTrue(
                (build_run / "bundle" / "capture-session.json").is_file()
            )

            copied = workspace / "copied-bundle"
            shutil.copytree(build_run / "bundle", copied)
            verified = run_tool(
                "--mode",
                "verify",
                "--spec",
                str(workspace / "migration-spec.md"),
                "--run-dir",
                str(workspace / "runs" / "handoff-verify"),
                "--bundle-dir",
                str(copied),
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            result = json.loads(
                (
                    workspace
                    / "runs"
                    / "handoff-verify"
                    / "result.json"
                ).read_text(encoding="utf-8")
            )
            self.assertTrue(result["passed"], result)
            self.assertEqual(
                result["operator_ids"],
                [item["operator_id"] for item in definitions],
            )

    def test_one_gap_builds_without_a_production_operator_constant(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            built, build_run = self.build(
                Path(temp_dir),
                operator_definitions()[:1],
            )
            self.assertEqual(built.returncode, 0, built.stderr)
            manifest = json.loads(
                (build_run / "bundle" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                [item["operator_id"] for item in manifest["operators"]],
                ["example.kernel.alpha"],
            )

    def test_record_samples_uses_the_gap_operator_from_capture_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            current_spec, _, scan_result, evidence = self.prepare_workspace(
                workspace,
                operator_definitions()[:1],
            )
            golden = evidence[0][0]
            (golden / "sample-files.json").unlink()
            shutil.rmtree(golden / "self-replay")
            state_path = golden / "capture-state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["status"] = "ACTIVE"
            state["capture_closed"] = False
            state.pop("self_replay")
            write_json(state_path, state)

            recorded = run_tool(
                "--mode",
                "record-samples",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(workspace / "runs" / "record-new"),
                "--scan-result",
                str(scan_result),
                "--golden-run",
                str(golden),
            )
            self.assertEqual(recorded.returncode, 0, recorded.stderr)
            result = json.loads(
                (
                    workspace / "runs" / "record-new" / "result.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(result["operator_id"], "example.kernel.alpha")

    def test_build_rejects_a_missing_gap_golden(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            definitions = operator_definitions()
            current_spec, bundle_spec, scan_result, evidence = (
                self.prepare_workspace(workspace, definitions)
            )
            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(workspace / "runs" / "handoff-build"),
                "--scan-result",
                str(scan_result),
                "--bundle-spec",
                str(bundle_spec),
                "--golden-run",
                str(evidence[0][0]),
                "--sample-record-result",
                str(evidence[0][1]),
            )
            self.assertEqual(built.returncode, 2)
            self.assertIn("missing Golden", built.stderr)

    def test_build_rejects_an_extra_operator_golden(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            definitions = operator_definitions()
            current_spec, bundle_spec, scan_result, evidence = (
                self.prepare_workspace(workspace, definitions)
            )
            contract = read_contract()
            binding = binding_for(contract)
            extra_definition = {
                "operator_id": "example.kernel.not-in-scan",
                "hook_target": "example.calls.extra",
                "inputs": ["value"],
                "parameters": [],
                "non_tensor_args": [],
                "outputs": ["output"],
            }
            evidence.append(
                write_golden(
                    workspace,
                    contract=contract,
                    binding=binding,
                    definition=extra_definition,
                    ordinal=3,
                )
            )
            arguments = [
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(workspace / "runs" / "handoff-build"),
                "--scan-result",
                str(scan_result),
                "--bundle-spec",
                str(bundle_spec),
            ]
            for golden, record in evidence:
                arguments.extend(["--golden-run", str(golden)])
                arguments.extend(["--sample-record-result", str(record)])
            built = run_tool(*arguments)
            self.assertEqual(built.returncode, 2)
            self.assertIn("not present in the Scan gap queue", built.stderr)

    def test_build_rejects_a_duplicate_gap_golden(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            definitions = operator_definitions()
            current_spec, bundle_spec, scan_result, evidence = (
                self.prepare_workspace(workspace, definitions)
            )
            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(workspace / "runs" / "handoff-build"),
                "--scan-result",
                str(scan_result),
                "--bundle-spec",
                str(bundle_spec),
                "--golden-run",
                str(evidence[0][0]),
                "--golden-run",
                str(evidence[0][0]),
                "--golden-run",
                str(evidence[1][0]),
                "--sample-record-result",
                str(evidence[0][1]),
                "--sample-record-result",
                str(evidence[1][1]),
            )
            self.assertEqual(built.returncode, 2)
            self.assertIn("duplicate Golden Run", built.stderr)

    def test_build_rejects_goldens_from_different_capture_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            definitions = operator_definitions()
            current_spec, bundle_spec, scan_result, evidence = (
                self.prepare_workspace(workspace, definitions)
            )
            second_golden = evidence[1][0]
            other_session = workspace / "runs" / "other-session" / "capture-config.json"
            config_path = second_golden / "capture-config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["session_config"] = str(other_session.resolve())
            write_json(config_path, config)
            state_path = second_golden / "capture-state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["capture_session_config"] = str(other_session.resolve())
            write_json(state_path, state)

            arguments = [
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(workspace / "runs" / "handoff-build"),
                "--scan-result",
                str(scan_result),
                "--bundle-spec",
                str(bundle_spec),
            ]
            for golden, record in evidence:
                arguments.extend(["--golden-run", str(golden)])
                arguments.extend(["--sample-record-result", str(record)])
            built = run_tool(*arguments)
            self.assertEqual(built.returncode, 2)
            self.assertIn("one capture session", built.stderr)

    def test_build_rejects_goldens_from_different_capture_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            definitions = operator_definitions()
            current_spec, bundle_spec, scan_result, evidence = (
                self.prepare_workspace(workspace, definitions)
            )
            state_path = evidence[1][0] / "capture-state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["capture_process_id"] = 4343
            write_json(state_path, state)

            arguments = [
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(workspace / "runs" / "handoff-build"),
                "--scan-result",
                str(scan_result),
                "--bundle-spec",
                str(bundle_spec),
            ]
            for golden, record in evidence:
                arguments.extend(["--golden-run", str(golden)])
                arguments.extend(["--sample-record-result", str(record)])
            built = run_tool(*arguments)
            self.assertEqual(built.returncode, 2)
            self.assertIn("one capture process", built.stderr)

    def test_build_rejects_missing_output_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            definitions = operator_definitions()[:1]
            current_spec, bundle_spec, scan_result, evidence = (
                self.prepare_workspace(workspace, definitions)
            )
            golden = evidence[0][0]
            state_path = golden / "capture-state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["samples"][0]["signature"].pop("outputs")
            write_json(state_path, state)

            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(workspace / "runs" / "handoff-build"),
                "--scan-result",
                str(scan_result),
                "--bundle-spec",
                str(bundle_spec),
                "--golden-run",
                str(golden),
                "--sample-record-result",
                str(evidence[0][1]),
            )
            self.assertEqual(built.returncode, 2)
            self.assertIn("signature outputs", built.stderr)

    def test_revision_6_build_cannot_fall_back_to_legacy_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            current_spec, bundle_spec, _, evidence = self.prepare_workspace(
                workspace,
                operator_definitions()[:1],
            )
            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(workspace / "runs" / "handoff-build"),
                "--bundle-spec",
                str(bundle_spec),
                "--golden-run",
                str(evidence[0][0]),
                "--sample-record-result",
                str(evidence[0][1]),
            )
            self.assertEqual(built.returncode, 2)
            self.assertIn("requires --scan-result", built.stderr)

    def test_build_rejects_a_symbolic_link_golden_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            definitions = operator_definitions()[:1]
            current_spec, bundle_spec, scan_result, evidence = (
                self.prepare_workspace(workspace, definitions)
            )
            linked_golden = workspace / "linked-golden"
            linked_golden.symlink_to(evidence[0][0], target_is_directory=True)
            built = run_tool(
                "--mode",
                "build",
                "--spec",
                str(current_spec),
                "--run-dir",
                str(workspace / "runs" / "handoff-build"),
                "--scan-result",
                str(scan_result),
                "--bundle-spec",
                str(bundle_spec),
                "--golden-run",
                str(linked_golden),
                "--sample-record-result",
                str(evidence[0][1]),
            )
            self.assertEqual(built.returncode, 2)
            self.assertIn("symbolic link", built.stderr)

    def test_verify_rejects_manifest_operator_drift_from_bundled_scan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            built, build_run = self.build(workspace, operator_definitions())
            self.assertEqual(built.returncode, 0, built.stderr)
            bundle = build_run / "bundle"
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["operators"] = manifest["operators"][:1]
            write_json(manifest_path, manifest)

            verified = run_tool(
                "--mode",
                "verify",
                "--spec",
                str(workspace / "migration-spec.md"),
                "--run-dir",
                str(workspace / "runs" / "handoff-verify"),
                "--bundle-dir",
                str(bundle),
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            result = json.loads(
                (
                    workspace
                    / "runs"
                    / "handoff-verify"
                    / "result.json"
                ).read_text(encoding="utf-8")
            )
            self.assertFalse(result["passed"])
            self.assertIn("does not match the Scan gap queue", result["error"])


if __name__ == "__main__":
    unittest.main()
