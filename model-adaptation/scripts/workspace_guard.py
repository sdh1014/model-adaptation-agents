#!/usr/bin/env python3
"""Guard one fixed Git baseline across a bounded P800 repair attempt."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Dict, Iterable

from _lib.spec_contract import (
    SpecContractError,
    load_contract_data,
    load_spec_binding,
)
from model_adaptation_capture.contracts import (
    KERNEL_REPLAY_CONFIG_SCHEMA,
    KUNLUN_SWIGLU_TARGET,
    SWIGLU_CLAMP_OPERATOR_ID,
    checkpoint_metadata,
)
from replay_compare import (
    ToolError as ReplayToolError,
    _kernel_sample_files_sha256,
    _load_kernel_capture_state,
    _validate_kernel_worker_result,
)


ATTEMPT_SCHEMA = "repair-attempt/v1"
ATTEMPT_CLAIM_SCHEMA = "repair-attempt-claim/v1"
CANDIDATE_SCHEMA = "repair-candidate/v1"
OUTCOME_SCHEMA = "repair-outcome/v1"
TEST_SPEC_ID = "ticket14-repair-loop-test"


class ToolError(RuntimeError):
    """The workspace cannot be changed without risking user work."""


def read_json_object(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ToolError(f"{label} must be one regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ToolError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ToolError(f"{label} must be one JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ToolError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def create_run_dir(run_dir: Path) -> None:
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise ToolError(
            f"run directory must be fresh and creatable: {error}"
        ) from error


def run_git(
    worktree: Path,
    arguments: Iterable[str],
    *,
    text: bool = True,
) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=worktree,
        text=text,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = (
            completed.stderr.strip()
            if text
            else completed.stderr.decode(errors="replace").strip()
        )
        raise ToolError(
            f"git {' '.join(arguments)} failed: {stderr}"
        )
    return completed


def git_output_paths(worktree: Path, arguments: list[str]) -> set[str]:
    completed = run_git(worktree, arguments, text=False)
    return {
        os.fsdecode(raw)
        for raw in completed.stdout.split(b"\0")
        if raw
    }


def contract_context(
    spec_path: Path,
) -> tuple[Dict[str, Any], Dict[str, Any], str, int]:
    binding = load_spec_binding(spec_path).as_result_dict()
    contract = load_contract_data(spec_path)
    try:
        baseline = contract["source"]["sglang_kunlun_revision"]
        max_attempts = contract["limits"]["max_repair_attempts"]
    except (KeyError, TypeError) as error:
        raise ToolError(
            "Contract is missing the P800 repair baseline or attempt limit"
        ) from error
    if not isinstance(baseline, str) or not baseline:
        raise ToolError("Contract P800 repair baseline is invalid")
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts < 1
    ):
        raise ToolError("Contract max_repair_attempts is invalid")
    return binding, contract, baseline, max_attempts


def validate_worktree(worktree: Path, baseline: str) -> Path:
    worktree = worktree.resolve()
    if not worktree.is_dir():
        raise ToolError(f"worktree does not exist: {worktree}")
    top_level = Path(
        run_git(
            worktree,
            ["rev-parse", "--show-toplevel"],
        ).stdout.strip()
    ).resolve()
    if top_level != worktree:
        raise ToolError("worktree must be the Git repository root")
    head = run_git(worktree, ["rev-parse", "HEAD"]).stdout.strip()
    if head != baseline:
        raise ToolError(
            "worktree HEAD does not match Contract "
            f"sglang_kunlun_revision: expected {baseline}, got {head}"
        )
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--"],
        cwd=worktree,
        capture_output=True,
        check=False,
    )
    if staged.returncode not in {0, 1}:
        raise ToolError(
            "cannot inspect staged changes: "
            + staged.stderr.decode(errors="replace").strip()
        )
    if staged.returncode == 1:
        raise ToolError(
            "staged workspace changes are not allowed in the repair loop"
        )
    return worktree


def validate_allowed_paths(
    worktree: Path,
    raw_paths: list[str],
) -> list[str]:
    if not raw_paths:
        raise ToolError("at least one --allowed-path is required")
    normalized = []
    for raw in raw_paths:
        pure = PurePosixPath(raw)
        if (
            not raw
            or pure.is_absolute()
            or str(pure) != raw
            or raw.endswith("/")
            or raw in {".", ".."}
            or ".." in pure.parts
            or ".git" in pure.parts
        ):
            raise ToolError(f"allowed path is unsafe: {raw!r}")
        candidate = worktree.joinpath(*pure.parts)
        cursor = worktree
        for part in pure.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ToolError(
                    f"allowed path must not contain a symbolic link: {raw}"
                )
        if candidate.exists() and not candidate.is_file():
            raise ToolError(f"allowed path must name one file: {raw}")
        normalized.append(raw)
    if len(normalized) != len(set(normalized)):
        raise ToolError("allowed paths must be unique")
    return sorted(normalized)


def changed_paths(worktree: Path) -> set[str]:
    tracked = git_output_paths(
        worktree,
        [
            "diff",
            "--name-only",
            "-z",
            "--no-renames",
            "HEAD",
            "--",
        ],
    )
    untracked = git_output_paths(
        worktree,
        [
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
    )
    return tracked | untracked


def require_known_changes(
    worktree: Path,
    allowed_paths: list[str],
) -> set[str]:
    changes = changed_paths(worktree)
    unknown = sorted(changes - set(allowed_paths))
    if unknown:
        raise ToolError(
            "unknown workspace changes: " + ", ".join(unknown)
        )
    return changes


def require_clean_baseline(
    worktree: Path,
    allowed_paths: list[str],
) -> None:
    changes = require_known_changes(worktree, allowed_paths)
    if changes:
        raise ToolError(
            "workspace must be clean at the fixed baseline: "
            + ", ".join(sorted(changes))
        )


def replay_evidence_sha256(
    result_path: Path,
    evidence_names: list[str],
) -> str:
    records = []
    for path in [
        result_path,
        *(result_path.parent / name for name in evidence_names),
    ]:
        if path.is_symlink() or not path.is_file():
            raise ToolError(f"replay evidence is missing: {path.name}")
        records.append(
            {
                "path": path.name,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    payload = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_p800_replay_evidence(
    result_path: Path,
    result: Dict[str, Any],
    binding: Dict[str, Any],
    contract: Dict[str, Any],
) -> str:
    evidence_names = [
        "replay-config.json",
        "worker-result.json",
        "replay.log",
    ]
    if result.get("evidence") != evidence_names:
        raise ToolError("P800 replay evidence list has drifted")
    config = read_json_object(
        result_path.parent / "replay-config.json",
        "P800 replay config",
    )
    worker = read_json_object(
        result_path.parent / "worker-result.json",
        "P800 replay worker result",
    )
    golden_raw = config.get("golden_run")
    if not isinstance(golden_raw, str) or not golden_raw:
        raise ToolError("P800 replay golden_run is invalid")
    golden_run = Path(golden_raw)
    if not golden_run.is_absolute() or golden_run.is_symlink():
        raise ToolError(
            "P800 replay golden_run must be absolute and not a symlink"
        )
    golden_run = golden_run.resolve()
    try:
        checkpoint = checkpoint_metadata(
            contract["checkpoint"]["id"],
            contract["checkpoint"]["config_digest"],
        )
        state = _load_kernel_capture_state(
            golden_run,
            binding,
            checkpoint,
            execution_site="p800",
        )
        sample_files_digest = _kernel_sample_files_sha256(
            golden_run,
            binding,
            SWIGLU_CLAMP_OPERATOR_ID,
            state,
        )
    except (KeyError, TypeError, ValueError, ReplayToolError) as error:
        raise ToolError(
            f"P800 Golden/replay evidence is invalid: {error}"
        ) from error

    expected_config = {
        "schema": KERNEL_REPLAY_CONFIG_SCHEMA,
        "spec_binding": binding,
        "operator_id": SWIGLU_CLAMP_OPERATOR_ID,
        "execution_site": "p800",
        "invocation_target": KUNLUN_SWIGLU_TARGET,
        "golden_run": str(golden_run),
        "run_dir": str(result_path.parent.resolve()),
        "tensor_parallel_size": contract["runtime"][
            "tensor_parallel_size"
        ],
        "tp_rank": contract["sample_policy"]["capture_tp_rank"],
        "precision_gate": contract["precision_gate"],
        "sample_files_sha256": sample_files_digest,
        "allow_active_capture": False,
    }
    if config != expected_config:
        raise ToolError("P800 replay config has drifted")
    try:
        _validate_kernel_worker_result(
            worker,
            config,
            [sample["shape_id"] for sample in state["samples"]],
        )
    except ReplayToolError as error:
        raise ToolError(
            f"P800 replay worker evidence is invalid: {error}"
        ) from error

    wrapper_checks = {
        "tool": "replay_compare.py",
        "action": "kernel-replay",
        "spec_binding": binding,
        "operator_id": SWIGLU_CLAMP_OPERATOR_ID,
        "execution_site": "p800",
        "invocation_target": KUNLUN_SWIGLU_TARGET,
        "passed": worker["passed"],
        "checked_shape_count": worker["checked_shape_count"],
        "failed_shape_count": worker["failed_shape_count"],
        "precision_gate": contract["precision_gate"],
        "sample_files_sha256": sample_files_digest,
        "actual_tensors_saved": False,
        "evidence": evidence_names,
    }
    for field, expected in wrapper_checks.items():
        if result.get(field) != expected:
            raise ToolError(f"P800 replay result {field} has drifted")
    log_path = result_path.parent / "replay.log"
    if log_path.is_symlink() or not log_path.is_file():
        raise ToolError("P800 replay log is missing")
    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    expected_returncode = 0 if worker["passed"] else 1
    expected_prefix = [
        "execution_site=p800",
        f"invocation_target={KUNLUN_SWIGLU_TARGET}",
        f"returncode={expected_returncode}",
    ]
    if log_lines[:3] != expected_prefix:
        raise ToolError("P800 replay process result has drifted")
    return replay_evidence_sha256(result_path, evidence_names)


def validate_replay_result(
    result_path: Path,
    binding: Dict[str, Any],
    contract: Dict[str, Any],
    *,
    allow_synthetic_replay: bool,
) -> tuple[Dict[str, Any], str, str]:
    result = read_json_object(result_path, "replay result")
    checks = {
        "tool": "replay_compare.py",
        "spec_binding": binding,
    }
    for field, expected in checks.items():
        if result.get(field) != expected:
            raise ToolError(f"replay result {field} has drifted")
    if not isinstance(result.get("passed"), bool):
        raise ToolError("replay result passed must be a boolean")
    action = result.get("action")
    if action == "synthetic":
        if binding.get("spec_id") != TEST_SPEC_ID:
            raise ToolError(
                "synthetic replay is test-only and cannot consume the "
                "production Migration Spec"
            )
        if not allow_synthetic_replay:
            raise ToolError(
                "synthetic replay requires the explicit test-only flag"
            )
        evidence_names = ["synthetic-case.json", "replay.log"]
        if result.get("evidence") != evidence_names:
            raise ToolError("synthetic replay evidence list has drifted")
        case = read_json_object(
            result_path.parent / "synthetic-case.json",
            "synthetic replay case",
        )
        if set(case) != {"expected", "actual"}:
            raise ToolError("synthetic replay case has drifted")
        if result["passed"] != (case["expected"] == case["actual"]):
            raise ToolError("synthetic replay result is inconsistent")
        evidence_digest = replay_evidence_sha256(
            result_path,
            evidence_names,
        )
    else:
        if action != "kernel-replay":
            raise ToolError(
                "P800 repair requires replay_compare.py kernel-replay"
            )
        if result.get("execution_site") != "p800":
            raise ToolError(
                "P800 repair requires execution_site p800"
            )
        evidence_digest = validate_p800_replay_evidence(
            result_path,
            result,
            binding,
            contract,
        )
    return result, file_sha256(result_path), evidence_digest


def common_result(
    *,
    action: str,
    binding: Dict[str, Any],
    baseline: str,
    worktree: Path,
    allowed_paths: list[str],
) -> Dict[str, Any]:
    return {
        "tool": "workspace_guard.py",
        "action": action,
        "spec_binding": binding,
        "baseline_revision": baseline,
        "worktree": str(worktree),
        "allowed_paths": allowed_paths,
    }


def run_check_baseline(
    spec_path: Path,
    run_dir: Path,
    worktree: Path,
    raw_allowed_paths: list[str],
) -> None:
    binding, _contract, baseline, _ = contract_context(spec_path)
    worktree = validate_worktree(worktree, baseline)
    allowed_paths = validate_allowed_paths(worktree, raw_allowed_paths)
    require_clean_baseline(worktree, allowed_paths)
    create_run_dir(run_dir)
    result = {
        **common_result(
            action="check-baseline",
            binding=binding,
            baseline=baseline,
            worktree=worktree,
            allowed_paths=allowed_paths,
        ),
        "passed": True,
        "workspace_clean": True,
        "evidence": ["workspace.log"],
    }
    write_json(run_dir / "result.json", result)
    (run_dir / "workspace.log").write_text(
        "action=check-baseline\n"
        "passed=true\n"
        "workspace_clean=true\n",
        encoding="utf-8",
    )


def run_assess_baseline(
    spec_path: Path,
    run_dir: Path,
    worktree: Path,
    raw_allowed_paths: list[str],
    replay_result_path: Path,
    allow_synthetic_replay: bool,
) -> None:
    binding, contract, baseline, _ = contract_context(spec_path)
    worktree = validate_worktree(worktree, baseline)
    allowed_paths = validate_allowed_paths(worktree, raw_allowed_paths)
    require_clean_baseline(worktree, allowed_paths)
    (
        replay_result,
        replay_digest,
        replay_evidence_digest,
    ) = validate_replay_result(
        replay_result_path,
        binding,
        contract,
        allow_synthetic_replay=allow_synthetic_replay,
    )
    create_run_dir(run_dir)
    passed = replay_result["passed"]
    result = {
        **common_result(
            action="assess-baseline",
            binding=binding,
            baseline=baseline,
            worktree=worktree,
            allowed_paths=allowed_paths,
        ),
        "passed": passed,
        "gap_observed": not passed,
        "attempts_used": 0,
        "workspace_clean": True,
        "replay_result": str(replay_result_path.resolve()),
        "replay_result_sha256": replay_digest,
        "replay_evidence_sha256": replay_evidence_digest,
        "evidence": ["workspace.log"],
    }
    write_json(run_dir / "result.json", result)
    (run_dir / "workspace.log").write_text(
        "action=assess-baseline\n"
        f"replay_passed={str(passed).lower()}\n"
        f"gap_observed={str(not passed).lower()}\n"
        "attempts_used=0\n"
        "workspace_clean=true\n",
        encoding="utf-8",
    )


def write_exclusive_json(path: Path, value: Dict[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
    except OSError as error:
        raise ToolError(
            f"cannot claim repair attempt: {error}"
        ) from error


def attempt_history_directory(
    worktree: Path,
    *,
    binding: Dict[str, Any],
    baseline: str,
) -> Path:
    identity = {
        "spec_binding": binding,
        "baseline_revision": baseline,
        "worktree": str(worktree),
    }
    digest = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    git_path_raw = run_git(
        worktree,
        [
            "rev-parse",
            "--git-path",
            "model-adaptation-repair",
        ],
    ).stdout.strip()
    git_path = Path(git_path_raw)
    if not git_path.is_absolute():
        git_path = worktree / git_path
    return git_path.resolve() / digest


def validate_previous_result(
    result_path: Path,
    *,
    attempt: int,
    binding: Dict[str, Any],
    baseline: str,
    worktree: Path,
    allowed_paths: list[str],
) -> tuple[Path, str, Path]:
    if result_path.is_symlink():
        raise ToolError("previous repair result must not be a symlink")
    result_path = result_path.resolve()
    previous = read_json_object(result_path, "previous repair result")
    checks = {
        "tool": "workspace_guard.py",
        "spec_binding": binding,
        "baseline_revision": baseline,
        "worktree": str(worktree),
        "allowed_paths": allowed_paths,
    }
    for field, expected in checks.items():
        if previous.get(field) != expected:
            raise ToolError(f"previous repair result {field} has drifted")

    previous_digest = file_sha256(result_path)
    if attempt == 1:
        baseline_checks = {
            "action": "assess-baseline",
            "passed": False,
            "gap_observed": True,
            "attempts_used": 0,
            "workspace_clean": True,
        }
        for field, expected in baseline_checks.items():
            if previous.get(field) != expected:
                raise ToolError(
                    "attempt 1 requires one failed baseline assessment"
                )
        history_dir = attempt_history_directory(
            worktree,
            binding=binding,
            baseline=baseline,
        )
    else:
        attempt_checks = {
            "action": "finish-attempt",
            "passed": False,
            "attempt": attempt - 1,
            "attempt_limit_reached": False,
            "workspace_state": "BASELINE_RESTORED",
        }
        for field, expected in attempt_checks.items():
            if previous.get(field) != expected:
                raise ToolError(
                    f"attempt {attempt} requires failed attempt "
                    f"{attempt - 1}"
                )
        raw_history = previous.get("attempt_history")
        if not isinstance(raw_history, str) or not raw_history:
            raise ToolError(
                "previous repair result is missing attempt history"
            )
        history_path = Path(raw_history)
        if not history_path.is_absolute() or history_path.is_symlink():
            raise ToolError(
                "attempt history must be absolute and not a symbolic link"
            )
        history_dir = history_path.resolve()

    if history_dir.exists():
        if history_dir.is_symlink() or not history_dir.is_dir():
            raise ToolError("attempt history must be one directory")
        actual_names = sorted(path.name for path in history_dir.iterdir())
    else:
        actual_names = []
    expected_names = [
        f"attempt-{number:03d}.json"
        for number in range(1, attempt)
    ]
    current_name = f"attempt-{attempt:03d}.json"
    if current_name in actual_names:
        raise ToolError(f"attempt {attempt} is already claimed")
    if actual_names != expected_names:
        raise ToolError(
            "repair attempt history is not one complete consecutive chain"
        )

    for number, name in enumerate(expected_names, start=1):
        claim = read_json_object(
            history_dir / name,
            f"attempt {number} claim",
        )
        claim_checks = {
            "schema": ATTEMPT_CLAIM_SCHEMA,
            "spec_binding": binding,
            "baseline_revision": baseline,
            "worktree": str(worktree),
            "allowed_paths": allowed_paths,
            "attempt": number,
        }
        for field, expected in claim_checks.items():
            if claim.get(field) != expected:
                raise ToolError(f"attempt {number} claim has drifted")
    if attempt > 1:
        previous_claim = read_json_object(
            history_dir / expected_names[-1],
            "previous attempt claim",
        )
        if previous_claim.get("result_path") != str(result_path):
            raise ToolError(
                "previous result does not match the consecutive "
                "attempt claim"
            )
    return result_path, previous_digest, history_dir


def run_start_attempt(
    spec_path: Path,
    run_dir: Path,
    worktree: Path,
    raw_allowed_paths: list[str],
    attempt: int,
    hypothesis: str,
    previous_result_path: Path,
) -> None:
    binding, _contract, baseline, max_attempts = contract_context(spec_path)
    if (
        isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or not 1 <= attempt <= max_attempts
    ):
        raise ToolError(
            "attempt must be between 1 and Contract max_repair_attempts "
            f"({max_attempts})"
        )
    if (
        not isinstance(hypothesis, str)
        or not hypothesis
        or hypothesis.strip() != hypothesis
        or "\n" in hypothesis
    ):
        raise ToolError("hypothesis must be one non-empty line")
    worktree = validate_worktree(worktree, baseline)
    allowed_paths = validate_allowed_paths(worktree, raw_allowed_paths)
    require_clean_baseline(worktree, allowed_paths)
    (
        previous_result_path,
        previous_result_digest,
        history_dir,
    ) = validate_previous_result(
        previous_result_path,
        attempt=attempt,
        binding=binding,
        baseline=baseline,
        worktree=worktree,
        allowed_paths=allowed_paths,
    )
    create_run_dir(run_dir)
    try:
        history_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ToolError(
            f"cannot create repair attempt history: {error}"
        ) from error
    attempt_record = {
        "schema": ATTEMPT_SCHEMA,
        **common_result(
            action="start-attempt",
            binding=binding,
            baseline=baseline,
            worktree=worktree,
            allowed_paths=allowed_paths,
        ),
        "attempt": attempt,
        "max_repair_attempts": max_attempts,
        "hypothesis": hypothesis,
        "previous_result": str(previous_result_path),
        "previous_result_sha256": previous_result_digest,
        "attempt_history": str(history_dir),
        "workspace_clean": True,
    }
    write_json(run_dir / "attempt.json", attempt_record)
    claim = {
        "schema": ATTEMPT_CLAIM_SCHEMA,
        "spec_binding": binding,
        "baseline_revision": baseline,
        "worktree": str(worktree),
        "allowed_paths": allowed_paths,
        "attempt": attempt,
        "run_dir": str(run_dir.resolve()),
        "result_path": str((run_dir / "result.json").resolve()),
        "previous_result": str(previous_result_path),
        "previous_result_sha256": previous_result_digest,
    }
    write_exclusive_json(
        history_dir / f"attempt-{attempt:03d}.json",
        claim,
    )
    (run_dir / "workspace.log").write_text(
        "action=start-attempt\n"
        f"attempt={attempt}\n"
        f"max_repair_attempts={max_attempts}\n"
        f"hypothesis={hypothesis}\n"
        f"previous_result_sha256={previous_result_digest}\n"
        "workspace_clean=true\n",
        encoding="utf-8",
    )


def baseline_contains(
    worktree: Path,
    baseline: str,
    relative_path: str,
) -> bool:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{baseline}:{relative_path}"],
        cwd=worktree,
        capture_output=True,
        check=False,
    )
    if completed.returncode not in {0, 1, 128}:
        raise ToolError(
            "cannot inspect baseline path "
            f"{relative_path}: "
            + completed.stderr.decode(errors="replace").strip()
        )
    return completed.returncode == 0


def build_candidate_patch(
    worktree: Path,
    baseline: str,
    paths: set[str],
) -> bytes:
    tracked = sorted(
        path
        for path in paths
        if baseline_contains(worktree, baseline, path)
    )
    untracked = sorted(paths - set(tracked))
    chunks = []
    if tracked:
        chunks.append(
            run_git(
                worktree,
                [
                    "diff",
                    "--binary",
                    "--no-ext-diff",
                    baseline,
                    "--",
                    *tracked,
                ],
                text=False,
            ).stdout
        )
    for path in untracked:
        completed = subprocess.run(
            [
                "git",
                "diff",
                "--no-index",
                "--binary",
                "--",
                os.devnull,
                path,
            ],
            cwd=worktree,
            capture_output=True,
            check=False,
        )
        if completed.returncode not in {0, 1}:
            raise ToolError(
                f"cannot create patch for {path}: "
                + completed.stderr.decode(errors="replace").strip()
            )
        chunks.append(completed.stdout)
    patch = b"".join(chunks)
    if not patch:
        raise ToolError("candidate patch is empty")
    return patch


def record_candidate_patch(
    run_dir: Path,
    worktree: Path,
    baseline: str,
    changes: set[str],
) -> tuple[bytes, list[str]]:
    patch_path = run_dir / "candidate.patch"
    metadata_path = run_dir / "candidate.json"
    if patch_path.exists() or metadata_path.exists():
        raise ToolError("candidate evidence already exists")
    if not changes:
        raise ToolError("attempt has no candidate workspace changes")
    patch = build_candidate_patch(worktree, baseline, changes)
    patch_path.write_bytes(patch)
    metadata = {
        "schema": CANDIDATE_SCHEMA,
        "modified_paths": sorted(changes),
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
    }
    write_json(metadata_path, metadata)
    return patch, metadata["modified_paths"]


def restore_candidate(
    worktree: Path,
    baseline: str,
    modified_paths: list[str],
) -> None:
    tracked = [
        path
        for path in modified_paths
        if baseline_contains(worktree, baseline, path)
    ]
    untracked = sorted(set(modified_paths) - set(tracked))
    if tracked:
        run_git(
            worktree,
            [
                "restore",
                "--source",
                baseline,
                "--worktree",
                "--",
                *tracked,
            ],
        )
    for relative in untracked:
        path = worktree.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            raise ToolError(
                f"refusing to remove non-file candidate path: {relative}"
            )


def load_attempt(
    run_dir: Path,
    *,
    binding: Dict[str, Any],
    baseline: str,
    max_attempts: int,
    worktree: Path,
) -> Dict[str, Any]:
    attempt = read_json_object(run_dir / "attempt.json", "repair attempt")
    checks = {
        "schema": ATTEMPT_SCHEMA,
        "tool": "workspace_guard.py",
        "action": "start-attempt",
        "spec_binding": binding,
        "baseline_revision": baseline,
        "worktree": str(worktree),
        "max_repair_attempts": max_attempts,
        "workspace_clean": True,
    }
    for field, expected in checks.items():
        if attempt.get(field) != expected:
            raise ToolError(f"repair attempt {field} has drifted")
    number = attempt.get("attempt")
    if (
        isinstance(number, bool)
        or not isinstance(number, int)
        or not 1 <= number <= max_attempts
    ):
        raise ToolError("repair attempt number is invalid")
    hypothesis = attempt.get("hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis:
        raise ToolError("repair attempt hypothesis is invalid")
    allowed_paths = validate_allowed_paths(
        worktree,
        attempt.get("allowed_paths", []),
    )
    if attempt["allowed_paths"] != allowed_paths:
        raise ToolError("repair attempt allowed_paths have drifted")
    previous_raw = attempt.get("previous_result")
    previous_digest = attempt.get("previous_result_sha256")
    history_raw = attempt.get("attempt_history")
    if (
        not isinstance(previous_raw, str)
        or not previous_raw
        or not isinstance(previous_digest, str)
        or len(previous_digest) != 64
        or not isinstance(history_raw, str)
        or not history_raw
    ):
        raise ToolError("repair attempt history binding is invalid")
    previous_path = Path(previous_raw)
    history_dir = Path(history_raw)
    if previous_path.is_symlink() or history_dir.is_symlink():
        raise ToolError("repair attempt history must not use symlinks")
    if file_sha256(previous_path) != previous_digest:
        raise ToolError("previous repair result has changed")
    if not history_dir.is_dir():
        raise ToolError("repair attempt history directory is missing")
    claim = read_json_object(
        history_dir / f"attempt-{number:03d}.json",
        "repair attempt claim",
    )
    claim_checks = {
        "schema": ATTEMPT_CLAIM_SCHEMA,
        "spec_binding": binding,
        "baseline_revision": baseline,
        "worktree": str(worktree),
        "allowed_paths": allowed_paths,
        "attempt": number,
        "run_dir": str(run_dir.resolve()),
        "result_path": str((run_dir / "result.json").resolve()),
        "previous_result": previous_raw,
        "previous_result_sha256": previous_digest,
    }
    for field, expected in claim_checks.items():
        if claim.get(field) != expected:
            raise ToolError(f"repair attempt claim {field} has drifted")
    return attempt


def run_record_candidate(
    spec_path: Path,
    run_dir: Path,
    worktree: Path,
) -> None:
    binding, _contract, baseline, max_attempts = contract_context(spec_path)
    worktree = validate_worktree(worktree, baseline)
    if run_dir.is_symlink():
        raise ToolError("attempt run directory must not be a symlink")
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise ToolError("attempt run directory does not exist")
    forbidden = (
        "candidate.patch",
        "candidate.json",
        "candidate-result.json",
        "outcome.json",
        "result.json",
    )
    if any((run_dir / name).exists() for name in forbidden):
        raise ToolError("candidate is already recorded or attempt is sealed")
    if (run_dir / "replay").exists():
        raise ToolError(
            "candidate must be recorded before creating the replay Run"
        )
    attempt = load_attempt(
        run_dir,
        binding=binding,
        baseline=baseline,
        max_attempts=max_attempts,
        worktree=worktree,
    )
    changes = require_known_changes(
        worktree,
        attempt["allowed_paths"],
    )
    patch, modified_paths = record_candidate_patch(
        run_dir,
        worktree,
        baseline,
        changes,
    )
    patch_digest = hashlib.sha256(patch).hexdigest()
    result = {
        **common_result(
            action="record-candidate",
            binding=binding,
            baseline=baseline,
            worktree=worktree,
            allowed_paths=attempt["allowed_paths"],
        ),
        "passed": True,
        "attempt": attempt["attempt"],
        "attempt_history": attempt["attempt_history"],
        "hypothesis": attempt["hypothesis"],
        "modified_paths": modified_paths,
        "patch_path": "candidate.patch",
        "patch_sha256": patch_digest,
        "workspace_state": "CANDIDATE_RECORDED",
        "evidence": [
            "attempt.json",
            "candidate.json",
            "candidate.patch",
            "workspace.log",
        ],
    }
    atomic_write_json(run_dir / "candidate-result.json", result)
    with (run_dir / "workspace.log").open(
        "a",
        encoding="utf-8",
    ) as stream:
        stream.write(
            "action=record-candidate\n"
            f"patch_sha256={patch_digest}\n"
            "workspace_state=CANDIDATE_RECORDED\n"
        )


def load_recorded_candidate(
    run_dir: Path,
    *,
    binding: Dict[str, Any],
    baseline: str,
    worktree: Path,
    attempt: Dict[str, Any],
    changes: set[str],
) -> tuple[bytes, list[str], Dict[str, Any]]:
    record = read_json_object(
        run_dir / "candidate-result.json",
        "candidate record",
    )
    metadata = read_json_object(
        run_dir / "candidate.json",
        "candidate metadata",
    )
    patch_path = run_dir / "candidate.patch"
    if patch_path.is_symlink() or not patch_path.is_file():
        raise ToolError("candidate patch must be one regular file")
    patch = patch_path.read_bytes()
    patch_digest = hashlib.sha256(patch).hexdigest()
    modified_paths = metadata.get("modified_paths")
    if (
        metadata.get("schema") != CANDIDATE_SCHEMA
        or metadata.get("patch_sha256") != patch_digest
        or not isinstance(modified_paths, list)
        or not modified_paths
        or len(modified_paths) != len(set(modified_paths))
        or not set(modified_paths).issubset(
            set(attempt["allowed_paths"])
        )
    ):
        raise ToolError("candidate metadata has drifted")
    record_checks = {
        "tool": "workspace_guard.py",
        "action": "record-candidate",
        "spec_binding": binding,
        "baseline_revision": baseline,
        "worktree": str(worktree),
        "allowed_paths": attempt["allowed_paths"],
        "passed": True,
        "attempt": attempt["attempt"],
        "attempt_history": attempt["attempt_history"],
        "hypothesis": attempt["hypothesis"],
        "modified_paths": modified_paths,
        "patch_path": "candidate.patch",
        "patch_sha256": patch_digest,
        "workspace_state": "CANDIDATE_RECORDED",
    }
    for field, expected in record_checks.items():
        if record.get(field) != expected:
            raise ToolError(f"candidate record {field} has drifted")
    if changes:
        if changes != set(modified_paths):
            raise ToolError(
                "workspace paths do not match recorded candidate.patch"
            )
        current_patch = build_candidate_patch(
            worktree,
            baseline,
            changes,
        )
        if current_patch != patch:
            raise ToolError(
                "workspace does not match recorded candidate.patch"
            )
    return patch, modified_paths, record


def run_finish_attempt(
    spec_path: Path,
    run_dir: Path,
    worktree: Path,
    replay_result_path: Path,
    allow_synthetic_replay: bool,
) -> None:
    binding, contract, baseline, max_attempts = contract_context(spec_path)
    worktree = validate_worktree(worktree, baseline)
    if run_dir.is_symlink():
        raise ToolError("attempt run directory must not be a symlink")
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise ToolError("attempt run directory does not exist")
    if (run_dir / "result.json").exists():
        raise ToolError("attempt Run is already sealed")
    attempt = load_attempt(
        run_dir,
        binding=binding,
        baseline=baseline,
        max_attempts=max_attempts,
        worktree=worktree,
    )
    expected_replay = (run_dir / "replay" / "result.json").resolve()
    if replay_result_path.resolve() != expected_replay:
        raise ToolError(
            "attempt replay result must be <run-dir>/replay/result.json"
        )
    (
        replay_result,
        replay_digest,
        replay_evidence_digest,
    ) = validate_replay_result(
        expected_replay,
        binding,
        contract,
        allow_synthetic_replay=allow_synthetic_replay,
    )
    changes = require_known_changes(
        worktree,
        attempt["allowed_paths"],
    )
    patch, modified_paths, _candidate_record = load_recorded_candidate(
        run_dir,
        binding=binding,
        baseline=baseline,
        worktree=worktree,
        attempt=attempt,
        changes=changes,
    )
    candidate_record_path = run_dir / "candidate-result.json"
    try:
        replay_mtime = expected_replay.stat().st_mtime_ns
        candidate_mtime = candidate_record_path.stat().st_mtime_ns
    except OSError as error:
        raise ToolError(
            f"cannot inspect candidate/replay order: {error}"
        ) from error
    if replay_mtime < candidate_mtime:
        raise ToolError(
            "replay result predates the recorded candidate.patch"
        )

    patch_digest = hashlib.sha256(patch).hexdigest()
    candidate_record_digest = file_sha256(candidate_record_path)
    recovery_action = (
        "RETAIN_PATCH"
        if replay_result["passed"]
        else "RESTORE_BASELINE"
    )
    outcome = {
        "schema": OUTCOME_SCHEMA,
        **common_result(
            action="attempt-outcome",
            binding=binding,
            baseline=baseline,
            worktree=worktree,
            allowed_paths=attempt["allowed_paths"],
        ),
        "passed": replay_result["passed"],
        "attempt": attempt["attempt"],
        "attempt_history": attempt["attempt_history"],
        "hypothesis": attempt["hypothesis"],
        "modified_paths": modified_paths,
        "patch_sha256": patch_digest,
        "candidate_result_sha256": candidate_record_digest,
        "replay_result_sha256": replay_digest,
        "replay_evidence_sha256": replay_evidence_digest,
        "recovery_action": recovery_action,
    }
    outcome_path = run_dir / "outcome.json"
    if outcome_path.exists():
        if read_json_object(outcome_path, "attempt outcome") != outcome:
            raise ToolError("attempt outcome has drifted")
    else:
        if not changes:
            raise ToolError(
                "candidate patch disappeared before outcome was sealed"
            )
        atomic_write_json(outcome_path, outcome)
        with (run_dir / "workspace.log").open(
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(
                "action=seal-outcome\n"
                f"replay_passed={str(replay_result['passed']).lower()}\n"
                f"patch_sha256={patch_digest}\n"
                f"recovery_action={recovery_action}\n"
            )

    if replay_result["passed"]:
        if not changes:
            raise ToolError(
                "passing attempt cannot finish without its candidate patch "
                "in the worktree"
            )
        if changed_paths(worktree) != set(modified_paths):
            raise ToolError(
                "passing patch is not the only workspace change"
            )
        workspace_state = "PATCH_RETAINED"
    else:
        if changes:
            restore_candidate(
                worktree,
                baseline,
                modified_paths,
            )
        remaining = changed_paths(worktree)
        if remaining:
            raise ToolError(
                "failed attempt could not restore the fixed baseline: "
                + ", ".join(sorted(remaining))
            )
        workspace_state = "BASELINE_RESTORED"

    result = {
        **common_result(
            action="finish-attempt",
            binding=binding,
            baseline=baseline,
            worktree=worktree,
            allowed_paths=attempt["allowed_paths"],
        ),
        "passed": replay_result["passed"],
        "attempt": attempt["attempt"],
        "attempt_history": attempt["attempt_history"],
        "max_repair_attempts": max_attempts,
        "attempt_limit_reached": (
            not replay_result["passed"]
            and attempt["attempt"] == max_attempts
        ),
        "hypothesis": attempt["hypothesis"],
        "modified_paths": modified_paths,
        "patch_path": "candidate.patch",
        "patch_sha256": patch_digest,
        "candidate_result_sha256": candidate_record_digest,
        "replay_result": "replay/result.json",
        "replay_result_sha256": replay_digest,
        "replay_evidence_sha256": replay_evidence_digest,
        "workspace_state": workspace_state,
        "evidence": [
            "attempt.json",
            "candidate.json",
            "candidate.patch",
            "candidate-result.json",
            "outcome.json",
            "replay/result.json",
            "workspace.log",
        ],
    }
    with (run_dir / "workspace.log").open(
        "a",
        encoding="utf-8",
    ) as stream:
        stream.write(
            "action=finish-attempt\n"
            f"replay_passed={str(replay_result['passed']).lower()}\n"
            f"patch_sha256={patch_digest}\n"
            f"workspace_state={workspace_state}\n"
        )
    atomic_write_json(run_dir / "result.json", result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "check-baseline",
            "assess-baseline",
            "start-attempt",
            "record-candidate",
            "finish-attempt",
        ),
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--allowed-path", action="append", default=[])
    parser.add_argument("--replay-result", type=Path)
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--hypothesis")
    parser.add_argument("--previous-result", type=Path)
    parser.add_argument(
        "--allow-synthetic-replay",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "check-baseline":
            if any(
                value is not None
                for value in (
                    args.replay_result,
                    args.attempt,
                    args.hypothesis,
                    args.previous_result,
                )
            ) or args.allow_synthetic_replay:
                raise ToolError(
                    "replay, attempt, hypothesis, and previous result "
                    "arguments are not "
                    "valid for check-baseline"
                )
            run_check_baseline(
                args.spec,
                args.run_dir,
                args.worktree,
                args.allowed_path,
            )
        elif args.mode == "assess-baseline":
            if args.replay_result is None:
                raise ToolError(
                    "--replay-result is required for assess-baseline"
                )
            if (
                args.attempt is not None
                or args.hypothesis is not None
                or args.previous_result is not None
            ):
                raise ToolError(
                    "attempt, hypothesis, and previous result are not valid for "
                    "assess-baseline"
                )
            run_assess_baseline(
                args.spec,
                args.run_dir,
                args.worktree,
                args.allowed_path,
                args.replay_result,
                args.allow_synthetic_replay,
            )
        elif args.mode == "start-attempt":
            if (
                args.attempt is None
                or args.hypothesis is None
                or args.previous_result is None
            ):
                raise ToolError(
                    "--attempt, --hypothesis, and --previous-result are "
                    "required for "
                    "start-attempt"
                )
            if (
                args.replay_result is not None
                or args.allow_synthetic_replay
            ):
                raise ToolError(
                    "replay arguments are not valid for start-attempt"
                )
            run_start_attempt(
                args.spec,
                args.run_dir,
                args.worktree,
                args.allowed_path,
                args.attempt,
                args.hypothesis,
                args.previous_result,
            )
        elif args.mode == "record-candidate":
            if (
                args.allowed_path
                or args.replay_result is not None
                or args.attempt is not None
                or args.hypothesis is not None
                or args.previous_result is not None
                or args.allow_synthetic_replay
            ):
                raise ToolError(
                    "allowed path, replay, attempt, hypothesis, and "
                    "previous result arguments are not valid for "
                    "record-candidate"
                )
            run_record_candidate(
                args.spec,
                args.run_dir,
                args.worktree,
            )
        else:
            if args.replay_result is None:
                raise ToolError(
                    "--replay-result is required for finish-attempt"
                )
            if (
                args.allowed_path
                or args.attempt is not None
                or args.hypothesis is not None
                or args.previous_result is not None
            ):
                raise ToolError(
                    "allowed path, attempt, hypothesis, and previous result "
                    "arguments are "
                    "not valid for finish-attempt"
                )
            run_finish_attempt(
                args.spec,
                args.run_dir,
                args.worktree,
                args.replay_result,
                args.allow_synthetic_replay,
            )
    except (SpecContractError, ToolError, OSError) as error:
        print(f"workspace_guard.py: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
