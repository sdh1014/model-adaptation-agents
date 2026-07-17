#!/usr/bin/env python3
"""Build or verify a human-copied Handoff Bundle."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
from typing import Any, Dict

from _lib.spec_contract import SpecContractError, load_spec_binding
from model_adaptation_capture.contracts import KERNEL_CALL_STATE_SCHEMA


MANIFEST_SCHEMA = "handoff-manifest/v1"


class ToolError(RuntimeError):
    """The handoff action could not form trustworthy evidence."""


def read_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ToolError(f"cannot read {label}: {error}") from error
    except json.JSONDecodeError as error:
        raise ToolError(f"{label} is not valid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ToolError(f"{label} must be one JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ToolError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def create_run_dir(run_dir: Path) -> None:
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise ToolError(f"run directory must be fresh and creatable: {error}") from error


def regular_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ToolError(f"directory does not exist: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ToolError(f"symbolic links are forbidden: {path}")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise ToolError(f"unsupported filesystem entry: {path}")
    return files


def validate_golden_run(golden_run: Path, binding: Dict[str, Any]) -> None:
    state = read_json_object(
        golden_run / "capture-state.json",
        "Golden capture state",
    )
    if state.get("schema") != KERNEL_CALL_STATE_SCHEMA:
        raise ToolError("Golden capture state schema has drifted")
    if state.get("spec_binding") != binding:
        raise ToolError("Golden capture state spec_binding does not match Contract Data")
    if state.get("status") != "SEALED" or state.get("capture_closed") is not True:
        raise ToolError("Golden Run must be SEALED and capture_closed")
    self_replay = state.get("self_replay")
    if not isinstance(self_replay, dict) or self_replay.get("passed") is not True:
        raise ToolError("Golden Run self-replay has not passed")
    samples = state.get("samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 3:
        raise ToolError("Golden Run must contain one to three samples")
    if state.get("saved_shape_count") != len(samples):
        raise ToolError("Golden Run saved_shape_count has drifted")
    if self_replay.get("checked_shape_count") != len(samples):
        raise ToolError("Golden Run self-replay did not check every sample")
    if not is_sha256(self_replay.get("worker_result_sha256")):
        raise ToolError("Golden Run self_replay worker_result_sha256 is invalid")
    for sample in samples:
        if not isinstance(sample, dict):
            raise ToolError("Golden sample entry must be one object")
        shape_id = sample.get("shape_id")
        expected_file = f"samples/{shape_id}.pt"
        if not isinstance(shape_id, str) or not shape_id:
            raise ToolError("Golden sample shape_id is invalid")
        if sample.get("file") != expected_file:
            raise ToolError("Golden sample path has drifted")
        sample_path = (golden_run / expected_file).resolve()
        if (
            not sample_path.is_relative_to(golden_run.resolve())
            or not sample_path.is_file()
        ):
            raise ToolError("Golden sample is missing or escapes its Run")
    regular_files(golden_run)


def manifest_entries(bundle_dir: Path) -> list[Dict[str, Any]]:
    entries = []
    for path in regular_files(bundle_dir):
        relative = path.relative_to(bundle_dir).as_posix()
        if relative == "manifest.json":
            continue
        entries.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return entries


def copy_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for source_path in regular_files(source):
        relative = source_path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)


def run_build(
    spec_path: Path,
    run_dir: Path,
    golden_run: Path,
    bundle_spec: Path,
) -> None:
    binding = load_spec_binding(spec_path).as_result_dict()
    bundle_binding = load_spec_binding(bundle_spec).as_result_dict()
    if bundle_binding != binding:
        raise ToolError("bundle Spec Contract Data does not match current Contract Data")
    if bundle_spec.is_symlink() or not bundle_spec.is_file():
        raise ToolError("bundle Spec must be one regular file")
    golden_run = golden_run.resolve()
    validate_golden_run(golden_run, binding)

    create_run_dir(run_dir)
    bundle_dir = run_dir / "bundle"
    bundle_dir.mkdir()
    shutil.copyfile(bundle_spec, bundle_dir / "migration-spec.md")
    bundled_golden = bundle_dir / "runs" / golden_run.name
    bundled_golden.parent.mkdir()
    copy_tree(golden_run, bundled_golden)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "spec_binding": binding,
        "golden_run": f"runs/{golden_run.name}",
        "files": manifest_entries(bundle_dir),
    }
    manifest_path = bundle_dir / "manifest.json"
    write_json(manifest_path, manifest)
    manifest_digest = file_sha256(manifest_path)
    result = {
        "tool": "handoff_bundle.py",
        "action": "build",
        "spec_binding": binding,
        "passed": True,
        "bundle_path": str(bundle_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_digest,
        "file_count": len(manifest["files"]),
        "golden_run": manifest["golden_run"],
        "evidence": ["bundle/manifest.json", "handoff.log"],
        "summary": "Handoff Bundle and integrity manifest were built.",
    }
    write_json(run_dir / "result.json", result)
    (run_dir / "handoff.log").write_text(
        "\n".join(
            [
                "action=build",
                f"passed=true",
                f"golden_run={manifest['golden_run']}",
                f"file_count={len(manifest['files'])}",
                f"manifest_sha256={manifest_digest}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def validate_manifest_entry(entry: Any) -> str:
    if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
        raise ToolError("manifest file entry must contain path, size, and sha256")
    relative = entry["path"]
    if not isinstance(relative, str) or not relative:
        raise ToolError("manifest file path must be a non-empty string")
    pure_path = PurePosixPath(relative)
    if pure_path.is_absolute() or ".." in pure_path.parts or str(pure_path) != relative:
        raise ToolError(f"manifest path is unsafe: {relative!r}")
    size = entry["size"]
    digest = entry["sha256"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ToolError(f"manifest size is invalid for {relative}")
    if not is_sha256(digest):
        raise ToolError(f"manifest sha256 is invalid for {relative}")
    return relative


def validate_bundle(bundle_dir: Path, binding: Dict[str, Any]) -> tuple[int, str]:
    manifest_path = bundle_dir / "manifest.json"
    manifest = read_json_object(manifest_path, "Handoff manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ToolError("Handoff manifest schema has drifted")
    if manifest.get("spec_binding") != binding:
        raise ToolError("Handoff manifest spec_binding does not match Contract Data")
    bundled_spec = bundle_dir / "migration-spec.md"
    if load_spec_binding(bundled_spec).as_result_dict() != binding:
        raise ToolError("bundled Migration Spec does not match Contract Data")
    golden_relative = manifest.get("golden_run")
    if (
        not isinstance(golden_relative, str)
        or not golden_relative.startswith("runs/")
        or len(PurePosixPath(golden_relative).parts) != 2
    ):
        raise ToolError("Handoff manifest golden_run is invalid")

    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ToolError("Handoff manifest files must be a non-empty list")
    paths = [validate_manifest_entry(entry) for entry in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ToolError("Handoff manifest file paths must be unique and sorted")
    actual_entries = manifest_entries(bundle_dir)
    actual_paths = [entry["path"] for entry in actual_entries]
    missing = sorted(set(paths) - set(actual_paths))
    if missing:
        raise ToolError("Handoff Bundle missing file: " + ", ".join(missing))
    unexpected = sorted(set(actual_paths) - set(paths))
    if unexpected:
        raise ToolError("Handoff Bundle unexpected file: " + ", ".join(unexpected))
    if actual_paths != paths:
        raise ToolError("Handoff Bundle file order does not match manifest")
    for expected, actual in zip(entries, actual_entries):
        if expected["size"] != actual["size"]:
            raise ToolError(f"Handoff Bundle file size differs: {expected['path']}")
        if expected["sha256"] != actual["sha256"]:
            raise ToolError(f"Handoff Bundle file sha256 differs: {expected['path']}")
    validate_golden_run(bundle_dir / golden_relative, binding)
    return len(entries), file_sha256(manifest_path)


def run_verify(spec_path: Path, run_dir: Path, bundle_dir: Path) -> None:
    binding = load_spec_binding(spec_path).as_result_dict()
    bundle_dir = bundle_dir.resolve()
    try:
        file_count, manifest_digest = validate_bundle(bundle_dir, binding)
    except (SpecContractError, ToolError) as error:
        create_run_dir(run_dir)
        result = {
            "tool": "handoff_bundle.py",
            "action": "verify",
            "spec_binding": binding,
            "passed": False,
            "bundle_path": str(bundle_dir),
            "manifest_path": str((bundle_dir / "manifest.json").resolve()),
            "manifest_verified": False,
            "error": str(error),
            "evidence": ["handoff.log"],
            "summary": "Handoff Bundle integrity verification failed.",
        }
        write_json(run_dir / "result.json", result)
        (run_dir / "handoff.log").write_text(
            "\n".join(
                [
                    "action=verify",
                    "passed=false",
                    f"error={error}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return
    create_run_dir(run_dir)
    result = {
        "tool": "handoff_bundle.py",
        "action": "verify",
        "spec_binding": binding,
        "passed": True,
        "bundle_path": str(bundle_dir.resolve()),
        "manifest_path": str((bundle_dir / "manifest.json").resolve()),
        "manifest_sha256": manifest_digest,
        "manifest_verified": True,
        "file_count": file_count,
        "evidence": ["handoff.log"],
        "summary": "Handoff Bundle matches its complete integrity manifest.",
    }
    write_json(run_dir / "result.json", result)
    (run_dir / "handoff.log").write_text(
        "\n".join(
            [
                "action=verify",
                "passed=true",
                f"file_count={file_count}",
                f"manifest_sha256={manifest_digest}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("build", "verify"))
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--golden-run", type=Path)
    parser.add_argument("--bundle-spec", type=Path)
    parser.add_argument("--bundle-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "build":
            if args.golden_run is None or args.bundle_spec is None:
                raise ToolError("--golden-run and --bundle-spec are required for build")
            if args.bundle_dir is not None:
                raise ToolError("--bundle-dir is not valid for build")
            run_build(args.spec, args.run_dir, args.golden_run, args.bundle_spec)
        else:
            if args.bundle_dir is None:
                raise ToolError("--bundle-dir is required for verify")
            if args.golden_run is not None or args.bundle_spec is not None:
                raise ToolError("--golden-run and --bundle-spec are not valid for verify")
            run_verify(args.spec, args.run_dir, args.bundle_dir)
    except (SpecContractError, ToolError, OSError) as error:
        print(f"handoff_bundle.py: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
