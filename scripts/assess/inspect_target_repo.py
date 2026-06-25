#!/usr/bin/env python3
"""Collect static evidence from an inference-engine or hardware-plugin repository.

The script never imports target repository code and never modifies the repository.
It intentionally produces evidence, not a support verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

TEXT_SUFFIXES = {
    ".py", ".pyi", ".toml", ".yaml", ".yml", ".json", ".md", ".rst",
    ".txt", ".sh", ".bash", ".c", ".cc", ".cpp", ".h", ".hpp",
}
SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "build",
    "dist", ".tox", ".mypy_cache", ".pytest_cache", "node_modules",
}
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_MATCHES_PER_PATTERN = 200

ENGINE_PATTERNS: dict[str, dict[str, list[str]]] = {
    "vllm-kunlun": {
        "plugin_registration": [
            r"vllm\.platform_plugins", r"vllm\.general_plugins",
            r"entry[_-]?points?", r"register_platform", r"Platform",
        ],
        "model_registry": [
            r"ModelRegistry", r"register_model", r"_MODELS", r"architectures",
        ],
        "weight_loading": [
            r"load_weights", r"weight_loader", r"packed_modules_mapping",
            r"AutoWeightsLoader",
        ],
        "attention_kv": [
            r"AttentionBackend", r"attention_backend", r"KVCache", r"kv_cache",
            r"paged_attention", r"unified_attention",
        ],
        "moe": [r"FusedMoE", r"fused_moe", r"expert_parallel", r"moe"],
        "quantization": [r"quantization", r"compressed_tensors", r"awq", r"gptq", r"w8a8"],
        "custom_ops": [r"custom_op", r"register_fake", r"torch\.library", r"ops\."],
        "distributed": [r"tensor_parallel", r"pipeline_parallel", r"expert_parallel", r"xccl"],
        "patches": [r"patch", r"eval_frame", r"monkey"],
        "tests": [r"pytest", r"unittest", r"model_test", r"tests?/"],
    },
    "sglang-kunlun": {
        "plugin_registration": [
            r"entry[_-]?points?", r"register_platform", r"platform_plugin",
            r"plugin", r"Platform",
        ],
        "model_registry": [
            r"ModelRegistry", r"register_model", r"architectures", r"model_registry",
        ],
        "weight_loading": [
            r"load_weights", r"weight_loader", r"default_weight_loader",
            r"packed_modules_mapping",
        ],
        "attention_kv": [
            r"RadixAttention", r"AttentionBackend", r"attention_backend",
            r"KVCache", r"kv_cache", r"token_to_kv_pool",
        ],
        "moe": [r"FusedMoE", r"fused_moe", r"expert_parallel", r"moe_a2a", r"moe"],
        "quantization": [r"quantization", r"compressed_tensors", r"awq", r"gptq", r"w8a8"],
        "custom_ops": [r"custom_op", r"torch\.library", r"sgl_kernel", r"ops\."],
        "distributed": [r"tp_size", r"tensor_parallel", r"dp_size", r"ep_size", r"xccl"],
        "server_runtime": [r"ServerArgs", r"launch_server", r"ModelRunner", r"TpModelWorker"],
        "tests": [r"pytest", r"unittest", r"model_test", r"tests?/"],
    },
}


def run_git(repo: Path, *args: str) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return {
            "argv": ["git", "-C", str(repo), *args],
            "exit_code": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": ["git", "-C", str(repo), *args], "error": str(exc)}


def iter_text_files(repo: Path) -> Iterable[Path]:
    for root, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            path = Path(root) / name
            if path.suffix.lower() not in TEXT_SUFFIXES and name not in {
                "Dockerfile", "Makefile", "requirements.txt", "setup.py",
            }:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield path


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def search_patterns(repo: Path, groups: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    compiled = {
        group: [(pattern, re.compile(pattern, re.IGNORECASE)) for pattern in patterns]
        for group, patterns in groups.items()
    }
    results: dict[str, list[dict[str, Any]]] = {group: [] for group in groups}

    for path in iter_text_files(repo):
        text = read_text(path)
        if text is None:
            continue
        rel = str(path.relative_to(repo))
        lines = text.splitlines()
        for group, patterns in compiled.items():
            if len(results[group]) >= MAX_MATCHES_PER_PATTERN:
                continue
            path_patterns = [raw for raw, regex in patterns if regex.search(rel)]
            if path_patterns and len(results[group]) < MAX_MATCHES_PER_PATTERN:
                results[group].append(
                    {
                        "path": rel,
                        "line": None,
                        "patterns": path_patterns,
                        "text": "<path match>",
                    }
                )
            for line_no, line in enumerate(lines, 1):
                matched_patterns = [raw for raw, regex in patterns if regex.search(line)]
                if not matched_patterns:
                    continue
                results[group].append(
                    {
                        "path": rel,
                        "line": line_no,
                        "patterns": matched_patterns,
                        "text": line.strip()[:500],
                    }
                )
                if len(results[group]) >= MAX_MATCHES_PER_PATTERN:
                    break
    return results


def find_terms(repo: Path, terms: list[str]) -> dict[str, list[dict[str, Any]]]:
    groups = {f"term:{term}": [re.escape(term)] for term in terms if term}
    if not groups:
        return {}
    return search_patterns(repo, groups)


def package_metadata(repo: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    pyproject = repo / "pyproject.toml"
    if pyproject.is_file():
        try:
            import tomllib

            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            metadata["pyproject"] = {
                "project": data.get("project", {}),
                "build_system": data.get("build-system", {}),
            }
        except Exception as exc:  # evidence collector: preserve parse error
            metadata["pyproject_error"] = str(exc)
    for filename in ("setup.py", "setup.cfg", "requirements.txt"):
        path = repo / filename
        if path.is_file():
            metadata[filename] = {
                "path": filename,
                "preview": path.read_text(encoding="utf-8", errors="replace")[:12000],
            }
    return metadata


def inspect_repo(repo: Path, engine: str, terms: list[str]) -> dict[str, Any]:
    return {
        "path": str(repo.resolve()),
        "exists": repo.is_dir(),
        "git": {
            "commit": run_git(repo, "rev-parse", "HEAD"),
            "branch": run_git(repo, "branch", "--show-current"),
            "status": run_git(repo, "status", "--short"),
            "remote": run_git(repo, "remote", "-v"),
        },
        "package_metadata": package_metadata(repo),
        "engine_markers": search_patterns(repo, ENGINE_PATTERNS[engine]),
        "model_terms": find_terms(repo, terms),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=sorted(ENGINE_PATTERNS), required=True)
    parser.add_argument("--target-repo", required=True)
    parser.add_argument("--upstream-repo")
    parser.add_argument("--architecture", action="append", default=[])
    parser.add_argument("--model-type", action="append", default=[])
    parser.add_argument("--model-family", action="append", default=[])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_repo = Path(args.target_repo).expanduser()
    if not target_repo.is_dir():
        print(f"target repo not found: {target_repo}", file=sys.stderr)
        return 2

    terms = list(dict.fromkeys(args.architecture + args.model_type + args.model_family))
    data: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "engine": args.engine,
        "search_terms": terms,
        "target": inspect_repo(target_repo, args.engine, terms),
        "upstream": None,
    }

    if args.upstream_repo:
        upstream_repo = Path(args.upstream_repo).expanduser()
        if upstream_repo.is_dir():
            data["upstream"] = inspect_repo(upstream_repo, args.engine, terms)
        else:
            data["upstream"] = {
                "path": str(upstream_repo),
                "exists": False,
                "error": "upstream repo not found",
            }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
