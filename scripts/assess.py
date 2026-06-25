#!/usr/bin/env python3
"""Static repository and non-destructive environment assessment helpers."""
from __future__ import annotations

import argparse
import glob
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "build", "dist", "node_modules"}
TEXT_SUFFIXES = {".py", ".toml", ".yaml", ".yml", ".json", ".md", ".rst", ".txt", ".sh", ".c", ".cc", ".cpp", ".h", ".hpp"}
PATTERNS = {
    "vllm-kunlun": {
        "plugin": [r"platform_plugins", r"entry.?points?", r"Platform"],
        "registry": [r"ModelRegistry", r"register_model", r"architectures"],
        "weights": [r"load_weights", r"weight_loader", r"packed_modules"],
        "attention": [r"AttentionBackend", r"kv_cache", r"paged_attention"],
        "moe": [r"fused_moe", r"expert_parallel"],
        "quantization": [r"quantization", r"awq", r"gptq"],
        "distributed": [r"tensor_parallel", r"pipeline_parallel", r"xccl"],
    },
    "sglang-kunlun": {
        "plugin": [r"entry.?points?", r"register_platform", r"plugin"],
        "registry": [r"register_model", r"architectures", r"model_registry"],
        "weights": [r"load_weights", r"weight_loader", r"packed_modules"],
        "attention": [r"RadixAttention", r"AttentionBackend", r"kv_cache"],
        "moe": [r"fused_moe", r"expert_parallel"],
        "quantization": [r"quantization", r"awq", r"gptq"],
        "distributed": [r"tp_size", r"dp_size", r"ep_size", r"xccl"],
    },
}
MODULES = {"vllm-kunlun": ["torch", "vllm", "vllm_kunlun"], "sglang-kunlun": ["torch", "sglang"]}
PACKAGES = ("torch", "transformers", "vllm", "vllm-kunlun", "sglang")
ENV_PREFIXES = ("KUNLUN_", "XPU_", "XCCL_", "VLLM_", "SGLANG_", "CUDA_VISIBLE_DEVICES")


def run_command(argv: list[str], timeout: int = 10) -> dict[str, Any]:
    try:
        proc = subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False)
        return {"argv": argv, "exit_code": proc.returncode, "stdout": proc.stdout[-8000:], "stderr": proc.stderr[-8000:]}
    except Exception as exc:  # noqa: BLE001
        return {"argv": argv, "error": str(exc)}


def git(repo: Path, *args: str) -> dict[str, Any]:
    return run_command(["git", "-C", str(repo), *args], 15)


def iter_files(repo: Path) -> Iterable[Path]:
    for base, dirs, names in os.walk(repo):
        dirs[:] = [name for name in sorted(dirs) if name not in SKIP_DIRS]
        for name in sorted(names):
            path = Path(base) / name
            try:
                if path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size <= 2 * 1024 * 1024:
                    yield path
            except OSError:
                continue


def scan_repo(repo: Path, engine: str, terms: list[str]) -> dict[str, Any]:
    groups = {**PATTERNS[engine], **{f"term:{term}": [re.escape(term)] for term in terms if term}}
    compiled = {name: [re.compile(value, re.I) for value in values] for name, values in groups.items()}
    matches: dict[str, list[dict[str, Any]]] = {name: [] for name in groups}
    for path in iter_files(repo):
        rel = str(path.relative_to(repo))
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), 1):
            for group, regexes in compiled.items():
                if len(matches[group]) >= 100:
                    continue
                if any(regex.search(line) or regex.search(rel) for regex in regexes):
                    matches[group].append({"path": rel, "line": line_no, "text": line.strip()[:400]})
    return {
        "path": str(repo.resolve()),
        "git": {"head": git(repo, "rev-parse", "HEAD"), "branch": git(repo, "branch", "--show-current"), "status": git(repo, "status", "--short")},
        "matches": matches,
    }


def repo_command(args: argparse.Namespace) -> int:
    target = Path(args.target_repo)
    if not target.is_dir():
        raise SystemExit(f"invalid target repo: {target}")
    terms = list(dict.fromkeys(args.architecture + args.model_type))
    data: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "engine": args.engine,
        "terms": terms,
        "target": scan_repo(target, args.engine, terms),
        "upstream": None,
    }
    if args.upstream_repo:
        upstream = Path(args.upstream_repo)
        data["upstream"] = scan_repo(upstream, args.engine, terms) if upstream.is_dir() else {"path": str(upstream), "exists": False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def path_fact(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    path = Path(value).expanduser()
    result: dict[str, Any] = {"path": str(path), "exists": path.exists(), "is_dir": path.is_dir(), "readable": os.access(path, os.R_OK)}
    try:
        usage = shutil.disk_usage(path if path.exists() else path.parent)
        result["filesystem"] = {"total": usage.total, "free": usage.free}
    except OSError:
        pass
    return result


def import_probe(python: str, module: str, timeout: int) -> dict[str, Any]:
    code = """import importlib.util,json,sys\nname=sys.argv[1]\nspec=importlib.util.find_spec(name)\nout={'module':name,'origin':getattr(spec,'origin',None)}\ntry:\n m=__import__(name);out.update(ok=True,file=getattr(m,'__file__',None),version=getattr(m,'__version__',None))\nexcept Exception as e:out.update(ok=False,error_type=type(e).__name__,error=str(e))\nprint(json.dumps(out,ensure_ascii=False))"""
    return run_command([python, "-c", code, module], timeout)


def torch_probe(python: str, timeout: int) -> dict[str, Any]:
    code = """import json\nout={}\ntry:\n import torch;out.update(ok=True,version=getattr(torch,'__version__',None),file=getattr(torch,'__file__',None))\n for n in ('cuda','xpu'):\n  api=getattr(torch,n,None);x={'present':api is not None}\n  if api is not None:\n   for f in ('is_available','device_count'):\n    fn=getattr(api,f,None)\n    if callable(fn):\n     try:x[f]=fn()\n     except Exception as e:x[f+'_error']=str(e)\n  out[n]=x\nexcept Exception as e:out.update(ok=False,error_type=type(e).__name__,error=str(e))\nprint(json.dumps(out,ensure_ascii=False))"""
    return run_command([python, "-c", code], timeout)


def env_command(args: argparse.Namespace) -> int:
    python = shutil.which(args.python) or args.python
    packages: dict[str, Any] = {}
    for name in PACKAGES:
        try:
            dist = importlib.metadata.distribution(name)
            packages[name] = {"version": dist.version, "location": str(dist.locate_file(""))}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
        except Exception as exc:  # noqa: BLE001
            packages[name] = {"error": str(exc)}
    imports = {name: import_probe(python, name, args.timeout) for name in MODULES[args.engine]}
    torch = torch_probe(python, args.timeout)
    parsed: dict[str, Any] | None = None
    try:
        parsed = json.loads(torch.get("stdout", "").splitlines()[-1])
    except Exception:  # noqa: BLE001
        pass
    counts: list[int] = []
    if isinstance(parsed, dict):
        for api_name in ("cuda", "xpu"):
            api = parsed.get(api_name)
            count = api.get("device_count") if isinstance(api, dict) else None
            if isinstance(count, int):
                counts.append(count)
    visible = max(counts) if counts else None
    issues: list[dict[str, Any]] = []
    for name, probe in imports.items():
        try:
            result = json.loads(probe.get("stdout", "").splitlines()[-1])
        except Exception:  # noqa: BLE001
            result = {}
        if not result.get("ok"):
            issues.append({"severity": "error", "code": f"import_failed:{name}"})
    if visible is None:
        issues.append({"severity": "warning", "code": "device_count_unknown"})
    elif visible < args.required_devices:
        issues.append({"severity": "error", "code": "insufficient_devices", "visible": visible, "required": args.required_devices})
    readiness = "unavailable" if any(item["severity"] == "error" for item in issues) else ("degraded" if issues else "ready")
    data = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "readiness": readiness,
        "issues": issues,
        "host": {"platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count()},
        "python": {"requested": args.python, "executable": python, "version": run_command([python, "--version"])},
        "engine": args.engine,
        "required_devices": args.required_devices,
        "visible_devices": visible,
        "paths": {"model": path_fact(args.model_path), "target_repo": path_fact(args.target_repo), "upstream_repo": path_fact(args.upstream_repo), "dev_shm": path_fact("/dev/shm")},
        "packages": packages,
        "imports": imports,
        "torch_probe": torch,
        "device_nodes": sorted({item for pattern in ("/dev/xpu*", "/dev/kunlun*", "/dev/kfd", "/dev/dri/render*") for item in glob.glob(pattern)}),
        "tools": {name: shutil.which(name) for name in ("xpu-smi", "klx-smi", "cnmon", "nvidia-smi")},
        "environment": {key: value for key, value in os.environ.items() if any(key == prefix or key.startswith(prefix) for prefix in ENV_PREFIXES)},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"readiness={readiness}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assessment helpers")
    sub = parser.add_subparsers(dest="command_name", required=True)
    repo = sub.add_parser("repo")
    repo.add_argument("--engine", choices=sorted(PATTERNS), required=True)
    repo.add_argument("--target-repo", required=True)
    repo.add_argument("--upstream-repo")
    repo.add_argument("--architecture", action="append", default=[])
    repo.add_argument("--model-type", action="append", default=[])
    repo.add_argument("--output", required=True)
    repo.set_defaults(handler=repo_command)
    env = sub.add_parser("env")
    env.add_argument("--engine", choices=sorted(MODULES), required=True)
    env.add_argument("--python", default=sys.executable)
    env.add_argument("--model-path")
    env.add_argument("--target-repo")
    env.add_argument("--upstream-repo")
    env.add_argument("--required-devices", type=int, default=1)
    env.add_argument("--timeout", type=int, default=20)
    env.add_argument("--output", required=True)
    env.set_defaults(handler=env_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
