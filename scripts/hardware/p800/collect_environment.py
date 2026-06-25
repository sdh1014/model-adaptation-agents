#!/usr/bin/env python3
"""Collect a minimal, non-destructive P800 assessment environment snapshot."""
from __future__ import annotations

import argparse
import glob
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PACKAGE_NAMES = [
    "torch",
    "transformers",
    "safetensors",
    "vllm",
    "vllm-kunlun",
    "sglang",
    "torch-plugin",
    "torch-xray",
    "xtorch-ops",
    "kunlun-ops",
    "xspeedgate-ops",
    "cocopod",
]
IMPORTS_BY_ENGINE = {
    "vllm-kunlun": ["torch", "vllm", "vllm_kunlun"],
    "sglang-kunlun": ["torch", "sglang"],
}
ENV_KEYS = {
    "CONDA_PREFIX",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "XPU_VISIBLE_DEVICES",
    "KUNLUN_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "VLLM_TARGET_DEVICE",
    "VLLM_PLUGINS",
    "SGLANG_USE_AITER",
}
ENV_PREFIXES = ("KUNLUN_", "XPU_", "XMLIR_", "XCCL_", "VLLM_", "SGLANG_")
TOOL_COMMANDS = {
    "xpu-smi": [["xpu-smi", "--version"], ["xpu-smi"]],
    "klx-smi": [["klx-smi", "--version"], ["klx-smi"]],
    "cnmon": [["cnmon", "--version"], ["cnmon"]],
    "nvidia-smi": [["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]],
}
DEVICE_GLOBS = [
    "/dev/xpu*",
    "/dev/kunlun*",
    "/dev/kfd",
    "/dev/dri/render*",
    "/dev/nvidia*",
]


def run_command(argv: list[str], timeout: int = 10) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "argv": argv,
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-12000:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "error": str(exc)}


def stat_path(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    path = Path(value).expanduser()
    result: dict[str, Any] = {
        "path": str(path),
        "resolved": str(path.resolve(strict=False)),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "readable": os.access(path, os.R_OK),
    }
    probe = path if path.exists() else path.parent
    try:
        usage = shutil.disk_usage(probe)
        result["filesystem"] = {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }
    except OSError as exc:
        result["filesystem_error"] = str(exc)
    return result


def meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            value = raw.strip().split()[0]
            result[key] = int(value) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return result


def python_runtime_info(python: str, timeout: int) -> dict[str, Any]:
    code = r'''
import json, platform, sys
print(json.dumps({
    "executable": sys.executable,
    "version": sys.version,
    "version_info": list(sys.version_info[:3]),
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "platform": platform.platform(),
}, ensure_ascii=False))
'''
    try:
        proc = subprocess.run(
            [python, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        parsed: Any = None
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if lines:
            try:
                parsed = json.loads(lines[-1])
            except json.JSONDecodeError:
                parsed = None
        return {
            "exit_code": proc.returncode,
            "result": parsed,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def package_info(python: str, names: list[str], timeout: int) -> dict[str, Any]:
    code = r'''
import importlib.metadata, json, os
names = json.loads(os.environ["ASSESS_PACKAGE_NAMES"])
out = {}
for name in names:
    try:
        dist = importlib.metadata.distribution(name)
        out[name] = {
            "version": dist.version,
            "location": str(dist.locate_file("")),
        }
    except importlib.metadata.PackageNotFoundError:
        out[name] = None
    except Exception as exc:
        out[name] = {"error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(out, ensure_ascii=False))
'''
    env = os.environ.copy()
    env["ASSESS_PACKAGE_NAMES"] = json.dumps(names)
    try:
        proc = subprocess.run(
            [python, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if proc.returncode == 0 and lines:
            try:
                return json.loads(lines[-1])
            except json.JSONDecodeError:
                pass
        return {
            "_probe_error": {
                "exit_code": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            }
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"_probe_error": {"error": str(exc)}}


def import_probe(python: str, module: str, timeout: int) -> dict[str, Any]:
    code = r'''
import importlib, importlib.util, json
name = __import__("os").environ["ASSESS_IMPORT_NAME"]
spec = importlib.util.find_spec(name)
out = {"module": name, "spec_origin": getattr(spec, "origin", None), "locations": list(getattr(spec, "submodule_search_locations", []) or [])}
try:
    mod = importlib.import_module(name)
    out["ok"] = True
    out["file"] = getattr(mod, "__file__", None)
    out["version"] = getattr(mod, "__version__", None)
except Exception as exc:
    out["ok"] = False
    out["error_type"] = type(exc).__name__
    out["error"] = str(exc)
print(json.dumps(out, ensure_ascii=False))
'''
    env = os.environ.copy()
    env["ASSESS_IMPORT_NAME"] = module
    try:
        proc = subprocess.run(
            [python, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        parsed: Any = None
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if lines:
            try:
                parsed = json.loads(lines[-1])
            except json.JSONDecodeError:
                parsed = None
        return {
            "exit_code": proc.returncode,
            "result": parsed,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def torch_probe(python: str, timeout: int) -> dict[str, Any]:
    code = r'''
import json
out = {}
try:
    import torch
    out["import_ok"] = True
    out["version"] = getattr(torch, "__version__", None)
    out["file"] = getattr(torch, "__file__", None)
    out["version_namespace"] = {k: getattr(torch.version, k, None) for k in dir(torch.version) if not k.startswith("_") and isinstance(getattr(torch.version, k, None), (str, int, float, bool, type(None)))}
    for api_name in ("cuda", "xpu"):
        api = getattr(torch, api_name, None)
        item = {"present": api is not None}
        if api is not None:
            for name in ("is_available", "device_count"):
                fn = getattr(api, name, None)
                if callable(fn):
                    try:
                        item[name] = fn()
                    except Exception as exc:
                        item[name + "_error"] = f"{type(exc).__name__}: {exc}"
            get_name = getattr(api, "get_device_name", None)
            count = item.get("device_count")
            if callable(get_name) and isinstance(count, int):
                names = []
                for idx in range(min(count, 32)):
                    try:
                        names.append(get_name(idx))
                    except Exception as exc:
                        names.append(f"ERROR: {type(exc).__name__}: {exc}")
                item["device_names"] = names
        out[api_name] = item
except Exception as exc:
    out["import_ok"] = False
    out["error_type"] = type(exc).__name__
    out["error"] = str(exc)
print(json.dumps(out, ensure_ascii=False))
'''
    try:
        proc = subprocess.run(
            [python, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        parsed: Any = None
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if lines:
            try:
                parsed = json.loads(lines[-1])
            except json.JSONDecodeError:
                parsed = None
        return {
            "exit_code": proc.returncode,
            "result": parsed,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def collect_tools(timeout: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, candidates in TOOL_COMMANDS.items():
        path = shutil.which(name)
        item: dict[str, Any] = {"path": path}
        if path:
            attempts = []
            for argv in candidates:
                attempt = run_command(argv, timeout)
                attempts.append(attempt)
                if attempt.get("exit_code") == 0:
                    break
            item["attempts"] = attempts
        result[name] = item
    return result


def device_count(torch_data: dict[str, Any]) -> int | None:
    result = torch_data.get("result") if isinstance(torch_data, dict) else None
    if not isinstance(result, dict):
        return None
    counts = []
    for api in ("cuda", "xpu"):
        item = result.get(api)
        if isinstance(item, dict) and isinstance(item.get("device_count"), int):
            counts.append(item["device_count"])
    return max(counts) if counts else None


def readiness(
    *,
    model: dict[str, Any] | None,
    target: dict[str, Any] | None,
    imports: dict[str, Any],
    torch_data: dict[str, Any],
    required_devices: int,
) -> tuple[str, list[dict[str, str]]]:
    issues: list[dict[str, str]] = []

    if model and not model.get("exists"):
        issues.append({"severity": "error", "code": "model_path_missing", "message": "model path does not exist"})
    if target and not target.get("exists"):
        issues.append({"severity": "error", "code": "target_repo_missing", "message": "target repo does not exist"})

    for module, probe in imports.items():
        result = probe.get("result") if isinstance(probe, dict) else None
        if not isinstance(result, dict) or not result.get("ok"):
            issues.append({"severity": "error", "code": f"import_failed:{module}", "message": f"failed to import {module}"})

    torch_result = torch_data.get("result") if isinstance(torch_data, dict) else None
    if not isinstance(torch_result, dict) or not torch_result.get("import_ok"):
        issues.append({"severity": "error", "code": "torch_import_failed", "message": "torch import failed"})
    else:
        count = device_count(torch_data)
        if count is None:
            issues.append({"severity": "warning", "code": "device_count_unknown", "message": "device count could not be determined"})
        elif count < required_devices:
            issues.append({"severity": "error", "code": "insufficient_devices", "message": f"visible devices {count} < required {required_devices}"})
        elif count == 0:
            issues.append({"severity": "error", "code": "no_visible_device", "message": "no visible torch device"})

    if any(item["severity"] == "error" for item in issues):
        return "unavailable", issues
    if issues:
        return "degraded", issues
    return "ready", issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--engine", choices=sorted(IMPORTS_BY_ENGINE), required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-path")
    parser.add_argument("--target-repo")
    parser.add_argument("--upstream-repo")
    parser.add_argument("--required-devices", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--extra-package", action="append", default=[])
    parser.add_argument("--extra-import", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python_path = shutil.which(args.python) or args.python
    packages = list(dict.fromkeys(PACKAGE_NAMES + args.extra_package))
    modules = list(dict.fromkeys(IMPORTS_BY_ENGINE[args.engine] + args.extra_import))

    path_data = {
        "model": stat_path(args.model_path),
        "target_repo": stat_path(args.target_repo),
        "upstream_repo": stat_path(args.upstream_repo),
        "dev_shm": stat_path("/dev/shm"),
    }
    imports = {module: import_probe(python_path, module, args.timeout) for module in modules}
    torch_data = torch_probe(python_path, args.timeout)
    state, issues = readiness(
        model=path_data["model"],
        target=path_data["target_repo"],
        imports=imports,
        torch_data=torch_data,
        required_devices=max(args.required_devices, 1),
    )

    env = {
        key: value
        for key, value in os.environ.items()
        if key in ENV_KEYS or key.startswith(ENV_PREFIXES)
    }
    nodes = sorted({path for pattern in DEVICE_GLOBS for path in glob.glob(pattern)})

    data = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "readiness": state,
        "issues": issues,
        "host": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "memory": meminfo(),
        },
        "python": {
            "requested": args.python,
            "executable": python_path,
            "runtime": python_runtime_info(python_path, args.timeout),
        },
        "engine": args.engine,
        "required_devices": max(args.required_devices, 1),
        "paths": path_data,
        "packages": package_info(python_path, packages, args.timeout),
        "imports": imports,
        "torch_probe": torch_data,
        "visible_device_count": device_count(torch_data),
        "device_nodes": nodes,
        "tools": collect_tools(args.timeout),
        "environment": env,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"readiness={state}")
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
