#!/usr/bin/env python3
"""Collect a small, non-invasive execution-context snapshot for adapt-implement."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ENGINE_MODULES = {
    "vllm-kunlun": ["vllm", "vllm_kunlun"],
    "sglang-kunlun": ["sglang"],
}
SAFE_ENV_KEYS = [
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "VLLM_TARGET_DEVICE",
    "CUDA_VISIBLE_DEVICES",
    "XPU_VISIBLE_DEVICES",
]


def module_info(
    name: str,
    target_repo: Path | None,
    package_map: dict[str, list[str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "found": False}
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    if spec is None:
        return result

    result["found"] = True
    result["origin"] = spec.origin
    locations = list(spec.submodule_search_locations or [])
    result["search_locations"] = locations

    distributions = sorted(set(package_map.get(name.split(".", 1)[0], [])))
    result["distributions"] = []
    for dist in distributions:
        try:
            version = metadata.version(dist)
        except metadata.PackageNotFoundError:
            version = None
        result["distributions"].append({"name": dist, "version": version})

    if target_repo is not None:
        candidates = [p for p in [spec.origin, *locations] if p]
        inside = False
        for candidate in candidates:
            try:
                Path(candidate).resolve().relative_to(target_repo)
            except (ValueError, OSError):
                continue
            inside = True
            break
        result["origin_inside_target_repo"] = inside
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--engine", choices=sorted(ENGINE_MODULES))
    parser.add_argument("--module", action="append", default=[])
    parser.add_argument("--target-repo")
    args = parser.parse_args()

    modules: list[str] = []
    if args.engine:
        modules.extend(ENGINE_MODULES[args.engine])
    modules.extend(args.module)
    modules = list(dict.fromkeys(modules))

    target_repo = Path(args.target_repo).expanduser().resolve() if args.target_repo else None
    if hasattr(metadata, "packages_distributions"):
        package_map = metadata.packages_distributions()
    else:
        package_map = {}
    payload = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "prefix": sys.prefix,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "working_directory": os.getcwd(),
        "target_repo": str(target_repo) if target_repo else None,
        "environment": {key: os.environ.get(key) for key in SAFE_ENV_KEYS},
        "modules": [
            module_info(name, target_repo, package_map) for name in modules
        ],
    }

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
