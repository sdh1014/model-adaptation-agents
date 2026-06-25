#!/usr/bin/env python3
"""Model-level deterministic helpers."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

JSON_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "processor_config.json",
    "preprocessor_config.json",
)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"_read_error": f"{type(exc).__name__}: {exc}"}


def inspect_model(args: argparse.Namespace) -> int:
    root = Path(args.model_path).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"model path is not a directory: {root}")
    data: dict[str, Any] = {
        "model_path": str(root),
        "json": {},
        "weight_indexes": [],
        "python_files": [],
        "weight_files": [],
    }
    for name in JSON_FILES:
        path = root / name
        if path.is_file():
            data["json"][name] = read_json(path)
    for pattern in ("*.safetensors.index.json", "pytorch_model.bin.index.json"):
        for path in sorted(root.glob(pattern)):
            data["weight_indexes"].append({"path": path.name, "content": read_json(path)})
    data["python_files"] = [path.name for path in sorted(root.glob("*.py"))]
    for pattern in ("*.safetensors", "*.bin", "*.pt"):
        data["weight_files"].extend(
            {"path": path.name, "size_bytes": path.stat().st_size}
            for path in sorted(root.glob(pattern))
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def run_reference(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("reference requires a command after --")
    runner = Path(__file__).resolve().parent / "run_bash.py"
    argv = [sys.executable, str(runner), "--run-dir", args.run_dir]
    if args.cwd:
        argv += ["--cwd", args.cwd]
    if args.timeout is not None:
        argv += ["--timeout", str(args.timeout)]
    return subprocess.call([*argv, "--", *command])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model analysis helpers")
    sub = parser.add_subparsers(dest="command_name", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--model-path", required=True)
    inspect.add_argument("--output", required=True)
    inspect.set_defaults(handler=inspect_model)
    reference = sub.add_parser("reference")
    reference.add_argument("--run-dir", required=True)
    reference.add_argument("--cwd")
    reference.add_argument("--timeout", type=float)
    reference.add_argument("command", nargs=argparse.REMAINDER)
    reference.set_defaults(handler=run_reference)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
