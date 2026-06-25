#!/usr/bin/env python3
"""Extract selected JSON paths into a benchmark sample file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


def load_loose(path: Path) -> Any:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for line in reversed(text.splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise ValueError("输入中没有可解析 JSON")


def get_path(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(path)
    return current


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从 benchmark JSON 提取标准指标")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric", action="append", default=[], help="输出名=JSON.path，可重复")
    args = parser.parse_args(argv)
    try:
        source = load_loose(Path(args.input))
        if not args.metric:
            result = source
        else:
            metrics: dict[str, Any] = {}
            for mapping in args.metric:
                if "=" not in mapping:
                    raise ValueError(f"无效 mapping: {mapping}")
                name, path = mapping.split("=", 1)
                metrics[name] = get_path(source, path)
            result = {"metrics": metrics}
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 64


if __name__ == "__main__":
    raise SystemExit(main())
