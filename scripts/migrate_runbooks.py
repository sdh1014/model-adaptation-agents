#!/usr/bin/env python3
"""Add missing validate/benchmark check templates to existing target runbooks.

Developer-owned files are never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

VALIDATE = '''#!/usr/bin/env bash
set -euo pipefail

source "$CONTROL_ROOT/scripts/validation/lib.sh"
validation_init
validation_mark reference-parity required blocked \\
  "请配置可信参考 oracle 和验证命令"
validation_finish
'''

BENCHMARK = '''#!/usr/bin/env bash
set -euo pipefail

source "$CONTROL_ROOT/scripts/benchmark/lib.sh"
benchmark_init
benchmark_mark serving-default required blocked \\
  "请配置 workload 和 benchmark 命令"
benchmark_finish
'''


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="为现有 Runbook 添加缺失的验证/压测入口")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    root = Path(args.repo_root).resolve()
    tasks = root / "tasks"
    created: list[str] = []
    skipped: list[str] = []
    if not tasks.exists():
        print(json.dumps({"created": [], "skipped": [], "reason": "tasks_missing"}, ensure_ascii=False, indent=2))
        return 0
    for runbook in sorted(tasks.glob("*/targets/*/runbook")):
        checks = runbook / "checks"
        checks.mkdir(parents=True, exist_ok=True)
        for name, content in (("validate.sh", VALIDATE), ("benchmark.sh", BENCHMARK)):
            path = checks / name
            relative = str(path.relative_to(root))
            if path.exists():
                skipped.append(relative)
                continue
            path.write_text(content, encoding="utf-8")
            path.chmod(0o755)
            created.append(relative)
    print(json.dumps({"created": created, "skipped": skipped}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
