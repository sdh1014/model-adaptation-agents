from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import RUNS_DIR, task_dir


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_simple_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def write_simple_yaml(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}: {value}" for key, value in data.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def init_task(
    task: str,
    *,
    model: str = "",
    target_repo: str = "",
    framework: str = "",
    hardware: str = "P800",
) -> Path:
    directory = task_dir(task)
    directory.mkdir(parents=True, exist_ok=False)

    write_simple_yaml(
        directory / "task.yaml",
        {
            "task_id": task,
            "model": model,
            "target_repo": target_repo,
            "framework": framework,
            "hardware": hardware,
            "current_stage": "model",
        },
    )
    (directory / "status.md").write_text(
        "# Status\n\n"
        "stage: model\n"
        "status: ready\n\n"
        "## Latest\n\n"
        "- Waiting for model research.\n",
        encoding="utf-8",
    )
    (directory / "notes.md").write_text(
        "# Notes\n\n"
        "## Scope\n\n\n"
        "## Focus\n\n\n"
        "## References\n\n\n"
        "## Do Not\n\n\n"
        "## Human Decisions\n\n\n"
        "## Rules\n\n"
        "- 人工确认的信息写在这里。\n"
        "- 实验事实写入 `runs/`。\n"
        "- 未验证的原因分析不写入本文件。\n",
        encoding="utf-8",
    )
    (directory / "references.md").write_text(
        "# References\n\n"
        "## Technical Reports\n\n\n"
        "## Upstream PRs\n\n\n"
        "## Local Files\n\n\n"
        "## Target Repo Files\n\n\n"
        "## Reading Priority\n\n\n"
        "## Notes\n\n\n",
        encoding="utf-8",
    )
    reports_dir = directory / "reports"
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "model-research.md").write_text(
        "# Model Research\n\n"
        "## Summary\n\n\n"
        "## Model Architecture\n\n\n"
        "## Key Structures\n\n\n"
        "## Reference Implementations\n\n\n"
        "## Target Gap Analysis\n\n\n"
        "## Missing Inputs\n\n\n",
        encoding="utf-8",
    )
    return directory


def read_task(task: str) -> dict[str, str]:
    directory = task_dir(task)
    return read_simple_yaml(directory / "task.yaml")


def create_run(task: str, *, stage: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    attempt_dir = RUNS_DIR / task / f"{timestamp}-{stage}"
    (attempt_dir / "logs").mkdir(parents=True)
    (attempt_dir / "raw").mkdir()
    write_json(
        attempt_dir / "input.json",
        {
            "task_id": task,
            "stage": stage,
            "created_at": utc_now(),
        },
    )
    return attempt_dir


def write_status(task: str, *, stage: str, status: str, run_dir: Path) -> None:
    directory = task_dir(task)
    task_path = directory / "task.yaml"
    if task_path.exists():
        task_data = read_simple_yaml(task_path)
        task_data["current_stage"] = stage
        write_simple_yaml(task_path, task_data)
    (directory / "status.md").write_text(
        "# Status\n\n"
        f"stage: {stage}\n"
        f"status: {status}\n\n"
        "## Latest\n\n"
        f"- run: {run_dir}\n",
        encoding="utf-8",
    )
