from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = REPO_ROOT / "tasks"
RUNS_DIR = REPO_ROOT / "runs"
SCRIPTS_DIR = REPO_ROOT / "scripts"


def task_dir(task: str) -> Path:
    return TASKS_DIR / task


def run_stage_dir(task: str, stage: str) -> Path:
    return RUNS_DIR / task / stage
