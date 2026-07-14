#!/usr/bin/env python3
"""PROTOTYPE ONLY: drive the SwiGLU paper walkthrough in a terminal.

Run interactively:
    python3 work/prototypes/swiglu_walkthrough_tui_prototype.py

Run the primary scenario without interaction:
    python3 work/prototypes/swiglu_walkthrough_tui_prototype.py --demo
"""

from __future__ import annotations

import argparse
import json

from swiglu_walkthrough_state_prototype import (
    advance_primary_path,
    initial_state,
    record_kernel_blocker,
    record_no_real_gap,
    record_repair_failure,
    record_repair_pass,
)


BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"


def render(state: dict, *, clear: bool = True) -> None:
    if clear:
        print("\033[2J\033[H", end="")
    print(f"{BOLD}PROTOTYPE — 特殊 SwiGLU 三样本纸面走查{RESET}")
    print(f"{DIM}T1/T2/T3/D 是 CUDA 实际采集值的占位，不是模型事实。{RESET}\n")
    for name in (
        "status",
        "phase",
        "execution_site",
        "active_operator",
        "max_repair_attempts",
        "attempts_used",
        "capture_session",
        "bundle",
        "last_run",
        "passing_run",
        "source_state",
        "next_action",
        "stop_reason",
    ):
        print(f"{BOLD}{name:17}{RESET} {state[name]}")

    print(f"\n{BOLD}Golden Samples{RESET}")
    print("id          input       output      golden             baseline  latest")
    for sample in state["samples"]:
        print(
            f"{sample['id']:11} {sample['input']:11} {sample['output']:11} "
            f"{sample['golden']:18} {sample['baseline']:9} {sample['latest_repair']}"
        )

    if state["status"] in {"PASS", "BLOCKED", "NEEDS_HUMAN"}:
        print(f"\n{DIM}流程已停止；[r] 重置  [q] 退出{RESET}")
    else:
        print(
            f"\n{BOLD}[n]{RESET} 主路径下一步  "
            f"{BOLD}[f]{RESET} 本轮失败  "
            f"{BOLD}[p]{RESET} 本轮通过  "
            f"{BOLD}[k]{RESET} 需要新 Kernel  "
            f"{BOLD}[a]{RESET} 所有 baseline 通过  "
            f"{BOLD}[r]{RESET} 重置  {BOLD}[q]{RESET} 退出"
        )


def run_demo() -> None:
    state = initial_state()
    snapshots = [state]
    while state["status"] not in {"PASS", "BLOCKED", "NEEDS_HUMAN"}:
        state = advance_primary_path(state)
        snapshots.append(state)
    print(json.dumps(snapshots, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="非交互执行主路径：前四轮失败，第五轮通过",
    )
    args = parser.parse_args()
    if args.demo:
        run_demo()
        return

    state = initial_state()
    while True:
        render(state)
        action = input("> ").strip().lower()
        try:
            if action == "q":
                return
            if action == "r":
                state = initial_state()
            elif action == "n":
                state = advance_primary_path(state)
            elif action == "f":
                state = record_repair_failure(state)
            elif action == "p":
                state = record_repair_pass(state)
            elif action == "k":
                state = record_kernel_blocker(state)
            elif action == "a":
                state = record_no_real_gap(state)
        except ValueError as error:
            input(f"\n{error}\n按回车继续")


if __name__ == "__main__":
    main()
