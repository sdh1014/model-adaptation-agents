"""PROTOTYPE ONLY: pure state transitions for the SwiGLU walkthrough.

Question: can the existing Migration Spec fields represent one special SwiGLU
candidate from scan through three Golden Samples and the Contract-fixed P800
repair limit, including the required stop paths, without adding another workflow
engine or state file?

The sample dimensions and outcomes below are illustrative. Real values must
come from the one CUDA Capture Session.
"""

from __future__ import annotations

from copy import deepcopy


OPERATOR = "sglang.srt.models.step3p5.step_swiglu_with_limit"
MAX_REPAIR_ATTEMPTS = 5


def initial_state() -> dict:
    """Return the in-memory state shown by the throwaway TUI."""
    return {
        "status": "ACTIVE",
        "phase": "SCAN",
        "execution_site": "SOURCE",
        "active_operator": None,
        "max_repair_attempts": MAX_REPAIR_ATTEMPTS,
        "attempts_used": 0,
        "capture_session": "NOT_STARTED",
        "bundle": "NOT_BUILT",
        "last_run": None,
        "passing_run": None,
        "source_state": "fixed clean baseline",
        "next_action": "完成 target 与 draft 扫描，并确认实际配置命中特殊 SwiGLU 分支",
        "stop_reason": None,
        "samples": [
            {
                "id": "sample-001",
                "input": "[T1, 2D]",
                "output": "[T1, D]",
                "golden": "PENDING",
                "baseline": "PENDING",
                "latest_repair": "PENDING",
            },
            {
                "id": "sample-002",
                "input": "[T2, 2D]",
                "output": "[T2, D]",
                "golden": "PENDING",
                "baseline": "PENDING",
                "latest_repair": "PENDING",
            },
            {
                "id": "sample-003",
                "input": "[T3, 2D]",
                "output": "[T3, D]",
                "golden": "PENDING",
                "baseline": "PENDING",
                "latest_repair": "PENDING",
            },
        ],
    }


def _require(state: dict, condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(
            f"当前状态不允许该动作：{message} "
            f"(status={state['status']}, phase={state['phase']})"
        )


def advance_primary_path(state: dict) -> dict:
    """Advance the illustrative path: four failed repairs, then PASS."""
    current = deepcopy(state)
    _require(current, current["status"] in {"ACTIVE", "WAITING"}, "流程已终止")

    if current["phase"] == "SCAN":
        current.update(
            phase="CUDA_CAPTURE",
            active_operator=None,
            last_run="runs/scan-001",
            next_action="在唯一 CUDA Capture Session 中采集 gap queue",
        )
        return current

    if current["phase"] == "CUDA_CAPTURE":
        current.update(
            status="WAITING",
            phase="HANDOFF",
            execution_site="CUDA",
            capture_session="SEALED",
            bundle="VALID",
            last_run="runs/golden-001",
            next_action="人工把已校验的 Handoff Bundle 复制到 P800",
        )
        for sample in current["samples"]:
            sample["golden"] = "SELF_REPLAY_PASS"
        return current

    if current["phase"] == "HANDOFF":
        current.update(
            status="ACTIVE",
            phase="P800_REPAIR",
            execution_site="P800",
            bundle="VERIFIED_ON_P800",
            last_run="runs/handoff-verify-001",
            next_action="从固定干净基线重放全部 Golden Samples",
        )
        return current

    _require(current, current["phase"] == "P800_REPAIR", "未知 phase")
    if current["active_operator"] is None:
        return record_baseline_gap(current)
    if current["attempts_used"] < MAX_REPAIR_ATTEMPTS - 1:
        return record_repair_failure(current)
    return record_repair_pass(current)


def record_baseline_gap(state: dict) -> dict:
    """Record an illustrative partial baseline failure without using an attempt."""
    current = deepcopy(state)
    _require(
        current,
        current["status"] == "ACTIVE"
        and current["phase"] == "P800_REPAIR"
        and current["active_operator"] is None,
        "必须先完成 P800 交接校验，且 baseline 尚未执行",
    )
    baseline_results = ("PASS", "FAIL", "FAIL")
    for sample, result in zip(current["samples"], baseline_results):
        sample["baseline"] = result
    current.update(
        active_operator=OPERATOR,
        last_run="runs/baseline-001",
        next_action="记录第一条修复假设，并从同一干净基线开始 repair-001",
    )
    return current


def record_repair_failure(state: dict) -> dict:
    """Consume one repair attempt, preserve its Run, and restore the baseline."""
    current = deepcopy(state)
    _require(
        current,
        current["status"] == "ACTIVE"
        and current["phase"] == "P800_REPAIR"
        and current["active_operator"] is not None
        and current["attempts_used"] < MAX_REPAIR_ATTEMPTS,
        "没有可失败的修复轮次",
    )
    attempt = current["attempts_used"] + 1
    current["attempts_used"] = attempt
    current["last_run"] = f"runs/repair-{attempt:03d}"
    current["source_state"] = "fixed clean baseline (failed patch restored)"
    results = ("FAIL", "FAIL", "FAIL") if attempt == 1 else ("PASS", "PASS", "FAIL")
    for sample, result in zip(current["samples"], results):
        sample["latest_repair"] = result

    if attempt == MAX_REPAIR_ATTEMPTS:
        current.update(
            status="BLOCKED",
            next_action="none",
            stop_reason=(
                f"{MAX_REPAIR_ATTEMPTS} 轮修复均未让全部 Golden Samples 通过"
            ),
        )
    else:
        current["next_action"] = (
            f"读取 repair-{attempt:03d} 证据，记录下一条单一假设，"
            f"从同一基线开始 repair-{attempt + 1:03d}"
        )
    return current


def record_repair_pass(state: dict) -> dict:
    """Consume the passing attempt and retain only its patch in the worktree."""
    current = deepcopy(state)
    _require(
        current,
        current["status"] == "ACTIVE"
        and current["phase"] == "P800_REPAIR"
        and current["active_operator"] is not None
        and current["attempts_used"] < MAX_REPAIR_ATTEMPTS,
        "没有可通过的修复轮次",
    )
    attempt = current["attempts_used"] + 1
    run = f"runs/repair-{attempt:03d}"
    current["attempts_used"] = attempt
    current["last_run"] = run
    current["passing_run"] = run
    current["source_state"] = "passing patch retained as the only uncommitted change"
    for sample in current["samples"]:
        sample["latest_repair"] = "PASS"
    current.update(
        status="PASS",
        phase="DONE",
        next_action="none",
        stop_reason=None,
    )
    return current


def record_no_real_gap(state: dict) -> dict:
    """Stop when every captured candidate passes the P800 baseline."""
    current = deepcopy(state)
    _require(
        current,
        current["status"] == "ACTIVE"
        and current["phase"] == "P800_REPAIR"
        and current["active_operator"] is None,
        "只允许在 baseline 选择阶段使用",
    )
    for sample in current["samples"]:
        sample["baseline"] = "PASS"
    current.update(
        status="BLOCKED",
        last_run="runs/baseline-all-candidates",
        next_action="none",
        stop_reason="所有已采集候选在 P800 baseline 均通过，没有真实 correctness gap",
    )
    return current


def record_kernel_blocker(state: dict) -> dict:
    """Stop rather than expanding a repair into new C++ or kernel registration."""
    current = deepcopy(state)
    _require(
        current,
        current["status"] == "ACTIVE"
        and current["phase"] == "P800_REPAIR"
        and current["active_operator"] is not None,
        "必须先确认真实 gap",
    )
    current.update(
        status="BLOCKED",
        source_state="fixed clean baseline",
        next_action="none",
        stop_reason="下一步需要新增 C++、自定义 Kernel 或底层注册，超出 Repair Boundary",
    )
    return current
