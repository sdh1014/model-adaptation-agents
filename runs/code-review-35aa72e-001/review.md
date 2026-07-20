# 35aa72e 正式代码与 Spec 审查

- 固定点：`03cb3a1451e6d34ce6ec4342954ebd2d8e4aa947`
- 目标：`35aa72ee8cfb3c87851f9ca2771c2738023890c2`
- 范围：`git diff 03cb3a1...35aa72e`

## Standards

1. 硬违规：`candidate.patch` 新增 `apply_gemm1_swiglu_clamp`，生产路径再绕经该
   helper；`kernel_replay.py` 仅凭模块位于 worktree 就把任意 callable 当作
   Kernel Call。它违反 `CONTEXT.md` 对 Kernel Call 的定义：扫描、捕获和重放必须
   保持源码中已有的最小设备调用，不得为边界新增 helper。当前回放证明的是新
   wrapper，不是原 `kunlun_ops.swiglu` 调用。
2. 判断性 smell：`repair_entry/worktree/revision` 及 baseline/repair 分支散落在
   `contracts.py`、`kernel_replay.py`、`replay_compare.py` 和
   `workspace_guard.py`，属于 Data Clumps / Shotgun Surgery。

## Spec

1. P1 缺失：Skill 在首个 PASS 就停止；gap queue 仍有四项 `NOT_PLANNED`，没有
   自动选择下一项。
2. P1 错误：Spec 明确要求不新增重放 helper、保持原 Kernel Call，但候选补丁新增
   wrapper，回放也只执行该 wrapper。
3. P2 扩张：一次内联 clamp 引入通用
   `repair-kernel-call/v1:<module>:<callable>` 协议及大段测试，需求没有要求该
   抽象。
4. P2 错误：状态规则要求 `PASS / DONE`，实际 Working State 是
   `PASS / P800_REPAIR`。

## Decision

三个 shape 的数值 PASS 可作为“`kunlun_ops.swiglu(limit=...)` 能消除差异”的设备
证据保留；`35aa72e` 的实现形态和闭环状态不通过审查。旧 Run 不改写，新 Run 提供
内联替代补丁并等待 P800 重新验证。
