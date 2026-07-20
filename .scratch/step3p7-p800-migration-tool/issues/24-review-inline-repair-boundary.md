# Ticket 24：审查并恢复内联 Kernel Call 修复边界

Status: In Progress (SOURCE implementation complete; P800 validation pending)

## Goal

对 `35aa72e` 相对 `03cb3a1` 的代码与 Spec 做正式双轴审查。保留 P800 三个
Golden shape 的数值结果，但删除为了让回放可导入而新增的生产 helper 和通用
`<module>:<callable>` 协议。SwiGLU clamp 修复必须直接写在原
`unquantized_fused_moe_apply_kunlun` 调用位置，候选回放继续调用已有
`kunlun_ops.swiglu` Kernel Call。

## Acceptance

- Standards 与 Spec 审查分别封存，且引用固定点和目标提交。
- 旧 `repair-attempt-1-r5-001` 不改写，只标记为数值证据有效、实现形态被替代。
- 新的建议补丁不增加函数，只在原调用位置传递 `gemm1_clamp_limit`。
- replay config 不再接受或生成 `repair-kernel-call/v1:<module>:<callable>`。
- 候选回放绑定 `candidate.patch`，但 `invocation_target` 仍为
  `kunlun_ops.swiglu`。
- 候选 replay 前后逐字节核对实际 worktree 与 `candidate.patch`；无关 patch
  不能改变 replay 参数装配。
- 定向测试和完整测试通过。

## Blocked by

- None

## Comments

- P800 实机重新应用与数值重放由后续 Run 完成；SOURCE 侧不得把建议补丁写成新的
  硬件 PASS。
- `runs/code-review-35aa72e-001` 已封存双轴审查；
  `runs/inline-repair-correction-r5-001` 保存内联替代补丁；
  `runs/operator-queue-tool-001` 保留为第一版 SOURCE 历史证据，
  `runs/operator-queue-tool-002` 保存复审后的当前源码摘要。
- 当前 SOURCE adapter 只覆盖历史 SwiGLU 示例；revision 6 的实际 operator
  adapters 必须等新 Scan Run 产生后逐项实现，不能由该示例冒充。
