# 原型化单文件 Migration Spec 与状态迁移

Type: prototype
Status: resolved
Blocked by:

## Question

一个最小但足以抵抗上下文压缩的 `migration-spec.md` 应包含哪些固定字段、允许哪些状态迁移，并如何在文件层面区分人工只读的 Contract 与 Agent 可写的 Working State？原型必须覆盖完整扫描、一次性采集、人工交接、P800 有限轮修复，以及 `PASS`、`BLOCKED`、`NEEDS_HUMAN` 等终止状态，同时避免引入额外状态机文件。

## Comments

用户确认原型方向，并补充约束：不要自行制造一批新概念。

## Answer

`migration-spec.md` 保持一个文件、两个所有权区：人工只读的 Contract 与 Agent 可写的 Working State，使用显式注释边界区分。Contract 固定目标、源码与 checkpoint、Demo Closure、Precision Gate、权限、状态迁移和 Recovery protocol；Working State 只保存当前 `status/phase`、唯一下一动作、活动算子、gap queue、一次 CUDA Capture、Handoff、P800 repair 次数及 Run 指针。

状态只使用两个维度：`status = ACTIVE | WAITING | PASS | BLOCKED | NEEDS_HUMAN`，`phase = SCAN | CUDA_CAPTURE | HANDOFF | P800_REPAIR | DONE`。`WAITING` 只表示已约定的人工跨机器复制，避免误用 `NEEDS_HUMAN`。`PASS/BLOCKED/NEEDS_HUMAN` 终止当前 Agent；后两者只有在人更新 Contract revision 并写明决定后才可恢复。P800 baseline replay 不计入 Contract 固定的五轮 repair attempt，且只有 baseline 真实失败的候选才能成为 Demo 的真实 Operator Gap。

为控制复杂度，完整扫描结果和尝试细节留在不可变 Run；Spec 只保存恢复执行所需的摘要与指针。领域词必须复用 `CONTEXT.md`，新增的 `status`、`phase`、`WAITING` 仅是流程字段；失败原因直接写自然语言和证据指针，不新增错误码体系、子状态机或额外状态文件。

经人工确认的纸面原型见 [单文件 Migration Spec 原型](../../../work/prototypes/migration-spec.prototype.md)。
