# 以 kernel 调用为边界重扫并选择最小 Demo

Type: task
Status: resolved
Blocked by: none

## What to build

把当前 Migration Contract 升级到 revision 5。Contract 不再预选 MLP 或任何
`active_operator`，只固定 Step-3.7 target-only eager 的扫描范围、文本与单图输入、
TP8/BF16 运行条件、最多三个 shape、精度门槛和 Golden Sample 参数保存规则。

新建不可变 `runs/scan-005`，以现有 kernel/device-compute 调用为最小边界，沿两种
输入的真实调用链比较 CUDA 已有实现与 Kunlun 等价路径。Triton 只是 CUDA 实现类型
之一，不能作为入选硬条件。首轮扫描完成后，比较 `topk_sigmoid`、视觉 attention、
Gemma RMSNorm 和其他真实缺口，再从 gap queue 中选择最小 Demo，写入 Working
State 的 `active_operator`。

## Acceptance criteria

- Contract Data 不包含 `operator_boundary`、`demo_input_mode` 或
  `active_operator`。
- `scan_scope` 固定 target-only、kernel-call 粒度，以及
  `text-only` / `single-image` 两种输入模式。
- Sample policy 允许保存当前 kernel 调用直接使用的当前 rank 参数 Tensor，但禁止
 保存完整 checkpoint、module `state_dict` 或无关参数。
- `scan-005` 逐项保存 CUDA/Kunlun 调用证据、现有调用边界、输入/参数/输出、
  replay 判断和 verdict；不能新增自定义算子函数作为捕获边界。
- 缺少同名 Kunlun symbol 不自动算缺口；若 Kunlun 在上层绕开 CUDA kernel 或已有
  等价执行路径，结果必须是 `READY`。
- 候选比较至少覆盖 `sgl_kernel.topk_sigmoid`、Step-3.7 视觉 attention Triton
  kernel、Gemma RMSNorm 和 Kunlun 已绕开的 MoE/text attention kernel。
- `active_operator` 只能在扫描完成后从 `scan-005` 的 gap queue 选择，Contract
  中不得预先写死。
- `scan-004` 与 `spec-binding-003` 保持字节不变；revision 5 使用新的
  `spec-binding-004`。
- 当前 Skill、Spec 模板、方案设计和领域词汇统一使用 kernel 调用边界，并明确旧
  MLP capture adapter 不能消费 revision 5。
- 自动化测试覆盖 Contract 解析、禁止预选算子、候选比较、选择结果、参数规则和历史
  Run 摘要。

## Comments

本票据只完成扫描与 Demo 选择，不把尚未实现的 kernel capture/replay adapter
伪装成可运行。选中算子的 adapter 作为下一张独立票据。

2026-07-17 已完成：Contract 升级到 revision 5，`spec-binding-004` 与
`scan-005` 已封存；扫描比较四个静态缺口候选，并选择
`sgl_kernel.gemma_rmsnorm`。当前 Working State 的唯一下一动作是 Ticket 22，
旧 MLP adapter 明确不可消费 revision 5。
