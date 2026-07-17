# 按 target-only eager 范围重新封存算子扫描

Type: task
Status: resolved
Blocked by: 17

## What to build

在 Contract revision 3 下，沿 `Step3p7ForConditionalGeneration.forward` 的 target 实际路径重新确认完整 Semantic Operator 清单和 CUDA/Kunlun 缺口。扫描不进入 `Step3p5MTP.forward`，也不把 CUDA Graph 管理逻辑列为模型算子。

复用 `scan-002` 中仍然成立的 target 调用链、两侧源码锚点、checkpoint 配置与特殊 SwiGLU helper 证据，但重新形成不可变的 `runs/scan-003`。不能原地修改 `scan-001` 或 `scan-002`，也不能直接把旧 EAGLE 计数声明为 revision 3 结果。

## Acceptance criteria

- `scan-003` 绑定 Contract revision 3，并明确 `speculative_algorithm: null`、decode/prefill graph backend 为 `disabled`。
- 每个算子只包含 target model path 和从 `Step3p7ForConditionalGeneration.forward` 开始的调用链。
- 完整 operator 列表仍逐项保存 CUDA/Kunlun 锚点、激活条件、边界、状态依赖、可重放/替换判断和 verdict。
- gap queue 与 capture plan 根据 target-only 证据重新确认；特殊 SwiGLU 仍最多采集三个真实 shape。
- `scan-003` 明确 supersede `scan-002`，但不修改任何旧 Run。
- Migration Spec Working State 指向 `scan-003`，返回 `ACTIVE / CUDA_CAPTURE`，下一动作是 revision 3 的 CUDA preflight。
- Ticket 12 在本票据完成后解除扫描阻塞；Ticket 15 只要求 target-only eager 扫描完成。
- 自动化测试验证 target-only 调用链、运行配置、完整计数、gap queue、Spec 指针和旧 Run 未被改写。

## Comments

已完成：

- `runs/scan-003` 绑定 Contract revision 3，只保存从 `Step3p7ForConditionalGeneration.forward` 开始的 target 调用链。
- 重新确认 13 个算子，结果为 `READY=8`、`CAPTURE_REQUIRED=1`、`NEEDS_HUMAN=4`；当前 capture plan 仍只有特殊 SwiGLU，最多三个去重真实 shape。
- `scan-003` 复用并明确标注 `scan-002` 中仍成立的 target 源码证据，同时声明 supersede `scan-002`，未修改旧 Run。
- Migration Spec 已推进到 `ACTIVE / CUDA_CAPTURE`，下一动作是 revision 3 的 CUDA preflight。
- Ticket 12 的扫描前置条件已满足；Ticket 15 后续只消费 target-only eager 结果。
- 自动化测试覆盖绑定、target-only 调用链、计数、gap queue、Spec 指针和历史 Run 摘要。
