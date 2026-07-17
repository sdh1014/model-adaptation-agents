# 恢复 Step3p5MLP 原始算子边界

Type: task
Status: resolved
Blocked by: 18

## What to build

把特殊 SwiGLU Demo 的 Semantic Operator 边界从抽取出的 helper 恢复为原始模型结构中的 `Step3p5MLP.forward`。扫描仍可下钻判断 CUDA/Kunlun 差异，但最终缺口归到整个 MLP forward，不把内部 `gate_up`、SwiGLU 表达式或 `down_proj` 升级为独立 Demo 算子。

Contract 升到 revision 4，CUDA SGLang 固定到原始提交 `49e384ce9d304648e9959666ecb8ce8cd98d0deb`，SGLang-Kunlun 固定到原始提交 `546ad8c682392922792bbbfe53a8bf575545f118`。删除 helper 源码、helper patch 和固定 helper commit 的要求。

## Acceptance criteria

- Contract revision 4 固定算子 `Step3p5MLP.forward` 和激活条件 `self.limit is not None`。
- 边界输入只有 `x`，输出只有 `output`；依赖同一 checkpoint、TP8 和当前 rank 模型权重。
- Golden Sample 不保存权重、`gate_up`、`gate`、`up` 或其他内部 Tensor。
- 旧 `max_samples_per_operator` 改为语义明确的 `max_shapes_per_operator: 3`。
- `runs/spec-binding-003` 绑定 revision 4 并通过。
- `runs/scan-004` 绑定 revision 4，用 `Step3p5MLP.forward` 取代 helper 算子，保留完整 target-only 算子缺口证据。
- `scan-004` supersede `scan-003`，但 revision 3 的 Spec Binding 和 Scan Run 不被改写。
- Migration Spec、方案设计、Skill、模板和 runbook 都采用原始 MLP 边界。

## Comments

2026-07-17 已完成：Contract revision 4 固定原始 CUDA `49e384ce9d304648e9959666ecb8ce8cd98d0deb` 与 Kunlun `546ad8c682392922792bbbfe53a8bf575545f118`，活动边界为 `Step3p5MLP.forward`、激活条件为 `self.limit is not None`、验证 rank 为 0。`runs/spec-binding-003` 和 `runs/scan-004` 已封存，旧 `spec-binding-002` 与 `scan-003` 摘要保持不变。
