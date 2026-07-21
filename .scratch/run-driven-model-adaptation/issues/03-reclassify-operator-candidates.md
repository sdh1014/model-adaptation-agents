# Reclassify Operator Candidates

Status: Done

Spec: `../spec.md`

## Work

- 以固定 SGLang-Kunlun revision 上 Step-3.7 的真实生产调用链为统计口径。
- 将历史 CUDA/Triton symbol 与语义算子分开，保留五个历史条目用于追溯和测试。
- 对已有精确 PyTorch/`forward_native` 语义的条目标注
  `NATIVE_IMPLEMENTATION`，不直接称为算子缺口。
- 将实现类型、Kunlun 路由状态和 P800 验证状态拆开记录。
- 产出源码锚定的算子缺口分析文档，并同步 Contract、Skill、Context、模板和行为测试。

## Comments

- 2026-07-21：用户指出 `GemmaRMSNorm.forward_native` 已提供 PyTorch native 实现，
  要求重新排查 Operator Candidate，并将“已有 native 实现但 Kunlun 路由未接入”与
  “算子实现缺失”分开。
- 本轮只做固定 revision 的 SOURCE 审计，没有执行 P800，因此五项验证状态仍为
  `PENDING`，`confirmed_gap` 均不能写为 true。
