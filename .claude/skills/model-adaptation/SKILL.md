---
name: model-adaptation
description: 按 Migration Spec 启动或恢复 Step-3.7-Flash 到 KLX P800 的模型适配。
argument-hint: "Step-3.7-Flash"
disable-model-invocation: true
---

# Model adaptation

把 `$ARGUMENTS` 原样视为模型名。当前只接受 `Step-3.7-Flash`；参数缺失或不同就说明正确调用是 `/model-adaptation Step-3.7-Flash`，然后停止。

完整读取并严格执行仓库中的 [主 Skill](../../../model-adaptation/SKILL.md)。它是唯一流程定义，不在本文件复制一套规则。

如果 `migration-spec.md` 的唯一 `next_action` 是 CUDA preflight，再完整读取 [CUDA 验证步骤](../../../model-adaptation/references/cuda-capture-validation.md)，只执行其中的 preflight。产出 `result.json` 后报告路径并停止：不要启动正式模型服务，不要消耗唯一 CUDA Capture Session，也不要把候选序列化格式写成已跨端确认。
