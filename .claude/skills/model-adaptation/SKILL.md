---
name: model-adaptation
description: 按 Migration Spec 启动或恢复 Step-3.7-Flash 到 KLX P800 的模型适配。
argument-hint: "Step-3.7-Flash"
disable-model-invocation: true
---

# Model adaptation

把 `$ARGUMENTS` 原样视为模型名。当前只接受 `Step-3.7-Flash`；参数缺失或不同就说明正确调用是 `/model-adaptation Step-3.7-Flash`，然后停止。

完整读取并严格执行仓库中的 [主 Skill](../../../model-adaptation/SKILL.md)。它是唯一流程定义，不在本文件复制一套规则。

只执行 `migration-spec.md` 当前的唯一 `next_action`。旧 Run 或旧 runbook 已经完成的
动作不得重跑；停止条件完全以主 Skill 和当前 Working State 为准。
