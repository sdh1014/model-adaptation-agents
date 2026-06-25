---
name: adapt-implement
description: 根据已确认的适配评估，在目标仓库中一次实现一个工作项；结合历史命令、日志和知识库定位问题，达到停止条件时生成可供开发者决策的阻塞单。
argument-hint: "<model-id>/<target-id> [--item WI-xxx] [--max-attempts 3] [--decision \"开发者决定\"]"
disable-model-invocation: true
---

# 适配实现

处理 `$ARGUMENTS` 指定的模型目标。

## 执行前必须读取

1. [requirements.md](requirements.md)：输入要求、修改边界和通过标准；
2. [workflow.md](workflow.md)：完整执行步骤；
3. [blocker-policy.md](blocker-policy.md)：失败分类、停止条件和升级规则；
4. [attempt-template.md](attempt-template.md)：单次假设记录格式；
5. [blocker-template.md](blocker-template.md)：需要开发者决策时的阻塞单格式；
6. [outcome-template.json](outcome-template.json)：本次 run 的结果格式。

## 必需输入

- `tasks/<model-id>/model-analysis.md`
- `tasks/<model-id>/targets/<target-id>/target.yaml`
- `tasks/<model-id>/targets/<target-id>/assessment.md`
- `tasks/<model-id>/targets/<target-id>/implementation.md`
- `knowledge/engines/<engine>/implement.md`
- 当前工作项对应的通用适配知识
- 当前工作项此前的 implement runs

## 核心规则

- 一次调用只处理一个工作项；
- 修改前先读取历史命令、日志、失败签名和已否定假设；
- 每轮只验证一个可证伪假设，只做支持该假设的最小修改；
- 一次只修改一个仓库；该仓库必须由当前工作项指定，并且已在 `target.yaml` 中声明为 `target_repo` 或 `upstream_repo`；
- 不安装或升级依赖，不修改模型权重，不自动 commit、push、reset、clean、merge 或 rebase；
- 失败签名只表示可观察现象，不等于根因；
- 没有新增证据时不得重复相同命令或改法；
- 默认最多验证 3 个不同假设；同一签名无新增证据重复 2 次时提前停止；
- 达到停止条件后更新 `implementation.md`，并创建 `blockers/<WI-ID>.md`；
- 不自动处理下一个工作项，也不自动进入验证阶段。

## 环境边界

本阶段不重新做完整环境勘测，只复核执行上下文是否相对 assessment 漂移：Python、import 来源、关键版本、目标仓 HEAD/patch，以及局部测试需要时的 P800 可见性。发现实质漂移时停止代码修改并返回 `/adapt-assess`。

## 输出

更新：

- `tasks/<model-id>/targets/<target-id>/implementation.md`
- 必要时创建或更新 `tasks/<model-id>/targets/<target-id>/blockers/<WI-ID>.md`

创建：

```text
runs/<model-id>/<target-id>/<timestamp>-implement-<item-id>/
```

其中保存仓库快照、执行上下文、历史摘要、每个假设、命令、日志、失败签名、patch、范围检查和最终结果。
