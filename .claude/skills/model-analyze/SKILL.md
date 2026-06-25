---
name: model-analyze
description: 分析模型架构、权重组织、Tokenizer 和参考行为，生成与推理引擎无关的模型事实基线。
argument-hint: "<model-id> [--model-path PATH] [--reference-repo PATH] [--source URL] [--update QUESTION]"
disable-model-invocation: true
---

# 模型分析

处理参数：`$ARGUMENTS`

## 开始前必须读取

1. [requirements.md](requirements.md)
2. [workflow.md](workflow.md)
3. [model-analysis-template.md](model-analysis-template.md)

## 输入

第一个位置参数是 `<model-id>`。

可选参数：

- `--model-path PATH`：本地模型目录；
- `--reference-repo PATH`：本地参考实现；
- `--source URL`：模型官方来源；
- `--update QUESTION`：只补充一个已发现的分析缺口。

已有任务必须优先读取：

- `tasks/<model-id>/model.yaml`
- `tasks/<model-id>/model-analysis.md`
- `tasks/<model-id>/context.md`，存在时读取。

## 输出

写入：

- `tasks/<model-id>/model.yaml`
- `tasks/<model-id>/model-analysis.md`
- `runs/<model-id>/model-analyze/<timestamp>/model-facts.json`

## 边界

- 不分析 vLLM-Kunlun、SGLang-Kunlun 或 P800 支持情况；
- 不修改任何目标引擎仓库；
- 不生成具体引擎实施计划；
- 不把未确认推断写入 `knowledge/`；
- 不自动执行后续 Skill。

严格按照 `workflow.md` 执行。
