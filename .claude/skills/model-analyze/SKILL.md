---
name: model-analyze
description: 分析模型架构、权重、Tokenizer 与参考行为，生成可被多个目标引擎复用的模型事实基线。
argument-hint: "<model-id> [--model-path PATH] [--reference-repo PATH] [--update 问题]"
disable-model-invocation: true
---

# 模型分析

处理 `$ARGUMENTS` 指定的模型。

执行前读取 [GUIDE.md](GUIDE.md) 和 `knowledge/model.md`。

输入：模型目录、官方来源、参考实现，以及已有 `tasks/<model>/model.yaml`。

输出：

```text
tasks/<model>/model.yaml
tasks/<model>/model-analysis.md
runs/<model>/model-analyze/<timestamp>/
```

写报告前读取 `templates/reports/model-analysis.md`。

边界：

- 不判断 vLLM-Kunlun、SGLang-Kunlun 或 P800 支持；
- 不修改目标仓；
- 不启动服务；
- 不把推测写成确认事实；
- 完成后停止。
