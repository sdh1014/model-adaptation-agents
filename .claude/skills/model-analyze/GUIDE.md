# model-analyze 指南

## 首次分析

1. 创建或读取 `tasks/<model>/model.yaml`；
2. 固定模型来源和 revision；
3. 执行：

```bash
python scripts/model.py inspect --model-path <path> --output <run>/model-facts.json
```

4. 阅读配置、权重索引、Tokenizer、Processor 和参考源码；
5. 确认 Attention、位置编码、KV Cache、FFN/MoE、权重组织和模型特有能力；
6. 提炼引擎无关的适配要求；
7. 按报告模板写 `model-analysis.md`。

## 增量分析

`--update` 只调查指定事实。增加 revision，记录新证据、修正事实和受影响能力，不重写无关章节。

## 证据

优先：固定 revision 的本地配置/源码/索引 > 官方实现 > 官方文档/技术报告。关键事实必须有精确文件、字段或代码位置。

状态：

```text
passed   关键事实完整
partial  存在非阻塞未知项
blocked  缺少来源或关键模型事实
```
