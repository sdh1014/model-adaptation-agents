# 架构分析

分析应沿实际 forward 路径进行：

```text
embedding → transformer block → normalization → output head
```

每个 block 至少检查：

- normalization 的位置和类型；
- attention 投影和 head 组织；
- 位置编码；
- dense FFN 或 MoE；
- residual 与门控结构；
- 特殊 cache 或附加预测头。

配置字段和参考实现不一致时，以实际代码路径为准，并记录差异。
