# Checkpoint 分析

重点确认：

- 单文件或多 shard；
- index 文件及参数到 shard 的映射；
- 参数命名规则；
- QKV、gate/up、expert 是否合并；
- tensor parallel 相关布局；
- 量化权重与 scale 元数据；
- tied weights。

只查看文件名不足以确认布局，应结合索引和参考加载逻辑。
