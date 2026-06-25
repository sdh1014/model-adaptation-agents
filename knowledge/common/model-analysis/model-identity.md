# 模型身份分析

## 检查来源

1. `config.json`
2. 模型卡或官方配置
3. 参考实现中的 config class
4. checkpoint 索引

## 必查事实

- `architectures`
- `model_type`
- dtype
- Transformers 兼容性
- 是否依赖 remote code
- 模型 revision 与权重格式

模型名称只能作为检索线索，不能作为架构事实证据。
