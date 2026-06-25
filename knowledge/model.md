# 模型分析知识

## 必须确认

- 模型身份、来源与 revision；
- `model_type`、`architectures`、主干 block；
- Attention、位置编码、KV Cache；
- FFN/MoE、router、shared expert；
- checkpoint 分片、权重命名和 packed mapping；
- Tokenizer、special tokens、chat template、generation config；
- 多模态、MTP、量化等模型特有能力；
- 可用于后续 parity 的参考行为。

## 证据状态

```text
confirmed       来源直接支持
inferred        多个事实推导，必须说明推导
unknown         现有资料无法确认
not_applicable  模型不涉及
```

本地固定 revision 的配置、源码和权重索引优先于 README。模型名称不能替代实际代码证据。
