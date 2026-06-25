# 权重加载评估

## 模型事实

确认：

- checkpoint 类型和 shard index；
- 参数命名空间；
- QKV、gate/up 是否 packed；
- tensor shape 和转置规则；
- expert、shared expert、router 权重；
- tied embeddings / LM head；
- 量化权重、scale、zero-point；
- MTP、vision encoder、projector 等附加权重。

## 引擎检查

- loader 是按名称、映射表还是模块递归加载；
- 参数是否在加载前合并或切分；
- TP/EP rank 如何选择 shard；
- missing/unexpected weights 如何处理；
- skip 规则是否会静默丢弃关键权重；
- dtype conversion 是否改变量化或归一化参数；
- 是否有模型专用 `load_weights`。

## 支持判定

仅看到“加载完成”日志不足以判定 supported。至少需要：

- 关键权重全部有映射；
- missing/unexpected 清单可解释；
- packed 和 shard 规则与模型事实一致；
- 局部 forward 或 parity 能进一步验证。

## 常见工作项

- architecture/model prefix 映射；
- packed QKV mapping；
- packed gate/up mapping；
- expert/shared expert mapping；
- tied weight 处理；
- TP/EP shard loader；
- quantized parameter loader。
