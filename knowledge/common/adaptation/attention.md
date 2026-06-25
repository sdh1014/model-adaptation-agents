# Attention 能力评估

## 模型事实

- attention 类型；
- hidden size、head count、KV head count、head dim；
- Q/K/V 投影组织和 bias；
- QK norm；
- attention scale；
- causal、sliding、sparse 或 block mask；
- prefill 和 decode 行为；
- cross-attention 或 multimodal attention；
- MLA latent、compressed KV 等特殊结构。

## 引擎与后端检查

- 模型层如何选择 attention backend；
- backend 支持的 dtype、head dim、mask 和 batch 形态；
- prefill、decode、chunked prefill 是否走不同路径；
- P800 是否有专用 kernel，或使用 generic fallback；
- fallback 是否保持相同 scale、mask 和 layout；
- graph/compile 模式是否改变路径。

## 部分支持的典型情况

- BF16 支持，INT8 不支持；
- prefill 支持，decode 不支持；
- 标准 GQA 支持，特殊 head dim 不支持；
- TP=1 支持，TP>1 的 KV head 切分不支持；
- dense attention 支持，sliding/sparse 模式不支持。

## 最小验证

- 单层 attention 输入输出 shape；
- 固定小张量与参考实现比较；
- prefill 与单步 decode；
- 不同 batch/sequence 边界；
- TP 情况下 head 切分。
