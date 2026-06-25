# SGLang-Kunlun 正确性验证

## 重点

- 模型实现/插件注册；
- 权重加载；
- Attention/KV Cache backend；
- batch scheduler 与请求归属；
- TP/EP；
- OpenAI 兼容接口和原生接口差异。

建议先固定 greedy 和 token ids，再比较 prefill/首 token，最后覆盖 batch、长上下文和目标并行配置。
