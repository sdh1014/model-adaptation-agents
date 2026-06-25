# KV Cache 评估

## 模型要求

- K/V shape 和 layout；
- KV head 数、head dim 和 dtype；
- paged/block cache；
- sliding window 或 sparse cache；
- MLA latent cache；
- cross-attention cache；
- cache scale 或量化；
- prefix cache、chunked prefill；
- speculative/MTP 额外状态。

## 引擎与 P800 检查

- cache 分配和 block size；
- write/read kernel 的输入约束；
- prefill 与 decode layout 是否一致；
- TP rank 的 KV 分片；
- graph capture 和动态 shape；
- cache dtype 是否与模型及 kernel 一致；
- fallback 是否支持目标 layout。

## 证据

- backend 代码和测试；
- cache 初始化日志；
- 首次 decode 的 shape/算子错误；
- 固定 token 的多步 decode parity。
