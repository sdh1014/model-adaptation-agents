# vLLM-Kunlun

## 评估边界

分别检查：

1. 上游 vLLM 的模型 registry、model class 和 weight loader；
2. vLLM-Kunlun 的 platform、Attention/KV Cache、MoE、量化和通信；
3. 本地 Python、插件和算子运行时。

上游支持模型不代表 P800 backend 自动支持全部能力；插件仓中没有模型类也不代表模型未支持。

## 实现优先级

```text
复用上游实现
→ 插件注册或最小 override
→ Kunlun backend wrapper
→ 必要时请求修改上游核心
```

## 运行与验证

Runbook `start.sh` 保存当前版本实际可用的完整命令。服务接口适合生成和协议验证；严格 logits parity 通常需要额外离线/测试入口。

## Benchmark

优先使用当前版本提供的 `vllm bench serve`，参数以该环境 `--help` 为准，并启用 JSON 结果保存。
