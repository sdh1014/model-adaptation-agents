# vLLM-Kunlun 正确性验证

## 重点

分别验证：

1. vLLM 模型实现和注册；
2. vLLM-Kunlun 平台插件与设备算子；
3. 权重加载、Attention/KV Cache、MoE 和 TP 路径；
4. OpenAI 兼容服务行为。

## 建议 case

- import 与模型 registry；
- 权重 loading summary；
- greedy single request；
- reference logits/token；
- batch；
- 目标 TP；
- 长上下文；
- 适配涉及的 Kunlun 算子回归。

## 失败定位

先区分上游 vLLM 模型逻辑与 Kunlun backend。TP=1/CPU 或参考 backend 可用于缩小范围，但不能替代 P800 最终验证。
