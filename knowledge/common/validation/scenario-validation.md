# 场景验证

## Batch

检查：

- 不同长度请求混合；
- padding/position ids；
- batch 内结果不相互污染；
- 调度后 token 顺序和请求归属。

## Long Context

检查：

- 模型声明长度以内的边界点；
- RoPE scaling/sliding window；
- chunked prefill；
- KV Cache 容量与回收；
- 长输入时的 logits 或关键 token parity。

## EOS 和 Stop

检查 EOS id、stop string/token、最大输出长度和多 token stop。

## Parallelism

至少比较 TP=1 与目标 TP 配置；涉及 EP/DP/PP 时分别确认切分、collective 和输出一致性。

## 模型特有能力

根据模型分析按需增加 MoE routing、多模态输入、MTP、speculative、量化和自定义生成协议。
