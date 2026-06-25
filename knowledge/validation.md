# 正确性验证知识

## 验证层次

1. 模型类与权重完整性；
2. prefill、first decode 和 KV cache；
3. logits、token 或确定性生成 parity；
4. batch、stop/EOS 和服务协议；
5. TP、多卡和长上下文；
6. MoE、特殊 RoPE、量化、多模态等模型特有能力。

参考优先级：同权重官方实现 > Transformers/reference > 已验证另一引擎 > Golden 输出 > 性质断言。

状态：

```text
passed   required case 全部有证据并通过
failed   至少一个 required case 确定失败
partial  执行成功但覆盖不足
blocked  参考、环境、实现或模型事实不足
```
