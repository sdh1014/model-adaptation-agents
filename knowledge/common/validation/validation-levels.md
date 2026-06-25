# 验证层次

## L1：加载完整性

检查模型类、配置、权重映射、missing/unexpected keys、dtype、量化和 tied weights。

通过标准不能只看进程未崩溃；必须确认没有未解释的缺失权重、随机初始化和静默 fallback。

## L2：局部执行

检查 prefill、首步 decode、logits shape、KV Cache 更新和 batch 维度。

适合定位：

- QKV/MLP shape；
- RoPE 位置；
- KV Cache layout；
- TP 切分；
- MoE routing。

## L3：数值或 Token Parity

优先比较相同输入下的 prefill logits，再比较首 token 和短 greedy generation。

不同 dtype 或 fused kernel 下允许小数值差异，但阈值必须在执行前明确。

## L4：场景行为

覆盖 batch、长上下文、EOS/stop、TP、模型特有能力和服务协议。

## L5：回归

验证适配修改没有破坏共享注册、算子、权重加载或已有模型路径。
