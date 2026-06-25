# 正确性失败分类

## 先排除输入不一致

优先核对 tokenizer、chat template、token ids、sampling、revision 和 dtype。输入不一致会伪装成模型实现错误。

## 常见现象与检查方向

| 现象 | 首要检查 |
|---|---|
| 模型加载成功但输出随机 | missing weights、权重映射、lm_head/tied embedding |
| 首 token 就不一致 | config、权重、RoPE、Attention、norm |
| prefill 一致而 decode 漂移 | KV Cache、position ids、decode backend |
| batch 才错误 | padding、sequence metadata、调度、cache slot |
| TP=1 正常、TP>1 错误 | shard 规则、all-reduce/all-gather、head/expert partition |
| 长上下文错误 | RoPE scaling、sliding window、chunked prefill、cache 容量 |
| MoE 偶发错误 | routing、top-k、normalization、expert mapping |
| 文本不同但 logits 接近 | sampling、tokenizer、stop/EOS、浮点 tie |

以上只是检查方向，必须由当前日志和最小区分测试确认。
