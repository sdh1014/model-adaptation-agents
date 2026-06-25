# 生成一致性

确定性生成至少固定：

- tokenizer 与 chat template；
- greedy decode；
- 最大输出长度；
- EOS、stop token；
- seed；
- prompt token 序列。

文本相同不代表 logits 一致；文本不同也需要先排除 tokenizer 和停止条件差异。
