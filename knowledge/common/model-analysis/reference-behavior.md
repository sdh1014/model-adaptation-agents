# 参考行为

模型分析阶段应定义后续可复现的参考行为：

- 固定 tokenizer 和 prompt；
- 固定 dtype、seed 和 decode 参数；
- 区分 prefill logits、decode logits 与最终文本；
- 记录特殊模型能力需要的输入；
- 不把一次随机生成文本当作正确性基线。
