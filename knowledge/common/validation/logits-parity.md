# Logits 一致性

后续验证应固定输入、dtype 和执行模式，分别比较：

- prefill 最后位置 logits；
- decode 每一步 logits；
- top-k token 与概率；
- 允许误差及其数据类型依据。

比较失败时先定位首个偏离位置，而不是只比较最终文本。
